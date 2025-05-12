import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares


def load_point_cloud(file_path):
    """点群データをファイルから読み込む（txtとplyに対応）"""
    file_extension = Path(file_path).suffix.lower()

    if file_extension == ".ply":
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            return np.asarray(pcd.points)
        except Exception as e:
            print(f"PLYファイル読み込みエラー: {e}")
            return None

    elif file_extension == ".txt":
        points = []
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                values = line.strip().split()
                if len(values) >= 3:  # ID, X, Y, Z at minimum
                    try:
                        x, y, z = map(float, values[:3])
                        points.append([x, y, z])
                    except ValueError:
                        continue
        return np.array(points)

    else:
        raise ValueError(f"未対応のファイル形式です: {file_extension}")


def distance_to_sphere(params, points):
    """点から球までの距離を計算"""
    center_x, center_y, center_z, radius = params
    translated_points = points - np.array([center_x, center_y, center_z])
    distances = np.sum(translated_points**2, axis=1) - radius**2
    return distances


def distance_to_ellipsoid(params, points):
    """点から楕円体までの距離を計算（球に近い形状を優先）"""
    center_x, center_y, center_z, a, b, c = params
    translated_points = points - np.array([center_x, center_y, center_z])

    # 各軸での正規化距離
    normalized_distances = (
        (translated_points[:, 0] ** 2) / (a**2)
        + (translated_points[:, 1] ** 2) / (b**2)
        + (translated_points[:, 2] ** 2) / (c**2)
    )

    # 球からの逸脱度を計算
    mean_axis = (a + b + c) / 3
    axis_deviation = (
        (abs(a - mean_axis) / mean_axis)
        + (abs(b - mean_axis) / mean_axis)
        + (abs(c - mean_axis) / mean_axis)
    )

    # 強いペナルティを設定（球形状からの逸脱に対して）
    sphericity_penalty = 1.0 * axis_deviation

    # 正規化された距離に球形状へのバイアスを加える
    distances = normalized_distances - 1 + sphericity_penalty

    return distances


def fit_shape(points, initial_params, shape_type="sphere"):
    """最小二乗法でフィッティング"""
    if shape_type == "sphere":
        result = least_squares(distance_to_sphere, initial_params, args=(points,))
        if result.x[3] <= 0:  # 半径が0以下の場合
            return None
    else:  # ellipsoid
        result = least_squares(distance_to_ellipsoid, initial_params, args=(points,))
        if any(axis <= 0 for axis in result.x[3:]):  # いずれかの軸が0以下の場合
            return None

    return result.x


def ransac_shape_single(
    points,
    shape_type="sphere",
    num_iterations=100,
    threshold=0.1,
    min_inliers_ratio=0.3,
):
    """単一の形状を検出するRANSACアルゴリズム（球に近い形状を優先）"""
    best_model = None
    best_inliers = []
    best_inliers_count = 0
    best_sphericity = float("inf")  # 球形状からの逸脱度を追跡
    n_points = len(points)
    min_points_required = 4 if shape_type == "sphere" else 9

    for _ in range(num_iterations):
        if n_points < min_points_required:
            raise ValueError(
                f"{shape_type}フィッティングに必要な点の数が不足しています"
            )

        # ランダムにサンプル点を選択
        sample_indices = random.sample(range(n_points), min_points_required)
        sample_points = points[sample_indices]

        # 初期パラメータの推定
        initial_center = np.mean(sample_points, axis=0)
        if shape_type == "sphere":
            initial_radius = np.mean(
                np.linalg.norm(sample_points - initial_center, axis=1)
            )
            if initial_radius <= 0:
                continue
            initial_params = np.array([*initial_center, initial_radius])
        else:  # ellipsoid
            # 主成分分析(PCA)を使用して軸を推定
            centered_points = sample_points - initial_center
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # 固有値から軸長を推定（より球に近い初期値を設定）
            mean_eigenvalue = np.mean(eigenvalues)
            initial_axes = np.full(3, mean_eigenvalue)
            # わずかな変動を許容（±20%）
            initial_axes = initial_axes * np.random.uniform(0.8, 1.2, 3)
            initial_axes = 2.0 * np.sqrt(initial_axes)

            # 最小軸長の制限
            min_axis_length = np.max(initial_axes) * 0.5  # より大きな最小値
            initial_axes = np.maximum(initial_axes, min_axis_length)

            initial_params = np.array([*initial_center, *initial_axes])

        try:
            # フィッティング
            model_params = fit_shape(sample_points, initial_params, shape_type)
            if model_params is None:
                continue

            if shape_type == "sphere":
                distances = np.abs(distance_to_sphere(model_params, points))
            else:
                distances = np.abs(distance_to_ellipsoid(model_params, points))

            inliers = distances < threshold
            inliers_count = np.sum(inliers)

            # 楕円体の場合、球形状からの逸脱度をチェック
            if shape_type == "ellipsoid":
                axes = model_params[3:]
                mean_axis = np.mean(axes)
                axis_ratios = axes / mean_axis

                # 球形状からの逸脱度を計算（1に近いほど球に近い）
                sphericity = np.sum(np.abs(axis_ratios - 1))

                # 軸比の制限（最大2:1まで）
                max_ratio = np.max(axes) / np.min(axes)
                if max_ratio > 2.0:
                    continue

                # より球に近い形状を優先
                if (
                    inliers_count > best_inliers_count * 0.8
                    and sphericity < best_sphericity
                ):
                    best_model = model_params
                    best_inliers = inliers
                    best_inliers_count = inliers_count
                    best_sphericity = sphericity
                elif (
                    inliers_count > best_inliers_count * 1.2
                ):  # インライア数が大幅に多い場合
                    best_model = model_params
                    best_inliers = inliers
                    best_inliers_count = inliers_count
                    best_sphericity = sphericity
            else:
                if inliers_count > best_inliers_count:
                    best_model = model_params
                    best_inliers = inliers
                    best_inliers_count = inliers_count

        except:
            continue

    if best_inliers_count < n_points * min_inliers_ratio:
        return None, None

    if best_model is not None:
        inlier_points = points[best_inliers]
        final_model = fit_shape(inlier_points, best_model, shape_type)
        if final_model is not None:
            if shape_type == "ellipsoid":
                # 最終的な軸比のチェック
                axes = final_model[3:]
                max_ratio = np.max(axes) / np.min(axes)
                if max_ratio > 2.0:  # より厳格な制限
                    return None, None
            return final_model, best_inliers

    return None, None


def detect_multiple_shapes(
    points, shape_type="sphere", max_shapes=100, min_points=10, **ransac_params
):
    """複数の形状を検出"""
    remaining_points = points.copy()
    shapes = []
    inliers_list = []

    for i in range(max_shapes):
        if len(remaining_points) < min_points:
            break

        model, inliers = ransac_shape_single(
            remaining_points, shape_type, **ransac_params
        )

        if model is None:
            break

        shapes.append(model)
        inliers_list.append(remaining_points[inliers])
        remaining_points = remaining_points[~inliers]

    return shapes, inliers_list


def create_sphere_mesh(center, radius, resolution=20):
    """Open3Dの球メッシュを生成"""
    if radius <= 0:
        print(f"警告: 無効な半径 ({radius}) のため、球を表示できません")
        return None

    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius, resolution=resolution
    )
    mesh_sphere.translate(center)
    # 半透明の材質を設定
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])  # 青色
    return mesh_sphere


def create_ellipsoid_mesh(center, axes, resolution=20):
    """Open3Dの楕円体メッシュを生成"""
    if any(axis <= 0 for axis in axes):
        print(f"警告: 無効な軸長 ({axes}) のため、楕円体を表示できません")
        return None

    # 単位球を作成して変形
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    # スケール変換
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * axes)
    # 平行移動
    mesh.translate(center)
    mesh.paint_uniform_color([0.1, 0.1, 0.7])  # 青色
    return mesh


def visualize_results(points, shapes, inliers_list, shape_type="sphere"):
    """検出された球と点群をOpen3Dで可視化"""
    # 可視化ウィンドウを2つ作成
    original_vis = o3d.visualization.Visualizer()
    result_vis = o3d.visualization.Visualizer()

    # ウィンドウの初期化
    original_vis.create_window(
        window_name="Original Point Cloud", width=800, height=600
    )
    result_vis.create_window(window_name="Detected Spheres", width=800, height=600)

    # 元の点群の表示（左ウィンドウ）
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(points)
    original_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # グレー
    original_vis.add_geometry(original_pcd)

    # 結果の表示（右ウィンドウ）
    # 残りの点群（インライアに含まれない点）をグレーで表示
    used_points = set()
    for inliers in inliers_list:
        used_points.update([tuple(p) for p in inliers])

    remaining_points = [p for p in points if tuple(p) not in used_points]
    if remaining_points:
        remaining_pcd = o3d.geometry.PointCloud()
        remaining_pcd.points = o3d.utility.Vector3dVector(remaining_points)
        remaining_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # グレー
        result_vis.add_geometry(remaining_pcd)

    # 検出された各形状とそのインライアを表示
    colors = plt.cm.rainbow(np.linspace(0, 1, len(shapes)))
    for i, (shape, inliers, color) in enumerate(zip(shapes, inliers_list, colors)):
        # インライアの点群を表示
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(inliers)
        inlier_pcd.paint_uniform_color(color[:3])
        result_vis.add_geometry(inlier_pcd)

        # 形状メッシュを表示
        center = shape[:3]
        if shape_type == "sphere":
            radius = shape[3]
            print(f"球 {i+1}: 中心 = {center}, 半径 = {radius}")
            mesh = create_sphere_mesh(center, radius)
        else:
            axes = shape[3:]
            print(f"楕円体 {i+1}: 中心 = {center}, 軸長 = {axes}")
            mesh = create_ellipsoid_mesh(center, axes)

        if mesh is not None:
            result_vis.add_geometry(mesh)

    # レンダリングオプションの設定
    original_vis.get_render_option().point_size = 1.0
    result_vis.get_render_option().point_size = 1.0
    result_vis.get_render_option().mesh_show_wireframe = True

    # 視点の設定
    original_vis.get_view_control().set_zoom(0.8)
    result_vis.get_view_control().set_zoom(0.8)

    # 可視化の実行
    print("左ウィンドウ: 元の点群")
    print("右ウィンドウ: 検出された球体とインライア")
    print("終了するには、両方のウィンドウを閉じてください")

    while True:
        original_vis.update_renderer()
        result_vis.update_renderer()
        if not original_vis.poll_events() or not result_vis.poll_events():
            break

    original_vis.destroy_window()
    result_vis.destroy_window()


def fit_sphere(points, initial_params):
    """最小二乗法で球フィッティング"""
    result = least_squares(distance_to_sphere, initial_params, args=(points,))

    # フィッティング結果のパラメータをチェック
    if result.x[3] <= 0:  # 半径が0以下の場合
        return None

    return result.x


def ransac_sphere_single(
    points, num_iterations=100, threshold=0.1, min_inliers_ratio=0.3
):
    """単一の球を検出するRANSACアルゴリズム"""
    best_model = None
    best_inliers = []
    best_inliers_count = 0
    n_points = len(points)
    min_points_required = 4  # 球の定義には最低4点必要

    for _ in range(num_iterations):
        if n_points < min_points_required:
            raise ValueError("球フィッティングに必要な点の数が不足しています")

        # ランダムにサンプル点を選択
        sample_indices = random.sample(range(n_points), min_points_required)
        sample_points = points[sample_indices]

        # 初期パラメータの推定
        initial_center = np.mean(sample_points, axis=0)
        initial_radius = np.mean(np.linalg.norm(sample_points - initial_center, axis=1))
        if initial_radius <= 0:  # 初期半径のチェック
            continue
        initial_params = np.array([*initial_center, initial_radius])

        try:
            # 球フィッティング
            model_params = fit_sphere(sample_points, initial_params)
            if model_params is None:  # フィッティング結果が無効な場合
                continue

            distances = np.abs(distance_to_sphere(model_params, points))
            inliers = distances < threshold
            inliers_count = np.sum(inliers)

            if inliers_count > best_inliers_count:
                best_model = model_params
                best_inliers = inliers
                best_inliers_count = inliers_count

        except:
            continue

    if best_inliers_count < n_points * min_inliers_ratio:
        return None, None

    if best_model is not None:
        inlier_points = points[best_inliers]
        final_model = fit_sphere(inlier_points, best_model)
        if final_model is not None:  # 最終的なフィッティング結果をチェック
            return final_model, best_inliers

    return None, None


def save_results(output_path, shapes, inliers_list, points, shape_type="sphere"):
    """結果をファイルに保存"""
    with open(output_path, "w") as f:
        f.write(f"検出された{shape_type}の数: {len(shapes)}\n\n")

        for i, (shape, inliers) in enumerate(zip(shapes, inliers_list)):
            if shape_type == "sphere":
                center_x, center_y, center_z, radius = shape
                f.write(f"球 {i+1}:\n")
                f.write(f"中心座標: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})\n")
                f.write(f"半径: {radius:.3f}\n")
            else:  # ellipsoid
                center_x, center_y, center_z, a, b, c = shape
                f.write(f"楕円体 {i+1}:\n")
                f.write(f"中心座標: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})\n")
                f.write(f"軸長: a={a:.3f}, b={b:.3f}, c={c:.3f}\n")

            f.write(f"インライア数: {len(inliers)}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="点群から複数の球を検出するRANSACプログラム"
    )
    parser.add_argument(
        "input_file", type=str, help="入力の点群ファイルパス (.txt または .ply)"
    )
    parser.add_argument(
        "--output", type=str, default="result.txt", help="結果出力ファイルパス"
    )
    parser.add_argument(
        "--iterations", type=int, default=200, help="RANSACのイテレーション回数"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="インライアー判定の閾値"
    )
    parser.add_argument(
        "--min-inliers-ratio", type=float, default=0.1, help="必要な最小インライア比率"
    )
    parser.add_argument(
        "--max-spheres", type=int, default=100, help="検出する球の最大数"
    )
    parser.add_argument(
        "--shape-type",
        type=str,
        choices=["sphere", "ellipsoid"],
        default="sphere",
        help="フィッティングする形状の種類",
    )
    args = parser.parse_args()

    # 入力ファイルの存在確認
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"エラー: 入力ファイル '{args.input_file}' が見つかりません。")
        return

    # 点群データの読み込み
    try:
        points = load_point_cloud(args.input_file)
        if points is None or len(points) == 0:
            print("エラー: 有効な点群データが読み込めませんでした。")
            return
    except Exception as e:
        print(f"エラー: データ読み込み中にエラーが発生しました: {e}")
        return

    # 複数の形状を検出
    shapes, inliers_list = detect_multiple_shapes(
        points,
        shape_type=args.shape_type,
        max_shapes=args.max_spheres,
        num_iterations=args.iterations,
        threshold=args.threshold,
        min_inliers_ratio=args.min_inliers_ratio,
    )

    # 結果の保存と可視化
    save_results(args.output, shapes, inliers_list, points, args.shape_type)
    visualize_results(points, shapes, inliers_list, args.shape_type)

    print(f"検出された{args.shape_type}の数: {len(shapes)}")
    print(f"結果は '{args.output}' に保存されました。")


if __name__ == "__main__":
    """
    使用例:
    # 球フィッティング
    python ransac.py output.ply --shape-type sphere --max-spheres 100 --iterations 1000 --threshold 0.1 --min-inliers-ratio 0.005

    # 楕円体フィッティング
    python ransac.py output.ply --shape-type ellipsoid --max-spheres 100 --iterations 1000 --threshold 0.1 --min-inliers-ratio 0.005
    """
    main()
