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


def fit_sphere(points, initial_params):
    """最小二乗法で球フィッティング"""
    result = least_squares(distance_to_sphere, initial_params, args=(points,))
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
        initial_params = np.array([*initial_center, initial_radius])

        try:
            # 球フィッティング
            model_params = fit_sphere(sample_points, initial_params)
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
        return final_model, best_inliers

    return None, None


def detect_multiple_spheres(points, max_spheres=100, min_points=10, **ransac_params):
    """複数の球を検出"""
    remaining_points = points.copy()
    spheres = []
    inliers_list = []

    for i in range(max_spheres):
        if len(remaining_points) < min_points:
            break

        model, inliers = ransac_sphere_single(remaining_points, **ransac_params)

        if model is None:
            break

        spheres.append(model)
        inliers_list.append(remaining_points[inliers])
        remaining_points = remaining_points[~inliers]

    return spheres, inliers_list


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


def visualize_results(points, spheres, inliers_list):
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

    # 検出された各球とそのインライアを表示
    colors = plt.cm.rainbow(np.linspace(0, 1, len(spheres)))
    for i, (sphere, inliers, color) in enumerate(zip(spheres, inliers_list, colors)):
        # インライアの点群を表示
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(inliers)
        inlier_pcd.paint_uniform_color(color[:3])  # RGBのみ使用
        result_vis.add_geometry(inlier_pcd)

        # 球メッシュを表示
        center = sphere[:3]
        radius = sphere[3]
        print(f"球 {i+1}: 中心 = {center}, 半径 = {radius}")  # デバッグ出力

        mesh_sphere = create_sphere_mesh(center, radius)
        if mesh_sphere is not None:
            result_vis.add_geometry(mesh_sphere)

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


def save_results(output_path, spheres, inliers_list, points):
    """結果をファイルに保存"""
    with open(output_path, "w") as f:
        f.write(f"検出された球の数: {len(spheres)}\n\n")

        for i, (sphere, inliers) in enumerate(zip(spheres, inliers_list)):
            center_x, center_y, center_z, radius = sphere
            f.write(f"球 {i+1}:\n")
            f.write(f"中心座標: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})\n")
            f.write(f"半径: {radius:.3f}\n")
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

    # 複数の球を検出
    spheres, inliers_list = detect_multiple_spheres(
        points,
        max_spheres=args.max_spheres,
        num_iterations=args.iterations,
        threshold=args.threshold,
        min_inliers_ratio=args.min_inliers_ratio,
    )

    # 結果の保存
    save_results(args.output, spheres, inliers_list, points)

    # 結果の可視化
    visualize_results(points, spheres, inliers_list)

    print(f"検出された球の数: {len(spheres)}")
    print(f"結果は '{args.output}' に保存されました。")


if __name__ == "__main__":
    """
    使用例:
    python ransac.py output.ply --max-spheres 100 --iterations 1000 --threshold 0.07 --min-inliers-ratio 0.005
    python sphere_ransac.py points3D.txt --max-spheres 100 --iterations 200 --threshold 0.15 --min-inliers-ratio 0.25
    """
    main()
