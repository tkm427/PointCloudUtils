import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN


def load_point_cloud(file_path):
    """点群データをPLYまたはテキストファイルから読み込む"""
    file_extension = Path(file_path).suffix.lower()

    if file_extension == ".ply":
        # PLYファイルの読み込み
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        return points
    else:
        # テキストファイルの読み込み
        points = []
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                values = line.strip().split()
                if len(values) >= 3:  # X, Y, Z at minimum
                    try:
                        x, y, z = map(float, values[:3])
                        points.append([x, y, z])
                    except ValueError:
                        continue
        return np.array(points)


def distance_to_ellipsoid(params, points):
    """点から楕円体までの距離を計算"""
    center_x, center_y, center_z, a, b, c = params
    translated_points = points - np.array([center_x, center_y, center_z])
    distances = (
        translated_points[:, 0] ** 2 / a**2
        + translated_points[:, 1] ** 2 / b**2
        + translated_points[:, 2] ** 2 / c**2
        - 1
    )
    return distances


def fit_ellipsoid(points, initial_params):
    """最小二乗法で楕円体フィッティング"""
    result = least_squares(distance_to_ellipsoid, initial_params, args=(points,))
    return result.x


def ransac_ellipsoid_single(
    points, num_iterations=100, threshold=0.1, min_inliers_ratio=0.3
):
    """単一の楕円体を検出するRANSACアルゴリズム"""
    best_model = None
    best_inliers = []
    best_inliers_count = 0
    n_points = len(points)
    min_points_required = 6

    for _ in range(num_iterations):
        if n_points < min_points_required:
            raise ValueError("Not enough points for ellipsoid fitting")

        sample_indices = random.sample(range(n_points), min_points_required)
        sample_points = points[sample_indices]

        initial_center = np.mean(sample_points, axis=0)
        initial_axes = np.std(sample_points, axis=0) * 2
        initial_params = np.concatenate([initial_center, initial_axes])

        try:
            model_params = fit_ellipsoid(sample_points, initial_params)
            distances = np.abs(distance_to_ellipsoid(model_params, points))
            inliers = distances < threshold
            inliers_count = np.sum(inliers)

            if inliers_count > best_inliers_count:
                best_model = model_params
                best_inliers = inliers
                best_inliers_count = inliers_count

        except:
            continue

    if best_model is not None:
        inlier_points = points[best_inliers]
        if len(inlier_points) >= min_points_required:
            final_model = fit_ellipsoid(inlier_points, best_model)
            return final_model, best_inliers

    return None, None


def detect_multiple_ellipsoids(
    points, max_ellipsoids=5, min_points=10, **ransac_params
):
    """複数の楕円体を検出（改善版）"""
    remaining_points = points.copy()
    remaining_indices = np.arange(len(points))
    ellipsoids = []
    inliers_list = []

    # 点群のスケールを計算
    points_range = np.ptp(points, axis=0)
    cluster_scale = np.mean(points_range) * 0.1

    for i in range(max_ellipsoids):
        if len(remaining_points) < min_points:
            break

        # 点群のクラスタリング
        clustering = DBSCAN(eps=cluster_scale, min_samples=min_points).fit(
            remaining_points
        )
        labels = clustering.labels_

        best_model = None
        best_inliers = None
        best_score = 0

        for label in np.unique(labels):
            if label == -1:
                continue

            cluster_points = remaining_points[labels == label]
            if len(cluster_points) < min_points:
                continue

            try:
                model, inliers = ransac_ellipsoid_single(
                    cluster_points, **ransac_params
                )

                if model is not None:
                    distances = np.abs(distance_to_ellipsoid(model, cluster_points))
                    score = np.sum(inliers) / (np.mean(distances[inliers]) + 1e-6)

                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_inliers = labels == label
            except:
                continue

        if best_model is None:
            break

        ellipsoids.append(best_model)
        original_indices = remaining_indices[best_inliers]
        inliers_list.append(points[original_indices])

        mask = ~best_inliers
        remaining_points = remaining_points[mask]
        remaining_indices = remaining_indices[mask]

    return ellipsoids, inliers_list


def visualize_results(points, ellipsoids, inliers_list, output_path=None):
    """検出された楕円体と点群を可視化（Open3Dバージョン）"""
    # Matplotlib での可視化
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 元の点群をグレーで表示
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c="gray",
        alpha=0.2,
        s=1,
        label="Original points",
    )

    # 各楕円体とそのインライアを表示
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ellipsoids)))

    for i, (ellipsoid, inliers, color) in enumerate(
        zip(ellipsoids, inliers_list, colors)
    ):
        # インライアを表示
        ax.scatter(
            inliers[:, 0],
            inliers[:, 1],
            inliers[:, 2],
            c=[color],
            s=2,
            label=f"Ellipsoid {i+1} inliers",
        )

        # 楕円体の表面をメッシュで表示
        center = ellipsoid[:3]
        axes = ellipsoid[3:]
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + axes[0] * np.outer(np.cos(u), np.sin(v))
        y = center[1] + axes[1] * np.outer(np.sin(u), np.sin(v))
        z = center[2] + axes[2] * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color=color, alpha=0.2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    if output_path:
        plt.savefig(output_path)
    plt.show()

    # Open3Dでの可視化
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 点群の表示
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    vis.add_geometry(pcd)

    # 各楕円体のインライアと楕円体を表示
    colors_o3d = plt.cm.rainbow(np.linspace(0, 1, len(ellipsoids)))[:, :3]
    for i, (ellipsoid, inliers, color) in enumerate(
        zip(ellipsoids, inliers_list, colors_o3d)
    ):
        # インライアの表示
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(inliers)
        inlier_pcd.paint_uniform_color(color)
        vis.add_geometry(inlier_pcd)

    # ビューの設定
    vis.get_render_option().point_size = 2
    vis.get_render_option().background_color = np.array([1, 1, 1])
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="点群から複数の楕円体を検出するRANSACプログラム"
    )
    parser.add_argument(
        "input_file", type=str, help="入力の点群ファイルパス (.ply or .txt)"
    )
    parser.add_argument(
        "--output", type=str, default="result.txt", help="結果出力ファイルパス"
    )
    parser.add_argument(
        "--plot", type=str, default="visualization.png", help="可視化結果の保存パス"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="RANSACのイテレーション回数"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.001, help="インライアー判定の閾値"
    )
    parser.add_argument(
        "--min-inliers-ratio",
        type=float,
        default=0.001,
        help="必要な最小インライア比率",
    )
    parser.add_argument(
        "--max-ellipsoids", type=int, default=70, help="検出する楕円体の最大数"
    )
    parser.add_argument(
        "--min-points", type=int, default=10, help="楕円体検出に必要な最小点数"
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
        if len(points) == 0:
            print("エラー: 有効な点群データが読み込めませんでした。")
            return
    except Exception as e:
        print(f"エラー: データ読み込み中にエラーが発生しました: {e}")
        return

    print(f"読み込んだ点群の数: {len(points)}")

    # 点群の正規化
    points_mean = np.mean(points, axis=0)
    points_std = np.std(points, axis=0)
    normalized_points = (points - points_mean) / points_std

    # 楕円体検出
    ellipsoids, inliers_list = detect_multiple_ellipsoids(
        normalized_points,
        max_ellipsoids=args.max_ellipsoids,
        min_points=args.min_points,
        num_iterations=args.iterations,
        threshold=args.threshold,
        min_inliers_ratio=args.min_inliers_ratio,
    )

    # パラメータを元のスケールに戻す
    for i in range(len(ellipsoids)):
        ellipsoids[i][:3] = ellipsoids[i][:3] * points_std + points_mean
        ellipsoids[i][3:] *= points_std

    # 結果の保存
    with open(args.output, "w") as f:
        f.write(f"検出された楕円体の数: {len(ellipsoids)}\n\n")
        for i, (ellipsoid, inliers) in enumerate(zip(ellipsoids, inliers_list)):
            center_x, center_y, center_z, a, b, c = ellipsoid
            f.write(f"楕円体 {i+1}:\n")
            f.write(f"中心座標: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})\n")
            f.write(f"半軸長: a={a:.3f}, b={b:.3f}, c={c:.3f}\n")
            f.write(f"インライア数: {len(inliers)}\n\n")

    # 結果の可視化
    visualize_results(points, ellipsoids, inliers_list, args.plot)

    print(f"検出された楕円体の数: {len(ellipsoids)}")
    print(f"結果は '{args.output}' に保存されました。")
    print(f"可視化結果は '{args.plot}' に保存されました。")


if __name__ == "__main__":
    """
    python ransac_ply.py ./chair_30fps/untitled.ply
    """
    main()
