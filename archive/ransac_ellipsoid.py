import argparse
import random
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares


def load_point_cloud(file_path):
    """点群データをファイルから読み込む"""
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

    if best_inliers_count < n_points * min_inliers_ratio:
        return None, None

    if best_model is not None:
        inlier_points = points[best_inliers]
        final_model = fit_ellipsoid(inlier_points, best_model)
        return final_model, best_inliers

    return None, None


def detect_multiple_ellipsoids(
    points, max_ellipsoids=100, min_points=10, **ransac_params
):
    """複数の楕円体を検出"""
    remaining_points = points.copy()
    ellipsoids = []
    inliers_list = []

    for i in range(max_ellipsoids):
        if len(remaining_points) < min_points:
            break

        model, inliers = ransac_ellipsoid_single(remaining_points, **ransac_params)

        if model is None:
            break

        # 検出された楕円体のパラメータと対応する点を保存
        ellipsoids.append(model)
        inliers_list.append(remaining_points[inliers])

        # インライアを除去して次の楕円体検出に備える
        remaining_points = remaining_points[~inliers]

    return ellipsoids, inliers_list


def create_ellipsoid_mesh(center, axes, resolution=20):
    """楕円体の描画用メッシュを生成"""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = center[0] + axes[0] * np.outer(np.cos(u), np.sin(v))
    y = center[1] + axes[1] * np.outer(np.sin(u), np.sin(v))
    z = center[2] + axes[2] * np.outer(np.ones_like(u), np.cos(v))

    return x, y, z


def visualize_results(points, ellipsoids, inliers_list, output_path=None):
    """検出された楕円体と点群を可視化"""
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

        # 楕円体を表示
        center = ellipsoid[:3]
        axes = ellipsoid[3:]
        x, y, z = create_ellipsoid_mesh(center, axes)
        ax.plot_surface(x, y, z, color=color, alpha=0.2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    if output_path:
        plt.savefig(output_path)
    plt.show()


def save_results(output_path, ellipsoids, inliers_list, points):
    """結果をファイルに保存"""
    with open(output_path, "w") as f:
        f.write(f"検出された楕円体の数: {len(ellipsoids)}\n\n")

        for i, (ellipsoid, inliers) in enumerate(zip(ellipsoids, inliers_list)):
            center_x, center_y, center_z, a, b, c = ellipsoid
            f.write(f"楕円体 {i+1}:\n")
            f.write(f"中心座標: ({center_x:.3f}, {center_y:.3f}, {center_z:.3f})\n")
            f.write(f"半軸長: a={a:.3f}, b={b:.3f}, c={c:.3f}\n")
            f.write(f"インライア数: {len(inliers)}\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="点群から複数の楕円体を検出するRANSACプログラム"
    )
    parser.add_argument("input_file", type=str, help="入力の点群ファイルパス")
    parser.add_argument(
        "--output", type=str, default="result.txt", help="結果出力ファイルパス"
    )
    parser.add_argument(
        "--plot", type=str, default="visualization.png", help="可視化結果の保存パス"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000, help="RANSACのイテレーション回数"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0001, help="インライアー判定の閾値"
    )
    parser.add_argument(
        "--min-inliers-ratio",
        type=float,
        default=0.0005,
        help="必要な最小インライア比率",
    )
    parser.add_argument(
        "--max-ellipsoids", type=int, default=100, help="検出する楕円体の最大数"
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

    # 複数の楕円体を検出
    ellipsoids, inliers_list = detect_multiple_ellipsoids(
        points,
        max_ellipsoids=args.max_ellipsoids,
        num_iterations=args.iterations,
        threshold=args.threshold,
        min_inliers_ratio=args.min_inliers_ratio,
    )

    # 結果の保存
    save_results(args.output, ellipsoids, inliers_list, points)

    # 結果の可視化と保存
    visualize_results(points, ellipsoids, inliers_list, args.plot)

    print(f"検出された楕円体の数: {len(ellipsoids)}")
    print(f"結果は '{args.output}' に保存されました。")
    print(f"可視化結果は '{args.plot}' に保存されました。")


if __name__ == "__main__":
    """
    python ransac.py ./chair_30fps/sfm-output/points3D.txt\
          --max-ellipsoids 100 --iterations 200 --threshold 0.15 --min-inliers-ratio 0.25
    """
    main()
