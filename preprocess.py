import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor


def setup_logger():
    """ロガーの設定"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def load_point_cloud(file_path):
    """点群データを読み込む"""
    file_extension = Path(file_path).suffix.lower()

    if file_extension == ".ply":
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)
    else:  # .txt file
        points = []
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                values = line.strip().split()
                if len(values) >= 3:
                    try:
                        x, y, z = map(float, values[:3])
                        points.append([x, y, z])
                    except ValueError:
                        continue
        points = np.array(points)

    return points


def save_point_cloud(points, file_path):
    """点群データを保存"""
    output_path = Path(file_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if output_path.suffix.lower() == ".ply":
        o3d.io.write_point_cloud(str(output_path), pcd)
    else:  # .txt file
        np.savetxt(output_path, points, fmt="%.6f")


def statistical_outlier_removal(points, nb_neighbors=20, std_ratio=2.0):
    """統計的外れ値除去"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )

    return np.asarray(cl.points)


def radius_outlier_removal(points, nb_points=16, radius=0.05):
    """半径ベースの外れ値除去"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)

    return np.asarray(cl.points)


def lof_outlier_removal(points, n_neighbors=20, contamination=0.1):
    """Local Outlier Factor によるノイズ除去"""
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

    # LOFによる予測（-1: 外れ値, 1: 正常値）
    predictions = lof.fit_predict(points)

    # 正常値のみを返す
    return points[predictions == 1]


def voxel_downsample(points, voxel_size):
    """ボクセルベースのダウンサンプリング"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    return np.asarray(downsampled_pcd.points)


def uniform_downsample(points, every_k_points):
    """一様なダウンサンプリング"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    downsampled_pcd = pcd.uniform_down_sample(every_k_points=every_k_points)

    return np.asarray(downsampled_pcd.points)


def calculate_point_density(points, radius=0.1):
    """各点の局所密度を計算"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # KDTreeの構築
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 各点の近傍点数をカウント
    densities = []
    for i in range(len(points)):
        [k, idx, _] = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        densities.append(k)

    return np.array(densities)


def extract_dense_regions(
    points, eps=0.1, min_samples=10, min_cluster_size=100, extract_densest_only=False
):
    """密度の高い領域を抽出する"""
    # DBSCANクラスタリングを実行
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    # クラスタごとの点数をカウント
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))

    # 十分な大きさのクラスタを選択（ノイズ点(-1)は除外）
    valid_clusters = [
        label
        for label in unique_labels
        if label != -1 and cluster_sizes[label] >= min_cluster_size
    ]

    if extract_densest_only and valid_clusters:
        # 各クラスタの密度を計算
        cluster_densities = {}
        for cluster_label in valid_clusters:
            cluster_mask = labels == cluster_label
            cluster_points = points[cluster_mask]

            # クラスタ内の平均密度を計算
            densities = calculate_point_density(cluster_points, radius=eps)
            avg_density = np.mean(densities)
            cluster_densities[cluster_label] = avg_density

        # 最も密度の高いクラスタを選択
        densest_cluster = max(cluster_densities, key=cluster_densities.get)
        valid_clusters = [densest_cluster]

    # 選択されたクラスタの点を抽出
    mask = np.isin(labels, valid_clusters)
    dense_points = points[mask]
    dense_labels = labels[mask]

    return dense_points, dense_labels


def visualize_point_cloud(points, window_name="Point Cloud"):
    """点群の可視化"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 点群の色を設定（青色）
    pcd.paint_uniform_color([0, 0, 1])

    # ビジュアライザーの作成と設定
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)

    # レンダリングオプションの設定
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([1, 1, 1])  # 白背景

    # ビューの実行
    vis.run()
    vis.destroy_window()


def visualize_point_cloud_with_density(
    points, densities, window_name="Point Cloud Density"
):
    """密度情報付きの点群を可視化"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 密度を色にマッピング
    densities_normalized = (densities - np.min(densities)) / (
        np.max(densities) - np.min(densities)
    )
    colors = plt.cm.jet(densities_normalized)[:, :3]  # RGB値のみ取得
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ビジュアライザーの作成と設定
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)

    # レンダリングオプションの設定
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.array([1, 1, 1])  # 白背景

    # ビューの実行
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="点群のノイズ除去とダウンサンプリング")

    # 入出力オプション
    parser.add_argument("input_file", type=str, help="入力点群ファイル (.ply or .txt)")
    parser.add_argument("output_file", type=str, help="出力点群ファイル (.ply or .txt)")

    # ノイズ除去オプション
    parser.add_argument(
        "--noise-removal",
        type=str,
        choices=["statistical", "radius", "lof", "none"],
        default="none",
        help="ノイズ除去方法の選択",
    )
    parser.add_argument(
        "--nb-neighbors", type=int, default=20, help="統計的除去の近傍点数"
    )
    parser.add_argument(
        "--std-ratio", type=float, default=2.0, help="統計的除去の標準偏差倍率"
    )
    parser.add_argument(
        "--radius", type=float, default=0.05, help="半径ベース除去の範囲"
    )
    parser.add_argument(
        "--min-neighbors", type=int, default=16, help="半径ベース除去の最小近傍点数"
    )
    parser.add_argument(
        "--contamination", type=float, default=0.1, help="LOFのコンタミネーション率"
    )

    # ダウンサンプリングオプション
    parser.add_argument(
        "--downsample",
        type=str,
        choices=["voxel", "uniform", "none"],
        default="none",
        help="ダウンサンプリング方法の選択",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.1, help="ボクセルのサイズ"
    )
    parser.add_argument(
        "--every-k-points", type=int, default=5, help="一様サンプリングの間隔"
    )

    # 密度ベース抽出オプション
    parser.add_argument(
        "--extract-dense", action="store_true", help="密度の高い領域を抽出"
    )
    parser.add_argument(
        "--density-eps", type=float, default=0.1, help="クラスタリングの探索半径"
    )
    parser.add_argument(
        "--density-min-samples",
        type=int,
        default=10,
        help="クラスタを形成するための最小近傍点数",
    )
    parser.add_argument(
        "--density-min-cluster-size",
        type=int,
        default=50,
        help="抽出するクラスタの最小サイズ",
    )
    parser.add_argument(
        "--show-density", action="store_true", help="点群の密度を可視化"
    )
    parser.add_argument(
        "--extract-densest-only",
        action="store_true",
        help="最も密度の高いクラスタのみを抽出",
    )

    # 可視化オプション
    parser.add_argument(
        "--visualize", action="store_true", help="処理前後の点群を可視化"
    )

    args = parser.parse_args()
    logger = setup_logger()

    try:
        # 点群の読み込み
        logger.info(f"点群データを読み込んでいます: {args.input_file}")
        points = load_point_cloud(args.input_file)
        initial_points = len(points)
        logger.info(f"読み込んだ点の数: {initial_points}")

        # 処理前の可視化
        if args.visualize:
            logger.info("処理前の点群を表示します")
            visualize_point_cloud(points, "Original Point Cloud")

        # 密度の可視化と高密度領域の抽出
        if args.show_density:
            logger.info("点群の密度を計算中...")
            densities = calculate_point_density(points, radius=args.density_eps)
            logger.info("密度の可視化を表示します")
            visualize_point_cloud_with_density(points, densities)

        if args.extract_dense:
            logger.info("密度の高い領域を抽出中...")
            points, labels = extract_dense_regions(
                points,
                eps=args.density_eps,
                min_samples=args.density_min_samples,
                min_cluster_size=args.density_min_cluster_size,
                extract_densest_only=args.extract_densest_only,
            )
            logger.info(f"抽出後の点の数: {len(points)}")

            # 高密度領域抽出後の可視化を追加
            if args.visualize:
                if args.extract_densest_only:
                    logger.info("最も密度の高いクラスタを表示します")
                    visualize_point_cloud(points, "Densest Cluster")
                else:
                    logger.info("抽出された高密度領域を表示します")
                    visualize_point_cloud(points, "Dense Regions")

                # クラスタごとに色分けして表示（複数クラスタの場合のみ）
                if len(points) > 0 and not args.extract_densest_only:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)

                    # 抽出された点群のラベルを使用
                    unique_valid_labels = np.unique(labels)
                    if len(unique_valid_labels) > 1:
                        colors = plt.cm.rainbow(
                            np.linspace(0, 1, len(unique_valid_labels))
                        )[:, :3]
                        point_colors = np.zeros((len(points), 3))

                        for i, label in enumerate(unique_valid_labels):
                            cluster_mask = labels == label
                            point_colors[cluster_mask] = colors[i]

                        pcd.colors = o3d.utility.Vector3dVector(point_colors)

                        # ビジュアライザーの作成と設定
                        vis = o3d.visualization.Visualizer()
                        vis.create_window(
                            window_name="Dense Regions (Colored by Cluster)"
                        )
                        vis.add_geometry(pcd)

                        # レンダリングオプションの設定
                        render_option = vis.get_render_option()
                        render_option.point_size = 2.0
                        render_option.background_color = np.array([1, 1, 1])

                        # ビューの実行
                        vis.run()
                        vis.destroy_window()

        # ノイズ除去
        if args.noise_removal != "none":
            if args.noise_removal == "statistical":
                logger.info("統計的外れ値除去を実行中...")
                points = statistical_outlier_removal(
                    points, nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio
                )
            elif args.noise_removal == "radius":
                logger.info("半径ベースの外れ値除去を実行中...")
                points = radius_outlier_removal(
                    points, nb_points=args.min_neighbors, radius=args.radius
                )
            elif args.noise_removal == "lof":
                logger.info("LOFによるノイズ除去を実行中...")
                points = lof_outlier_removal(
                    points,
                    n_neighbors=args.nb_neighbors,
                    contamination=args.contamination,
                )

            after_noise_removal = len(points)
            logger.info(f"ノイズ除去後の点の数: {after_noise_removal}")

        # ダウンサンプリング
        if args.downsample != "none":
            if args.downsample == "voxel":
                logger.info("ボクセルベースのダウンサンプリングを実行中...")
                points = voxel_downsample(points, args.voxel_size)
            elif args.downsample == "uniform":
                logger.info("一様なダウンサンプリングを実行中...")
                points = uniform_downsample(points, args.every_k_points)

        final_points = len(points)
        logger.info(f"最終的な点の数: {final_points}")

        # 処理後の可視化
        if args.visualize:
            logger.info("処理後の点群を表示します")
            visualize_point_cloud(points, "Processed Point Cloud")

        # 結果の保存
        logger.info(f"処理結果を保存しています: {args.output_file}")
        save_point_cloud(points, args.output_file)

        # 処理の要約
        logger.info("\n処理の要約:")
        logger.info(f"- 初期点数: {initial_points}")
        logger.info(f"- 最終点数: {final_points}")
        logger.info(
            f"- 削減率: {((initial_points - final_points) / initial_points * 100):.1f}%"
        )
        # 密度情報の要約（密度抽出を実行した場合）
        if args.extract_dense:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            logger.info(f"- 検出されたクラスタ数: {n_clusters}")

            if args.show_density:
                density_mean = np.mean(densities)
                density_std = np.std(densities)
                logger.info(f"- 平均点密度: {density_mean:.2f}")
                logger.info(f"- 密度標準偏差: {density_std:.2f}")

    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return


if __name__ == "__main__":
    """
    使用例:
    
    1. 基本的な使用:
    python preprocess.py ./chair_30fps/untitled.ply output.ply
    
    2. 密度の可視化と高密度領域の抽出:
    uv run preprocess.py processed_ply/field/output.ply processed_ply/field/output_test.ply \
        --show-density \
        --extract-dense \
        --density-eps 0.2 \
        --density-min-samples 30 \
        --density-min-cluster-size 2000

    2b. 最も密度の高いクラスタのみを抽出:
    uv run preprocess.py processed_ply/field/output.ply processed_ply/field/output_test.ply \
        --extract-dense \
        --extract-densest-only \
        --density-eps 0.1 \
        --density-min-samples 30 \
        --density-min-cluster-size 500 \
        --visualize

    3. ノイズ除去とダウンサンプリングを組み合わせる:
    uv run preprocess.py raw_ply/field/field.ply processed_ply/field/output_down.ply \
        --noise-removal statistical \
        --nb-neighbors 50 \
        --std-ratio 1.0 \
        --downsample voxel \
        --voxel-size 0.01 \
        --visualize
    
    4. 密度ベースの処理と他の処理を組み合わせる:
    python preprocess.py ./chair_30fps/untitled.ply output.ply \
        --extract-dense \
        --density-eps 0.1 \
        --density-min-samples 10 \
        --noise-removal statistical \
        --downsample voxel \
        --visualize

    5. ノイズ除去のみを実行:
    uv run preprocess.py raw_ply/iphone_60fps/iphone_60fps.ply processed_ply/iphone_60fps/output_down.ply \
        --noise-removal statistical \
        --nb-neighbors 200 \
        --std-ratio 0.3  \
        --visualize

    注意：
    メモリ不足回避のため、ノイズ除去、ダウンサンプリングの後に密度ベースの処理を実行することをお勧めします。

    ノイズ除去パラメータ

    nb-neighbors: 20-50（複雑な形状ほど大きく）
    std-ratio: 1.0-2.0（小さいほど厳しい除去）
    contamination (LOF): 0.05-0.2（ノイズの割合を推定）

    密度ベース抽出

    density-eps: 0.03-0.1（点群の密度に応じて調整）
    density-min-samples: 10-30
    density-min-cluster-size: 500-2000（保持したい部位のサイズ）

    ダウンサンプリング

    voxel-size: 0.01-0.05（小さいほど詳細保持、大きいほど軽量化）
    """
    main()
