#!/usr/bin/env python3
"""
重心点の最適化結果を用いて、不要なtxtファイルの点群を色付けして可視化するスクリプト
"""

import glob
import json
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

# optimize.pyから必要なクラスをインポート
sys.path.append("/Users/yamasakit3a/Documents/lab/ply")
from optimize import GrapeThinningSystem


class ThinningVisualizationSystem:
    """摘粒結果可視化システム"""

    def __init__(self, input_folder: str):
        self.input_folder = input_folder
        self.txt_files = []
        self.centroids = []
        self.file_mapping = {}  # centroid index -> file path

    def load_txt_files_and_centroids(self) -> bool:
        """txtファイルを読み込み、重心を計算"""
        # txtファイルのパターン
        txt_pattern = os.path.join(self.input_folder, "*.txt")
        txt_files = glob.glob(txt_pattern)

        if not txt_files:
            print(f"No txt files found in {self.input_folder}")
            return False

        # ファイルをソートして最後のファイルを除外（calculate_centroids.pyと同じ処理）
        sorted_files = sorted(txt_files)[:-1]

        print(f"Found {len(txt_files)} txt files")
        print(f"Processing {len(sorted_files)} files (excluding the last one)")

        centroids = []

        # 各txtファイルを処理
        for i, txt_file in enumerate(sorted_files):
            print(f"Processing: {os.path.basename(txt_file)}")

            # 点群データを読み込み
            points = self._read_point_cloud_txt(txt_file)

            if points is not None:
                # 重心を計算
                centroid = np.mean(points, axis=0)
                centroids.append(centroid)
                self.file_mapping[i] = txt_file
                x, y, z = centroid
                print(
                    f"  Points: {len(points)}, "
                    f"Centroid: ({x:.3f}, {y:.3f}, {z:.3f})"
                )
            else:
                print(f"  Failed to read file: {txt_file}")

        self.txt_files = sorted_files
        self.centroids = centroids

        return True

    def _read_point_cloud_txt(self, file_path: str) -> Optional[np.ndarray]:
        """txtファイルから点群データを読み込む"""
        try:
            # ヘッダー行をスキップしてデータを読み込み
            data = np.loadtxt(file_path, skiprows=1)

            # X, Y, Z座標のみを抽出（最初の3列）
            points = data[:, :3]

            return points
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def run_optimization(
        self,
        target_count: int = 36,
        max_generations: int = 100,
        population_size: int = 50,
        output_folder: Optional[str] = None,
    ) -> Tuple[List[int], List[int], dict]:
        """重心点に対して最適化を実行し、残すべき点と除去すべき点のインデックスを返す"""
        if not self.centroids:
            msg = "No centroids available. " "Run load_txt_files_and_centroids first."
            raise ValueError(msg)

        # 重心点をGrapeオブジェクトに変換
        from optimize import Grape

        grapes = []
        for i, centroid in enumerate(self.centroids):
            grapes.append(Grape(id=i, position=(centroid[0], centroid[1], centroid[2])))

        # 摘粒システムの初期化
        system = GrapeThinningSystem(grapes, target_count=target_count)

        print(f"\n重心点の最適化開始")
        print(f"重心点の数: {len(grapes)}")
        print(f"目標残し数: {target_count}")
        print("=" * 50)

        # 手法1: 距離の近い実を順に間引く
        method1_result = system.method1_nearest_neighbor_removal()
        method1_ids = {grape.id for grape in method1_result}

        # 手法2: 組合せ最適化
        method2_result = system.method2_combinatorial_optimization(
            max_generations=max_generations, population_size=population_size
        )
        method2_ids = {grape.id for grape in method2_result}

        # 統計情報の表示
        method1_stats = system.calculate_statistics(method1_result)
        method2_stats = system.calculate_statistics(method2_result)

        print("\n" + "=" * 50)
        print("最適化結果:")
        print(
            f"手法1 - 残し数: {method1_stats['count']}, 最小距離: {method1_stats['min_distance']:.3f}"
        )
        print(
            f"手法2 - 残し数: {method2_stats['count']}, 最小距離: {method2_stats['min_distance']:.3f}"
        )

        # より良い結果を選択
        if method2_stats["min_distance"] > method1_stats["min_distance"]:
            print("→ 手法2（組合せ最適化）を採用")
            keep_indices = list(method2_ids)
            selected_method = "手法2（組合せ最適化）"
        else:
            print("→ 手法1（近接実除去）を採用")
            keep_indices = list(method1_ids)
            selected_method = "手法1（近接実除去）"

        # 除去対象のインデックス
        all_indices = set(range(len(self.centroids)))
        remove_indices = list(all_indices - set(keep_indices))

        # 実験ログを保存
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            self.save_experiment_log(
                keep_indices=keep_indices,
                remove_indices=remove_indices,
                method1_stats=method1_stats,
                method2_stats=method2_stats,
                selected_method=selected_method,
                target_count=target_count,
                max_generations=max_generations,
                population_size=population_size,
                output_folder=output_folder,
            )

        experiment_info = {
            "method1_stats": method1_stats,
            "method2_stats": method2_stats,
            "selected_method": selected_method,
            "target_count": target_count,
            "max_generations": max_generations,
            "population_size": population_size,
        }

        return keep_indices, remove_indices, experiment_info

    def load_all_point_clouds(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """すべてのtxtファイルの点群を読み込み、統合する（最後のファイルも含む）"""
        # 可視化では全てのtxtファイルを含める
        txt_pattern = os.path.join(self.input_folder, "*.txt")
        all_txt_files = sorted(glob.glob(txt_pattern))  # 最後のファイルも含める

        all_points = []
        file_labels = []
        colors = []

        for i, txt_file in enumerate(all_txt_files):
            points = self._read_point_cloud_txt(txt_file)
            if points is not None:
                all_points.append(points)
                # ファイル番号をラベルとして使用
                file_labels.extend([i] * len(points))

        if all_points:
            combined_points = np.vstack(all_points)
            file_labels = np.array(file_labels)

            # ファイルマッピングを更新（可視化用）
            self.all_files_for_viz = all_txt_files

            return combined_points, file_labels, colors
        else:
            return np.array([]), np.array([]), []

    def visualize_thinning_result(
        self,
        keep_indices: List[int],
        remove_indices: List[int],
        save_path: Optional[str] = None,
        show_centroids: bool = True,
    ):
        """摘粒結果をOpen3Dで可視化"""
        print("\n点群データを読み込み中...")
        all_points, file_labels, _ = self.load_all_point_clouds()

        if len(all_points) == 0:
            print("No valid point clouds found")
            return

        print(f"総点数: {len(all_points)}")
        print(f"ファイル数: {len(self.all_files_for_viz)}")

        # Open3Dの点群オブジェクトを作成
        geometries = []

        # 除去対象ファイルのセット
        removed_file_indices = set(remove_indices)

        # 残すファイルの点群（グレー）
        keep_points = []
        for i in range(len(self.all_files_for_viz)):
            if i not in removed_file_indices and i < len(self.txt_files):
                # 重心計算に使用されたファイルのみ判定対象
                mask = file_labels == i
                if np.any(mask):
                    points_subset = all_points[mask]
                    keep_points.extend(points_subset)
            elif i >= len(self.txt_files):
                # 最後のファイル（重心計算に使用されていない）はグレーで表示
                mask = file_labels == i
                if np.any(mask):
                    points_subset = all_points[mask]
                    keep_points.extend(points_subset)

        if keep_points:
            keep_pcd = o3d.geometry.PointCloud()
            keep_pcd.points = o3d.utility.Vector3dVector(np.array(keep_points))
            # グレー色を設定
            keep_pcd.paint_uniform_color([0.7, 0.7, 0.7])
            geometries.append(keep_pcd)

        # 除去対象ファイルの点群（赤色）
        remove_points = []
        for i in removed_file_indices:
            if i < len(self.txt_files):  # 重心計算に使用されたファイルのみ
                mask = file_labels == i
                if np.any(mask):
                    points_subset = all_points[mask]
                    remove_points.extend(points_subset)

        if remove_points:
            remove_pcd = o3d.geometry.PointCloud()
            remove_pcd.points = o3d.utility.Vector3dVector(np.array(remove_points))
            # 赤色を設定
            remove_pcd.paint_uniform_color([1.0, 0.0, 0.0])
            geometries.append(remove_pcd)

        # 重心点の表示（オプション）
        if show_centroids and self.centroids:
            centroids_array = np.array(self.centroids)

            # 残す重心点（緑の球体）
            if len(keep_indices) > 0:
                keep_centroids = centroids_array[keep_indices]
                for centroid in keep_centroids:
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                    sphere.translate(centroid)
                    sphere.paint_uniform_color([0.0, 1.0, 0.0])  # 緑色
                    geometries.append(sphere)

            # 除去する重心点（赤の立方体）
            if len(remove_indices) > 0:
                remove_centroids = centroids_array[remove_indices]
                for centroid in remove_centroids:
                    cube = o3d.geometry.TriangleMesh.create_box(
                        width=0.03, height=0.03, depth=0.03
                    )
                    cube.translate(centroid - np.array([0.015, 0.015, 0.015]))
                    cube.paint_uniform_color([1.0, 0.0, 0.0])  # 赤色
                    geometries.append(cube)

        # 座標軸を追加
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        geometries.append(coordinate_frame)

        print("\nOpen3D可視化ウィンドウを開いています...")
        print("統計情報:")
        print(f"  総ファイル数（表示用）: {len(self.all_files_for_viz)}")
        print(f"  重心計算対象ファイル数: {len(self.txt_files)}")
        print(f"  残すファイル数: {len(keep_indices)}")
        print(f"  除去ファイル数: {len(remove_indices)}")
        print(f"  総点数: {len(all_points):,}")

        print("\n除去対象ファイル:")
        for i, idx in enumerate(remove_indices[:10]):
            if idx < len(self.txt_files):
                filename = os.path.basename(self.txt_files[idx])
                print(f"  {filename}")
        if len(remove_indices) > 10:
            print(f"  ... and {len(remove_indices) - 10} more files")

        # 最後のファイルの情報も表示
        if hasattr(self, "all_files_for_viz") and len(self.all_files_for_viz) > len(
            self.txt_files
        ):
            last_file = os.path.basename(self.all_files_for_viz[-1])
            print(f"\n最後のファイル（グレー表示）: {last_file}")

        # 可視化ウィンドウの設定
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="摘粒結果可視化 (グレー: 残すファイル, 赤: 除去対象ファイル)",
            width=1200,
            height=800,
        )

        # ジオメトリを追加
        for geometry in geometries:
            vis.add_geometry(geometry)

        # レンダリングオプションを設定
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([0.1, 0.1, 0.1])  # 暗いグレー背景
        render_option.point_size = 3.0

        # カメラ位置を設定
        view_control = vis.get_view_control()
        view_control.set_front([0.0, 0.0, -1.0])
        view_control.set_lookat([0.0, 0.0, 0.0])
        view_control.set_up([0.0, 1.0, 0.0])
        view_control.set_zoom(0.8)

        print("\n操作方法:")
        print("  - マウスドラッグ: 回転")
        print("  - Shift + マウスドラッグ: 平行移動")
        print("  - マウスホイール: ズーム")
        print("  - ESCキー: ウィンドウを閉じる")

        # 可視化実行
        vis.run()

        # スクリーンショットの保存
        if save_path:
            vis.capture_screen_image(save_path)
            print(f"スクリーンショットを保存しました: {save_path}")

        vis.destroy_window()

        # 色付き点群の保存
        self._save_colored_point_clouds(keep_points, remove_points, save_path)

    def create_summary_visualization(
        self,
        keep_indices: List[int],
        remove_indices: List[int],
        save_path: Optional[str] = None,
    ):
        """重心点のみのサマリー可視化をOpen3Dで作成"""
        if not self.centroids:
            print("No centroids available for summary visualization")
            return

        print("\n重心点のサマリー可視化を作成中...")

        geometries = []
        centroids_array = np.array(self.centroids)

        # 残す重心点（緑の球体）
        if len(keep_indices) > 0:
            keep_centroids = centroids_array[keep_indices]
            for i, centroid in enumerate(keep_centroids):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                sphere.translate(centroid)
                sphere.paint_uniform_color([0.0, 0.8, 0.0])  # 緑色
                geometries.append(sphere)

        # 除去する重心点（赤の立方体）
        if len(remove_indices) > 0:
            remove_centroids = centroids_array[remove_indices]
            for i, centroid in enumerate(remove_centroids):
                cube = o3d.geometry.TriangleMesh.create_box(
                    width=0.04, height=0.04, depth=0.04
                )
                cube.translate(centroid - np.array([0.02, 0.02, 0.02]))
                cube.paint_uniform_color([0.8, 0.0, 0.0])  # 赤色
                geometries.append(cube)

        # 座標軸を追加
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        geometries.append(coordinate_frame)

        # 可視化
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="重心点分布 (緑: 残す, 赤: 除去)", width=800, height=600
        )

        for geometry in geometries:
            vis.add_geometry(geometry)

        # レンダリングオプション
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([0.05, 0.05, 0.05])

        # カメラ設定
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)

        vis.run()

        if save_path:
            summary_path = save_path.replace(".png", "_summary.png")
            vis.capture_screen_image(summary_path)
            print(f"サマリー画像を保存しました: {summary_path}")

        vis.destroy_window()

    def save_colored_point_clouds(
        self,
        keep_indices: List[int],
        remove_indices: List[int],
        output_folder: str,
    ):
        """色付きの点群を保存"""
        print("\n色付き点群データを保存中...")
        all_points, file_labels, _ = self.load_all_point_clouds()

        if len(all_points) == 0:
            print("No valid point clouds found for saving")
            return

        # 出力フォルダを作成
        os.makedirs(output_folder, exist_ok=True)

        # 除去対象ファイルのセット
        removed_file_indices = set(remove_indices)

        # 残すファイルの点群（グレー）
        keep_points = []
        for i in range(len(self.all_files_for_viz)):
            if i not in removed_file_indices and i < len(self.txt_files):
                # 重心計算に使用されたファイルのみ判定対象
                mask = file_labels == i
                if np.any(mask):
                    points_subset = all_points[mask]
                    keep_points.extend(points_subset)
            elif i >= len(self.txt_files):
                # 最後のファイル（重心計算に使用されていない）はグレーで表示
                mask = file_labels == i
                if np.any(mask):
                    points_subset = all_points[mask]
                    keep_points.extend(points_subset)

        # 除去対象ファイルの点群（赤色）
        remove_points = []
        for i in removed_file_indices:
            if i < len(self.txt_files):  # 重心計算に使用されたファイルのみ
                mask = file_labels == i
                if np.any(mask):
                    points_subset = all_points[mask]
                    remove_points.extend(points_subset)

        # 1. 残す点群を保存（グレー）
        if keep_points:
            keep_pcd = o3d.geometry.PointCloud()
            keep_pcd.points = o3d.utility.Vector3dVector(np.array(keep_points))
            keep_pcd.paint_uniform_color([0.7, 0.7, 0.7])

            keep_ply_path = os.path.join(output_folder, "keep_files_points.ply")
            keep_xyz_path = os.path.join(output_folder, "keep_files_points.xyz")

            o3d.io.write_point_cloud(keep_ply_path, keep_pcd)

            # XYZ形式でも保存
            with open(keep_xyz_path, "w") as f:
                f.write("X Y Z R G B\n")
                for point in keep_points:
                    line = f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}"
                    f.write(f"{line} 179 179 179\n")  # RGB値はグレー

            print(f"残すファイルの点群を保存: {keep_ply_path}")
            print(f"残すファイルの点群を保存: {keep_xyz_path}")
            print(f"  点数: {len(keep_points):,}")

        # 2. 除去対象点群を保存（赤色）
        if remove_points:
            remove_pcd = o3d.geometry.PointCloud()
            remove_pcd.points = o3d.utility.Vector3dVector(np.array(remove_points))
            remove_pcd.paint_uniform_color([1.0, 0.0, 0.0])

            remove_ply_path = os.path.join(output_folder, "remove_files_points.ply")
            remove_xyz_path = os.path.join(output_folder, "remove_files_points.xyz")

            o3d.io.write_point_cloud(remove_ply_path, remove_pcd)

            # XYZ形式でも保存
            with open(remove_xyz_path, "w") as f:
                f.write("X Y Z R G B\n")
                for point in remove_points:
                    line = f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}"
                    f.write(f"{line} 255 0 0\n")  # RGB値は赤

            print(f"除去対象ファイルの点群を保存: {remove_ply_path}")
            print(f"除去対象ファイルの点群を保存: {remove_xyz_path}")
            print(f"  点数: {len(remove_points):,}")

        # 3. 統合された色付き点群を保存
        if keep_points or remove_points:
            combined_points = []
            combined_colors = []

            # グレーの点群を追加
            if keep_points:
                combined_points.extend(keep_points)
                combined_colors.extend([[0.7, 0.7, 0.7]] * len(keep_points))

            # 赤の点群を追加
            if remove_points:
                combined_points.extend(remove_points)
                combined_colors.extend([[1.0, 0.0, 0.0]] * len(remove_points))

            combined_pcd = o3d.geometry.PointCloud()
            combined_pcd.points = o3d.utility.Vector3dVector(np.array(combined_points))
            combined_pcd.colors = o3d.utility.Vector3dVector(np.array(combined_colors))

            combined_ply_path = os.path.join(
                output_folder, "combined_colored_points.ply"
            )
            combined_xyz_path = os.path.join(
                output_folder, "combined_colored_points.xyz"
            )

            o3d.io.write_point_cloud(combined_ply_path, combined_pcd)

            # XYZ形式でも保存
            with open(combined_xyz_path, "w") as f:
                f.write("X Y Z R G B\n")
                for i, point in enumerate(combined_points):
                    color = combined_colors[i]
                    r = int(color[0] * 255)
                    g = int(color[1] * 255)
                    b = int(color[2] * 255)
                    line = f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}"
                    f.write(f"{line} {r} {g} {b}\n")

            print(f"統合された色付き点群を保存: {combined_ply_path}")
            print(f"統合された色付き点群を保存: {combined_xyz_path}")
            print(f"  総点数: {len(combined_points):,}")

        # 4. 保存統計情報をJSONで出力
        save_stats = {
            "timestamp": str(datetime.now()),
            "keep_files_count": len(keep_indices),
            "remove_files_count": len(remove_indices),
            "keep_points_count": len(keep_points) if keep_points else 0,
            "remove_points_count": len(remove_points) if remove_points else 0,
            "total_points_saved": (
                len(combined_points) if (keep_points or remove_points) else 0
            ),
            "files_saved": {
                "keep_ply": "keep_files_points.ply",
                "keep_xyz": "keep_files_points.xyz",
                "remove_ply": "remove_files_points.ply",
                "remove_xyz": "remove_files_points.xyz",
                "combined_ply": "combined_colored_points.ply",
                "combined_xyz": "combined_colored_points.xyz",
            },
        }

        stats_path = os.path.join(output_folder, "saved_point_clouds_info.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(save_stats, f, indent=2, ensure_ascii=False)

        print(f"保存統計情報: {stats_path}")
        print("色付き点群の保存が完了しました!")

    def save_experiment_log(
        self,
        keep_indices: List[int],
        remove_indices: List[int],
        method1_stats: dict,
        method2_stats: dict,
        selected_method: str,
        target_count: int,
        max_generations: int,
        population_size: int,
        output_folder: str,
    ):
        """実験結果のログをテキストファイルとして保存"""
        log_path = os.path.join(output_folder, "experiment_log.txt")

        with open(log_path, "w", encoding="utf-8") as f:
            # ヘッダー情報
            f.write("=" * 60 + "\n")
            f.write("点群最適化実験結果ログ\n")
            f.write("=" * 60 + "\n")
            f.write(f"実行日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            f.write(f"入力フォルダ: {self.input_folder}\n")
            f.write(f"出力フォルダ: {output_folder}\n")
            f.write("\n")

            # 入力データ情報
            f.write("-" * 40 + "\n")
            f.write("入力データ情報\n")
            f.write("-" * 40 + "\n")
            f.write(f"処理対象txtファイル数: {len(self.txt_files)}\n")
            f.write(f"重心点の数: {len(self.centroids)}\n")
            f.write(f"目標残し数: {target_count}\n")
            f.write(f"除去対象数: {len(self.centroids) - target_count}\n")
            f.write("\n")

            # 最適化パラメータ
            f.write("-" * 40 + "\n")
            f.write("最適化パラメータ\n")
            f.write("-" * 40 + "\n")
            f.write(f"遺伝的アルゴリズム世代数: {max_generations}\n")
            f.write(f"遺伝的アルゴリズム個体数: {population_size}\n")
            f.write("\n")

            # 手法1の結果
            f.write("-" * 40 + "\n")
            f.write("手法1: 近接実除去法\n")
            f.write("-" * 40 + "\n")
            f.write(f"残し数: {method1_stats['count']}\n")
            f.write(f"最小距離: {method1_stats['min_distance']:.6f}\n")
            f.write(f"平均距離: {method1_stats['avg_distance']:.6f}\n")
            f.write(f"最大距離: {method1_stats['max_distance']:.6f}\n")
            f.write(f"距離の標準偏差: {method1_stats['std_distance']:.6f}\n")
            f.write("\n")

            # 手法2の結果
            f.write("-" * 40 + "\n")
            f.write("手法2: 遺伝的アルゴリズム\n")
            f.write("-" * 40 + "\n")
            f.write(f"残し数: {method2_stats['count']}\n")
            f.write(f"最小距離: {method2_stats['min_distance']:.6f}\n")
            f.write(f"平均距離: {method2_stats['avg_distance']:.6f}\n")
            f.write(f"最大距離: {method2_stats['max_distance']:.6f}\n")
            f.write(f"距離の標準偏差: {method2_stats['std_distance']:.6f}\n")
            if "final_fitness" in method2_stats:
                f.write(f"最終適応度: {method2_stats['final_fitness']:.6f}\n")
            f.write("\n")

            # 選択された手法
            f.write("-" * 40 + "\n")
            f.write("最終選択結果\n")
            f.write("-" * 40 + "\n")
            f.write(f"採用手法: {selected_method}\n")
            if selected_method == "手法1（近接実除去）":
                selected_stats = method1_stats
            else:
                selected_stats = method2_stats
            f.write(f"最終最小距離: {selected_stats['min_distance']:.6f}\n")
            f.write(f"最終平均距離: {selected_stats['avg_distance']:.6f}\n")
            f.write("\n")

            # 残すファイルの詳細
            f.write("-" * 40 + "\n")
            f.write("残すファイル一覧\n")
            f.write("-" * 40 + "\n")
            for i, idx in enumerate(keep_indices):
                if idx < len(self.txt_files):
                    filename = os.path.basename(self.txt_files[idx])
                    centroid = self.centroids[idx]
                    f.write(
                        f"{i+1:2d}. {filename} "
                        f"(重心: {centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})\n"
                    )
            f.write("\n")

            # 除去ファイルの詳細
            f.write("-" * 40 + "\n")
            f.write("除去対象ファイル一覧\n")
            f.write("-" * 40 + "\n")
            for i, idx in enumerate(remove_indices):
                if idx < len(self.txt_files):
                    filename = os.path.basename(self.txt_files[idx])
                    centroid = self.centroids[idx]
                    f.write(
                        f"{i+1:2d}. {filename} "
                        f"(重心: {centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})\n"
                    )
            f.write("\n")

            # 統計サマリー
            f.write("-" * 40 + "\n")
            f.write("統計サマリー\n")
            f.write("-" * 40 + "\n")
            f.write(f"総ファイル数: {len(self.txt_files)}\n")
            f.write(f"残すファイル数: {len(keep_indices)}\n")
            f.write(f"除去ファイル数: {len(remove_indices)}\n")
            f.write(f"残存率: {len(keep_indices)/len(self.txt_files)*100:.1f}%\n")
            f.write(f"除去率: {len(remove_indices)/len(self.txt_files)*100:.1f}%\n")
            f.write("\n")

            # フッター
            f.write("=" * 60 + "\n")
            f.write("ログファイル終了\n")
            f.write("=" * 60 + "\n")

        print(f"実験ログを保存しました: {log_path}")
        return log_path

    def export_removal_list(self, remove_indices: List[int], output_path: str):
        """除去対象ファイルのリストを出力"""
        removal_info = {
            "total_files": len(self.txt_files),
            "files_to_remove": len(remove_indices),
            "files_to_keep": len(self.txt_files) - len(remove_indices),
            "removed_files": [],
        }

        for idx in remove_indices:
            if idx < len(self.txt_files):
                filename = os.path.basename(self.txt_files[idx])
                filepath = self.txt_files[idx]
                removal_info["removed_files"].append(
                    {"index": int(idx), "filename": filename, "filepath": filepath}
                )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(removal_info, f, indent=2, ensure_ascii=False)

        print(f"除去対象ファイル情報を保存しました: {output_path}")

    def _save_colored_point_clouds(self, keep_points, remove_points, base_save_path):
        """色付き点群をファイルに保存"""
        if not base_save_path:
            return

        # 保存パスの準備
        base_dir = os.path.dirname(base_save_path)
        base_name = os.path.splitext(os.path.basename(base_save_path))[0]

        # 残すファイルの点群を保存（グレー）
        if keep_points:
            keep_pcd = o3d.geometry.PointCloud()
            keep_pcd.points = o3d.utility.Vector3dVector(np.array(keep_points))
            keep_pcd.paint_uniform_color([0.7, 0.7, 0.7])

            # PLY形式で保存
            keep_ply_path = os.path.join(base_dir, f"{base_name}_keep_files.ply")
            o3d.io.write_point_cloud(keep_ply_path, keep_pcd)
            print(f"残すファイルの点群を保存しました: {keep_ply_path}")

            # XYZ形式でも保存
            keep_xyz_path = os.path.join(base_dir, f"{base_name}_keep_files.xyz")
            self._save_points_as_xyz(keep_points, keep_xyz_path, color=[0.7, 0.7, 0.7])

        # 除去対象ファイルの点群を保存（赤）
        if remove_points:
            remove_pcd = o3d.geometry.PointCloud()
            remove_pcd.points = o3d.utility.Vector3dVector(np.array(remove_points))
            remove_pcd.paint_uniform_color([1.0, 0.0, 0.0])

            # PLY形式で保存
            remove_ply_path = os.path.join(base_dir, f"{base_name}_remove_files.ply")
            o3d.io.write_point_cloud(remove_ply_path, remove_pcd)
            print(f"除去対象ファイルの点群を保存しました: {remove_ply_path}")

            # XYZ形式でも保存
            remove_xyz_path = os.path.join(base_dir, f"{base_name}_remove_files.xyz")
            self._save_points_as_xyz(
                remove_points, remove_xyz_path, color=[1.0, 0.0, 0.0]
            )

        # 統合した点群を保存
        if keep_points and remove_points:
            combined_pcd = o3d.geometry.PointCloud()

            # 点座標を統合
            all_combined_points = np.vstack([keep_points, remove_points])
            combined_pcd.points = o3d.utility.Vector3dVector(all_combined_points)

            # 色を設定（グレー + 赤）
            keep_colors = np.tile([0.7, 0.7, 0.7], (len(keep_points), 1))
            remove_colors = np.tile([1.0, 0.0, 0.0], (len(remove_points), 1))
            all_colors = np.vstack([keep_colors, remove_colors])
            combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)

            # PLY形式で保存
            combined_ply_path = os.path.join(base_dir, f"{base_name}_combined.ply")
            o3d.io.write_point_cloud(combined_ply_path, combined_pcd)
            print(f"統合した色付き点群を保存しました: {combined_ply_path}")

            # 詳細情報付きのXYZ形式でも保存
            combined_xyz_path = os.path.join(base_dir, f"{base_name}_combined.xyz")
            self._save_combined_points_as_xyz(
                keep_points, remove_points, combined_xyz_path
            )

    def _save_points_as_xyz(self, points, file_path, color=None):
        """点群をXYZ形式で保存"""
        with open(file_path, "w") as f:
            f.write("# X Y Z R G B\n")
            for point in points:
                if color:
                    r, g, b = [int(c * 255) for c in color]
                    f.write(
                        f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n"
                    )
                else:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        print(f"XYZ形式で保存しました: {file_path}")

    def _save_combined_points_as_xyz(self, keep_points, remove_points, file_path):
        """統合した点群をXYZ形式で保存（色情報付き）"""
        with open(file_path, "w") as f:
            f.write("# Combined point cloud with colors\n")
            f.write("# X Y Z R G B Type\n")

            # 残すファイルの点群（グレー）
            for point in keep_points:
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} 179 179 179 keep\n"
                )

            # 除去対象ファイルの点群（赤）
            for point in remove_points:
                f.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} 255 0 0 remove\n"
                )

        print(f"統合XYZ形式で保存しました: {file_path}")

    def save_centroids_with_colors(self, keep_indices, remove_indices, save_path):
        """重心点を色付きで保存"""
        if not self.centroids or not save_path:
            return

        base_dir = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]

        centroids_array = np.array(self.centroids)

        # PLY形式で保存
        centroids_ply_path = os.path.join(base_dir, f"{base_name}_centroids.ply")

        # 重心点の点群オブジェクトを作成
        centroid_pcd = o3d.geometry.PointCloud()
        centroid_pcd.points = o3d.utility.Vector3dVector(centroids_array)

        # 色を設定
        colors = np.zeros((len(centroids_array), 3))
        for idx in keep_indices:
            colors[idx] = [0.0, 1.0, 0.0]  # 緑
        for idx in remove_indices:
            colors[idx] = [1.0, 0.0, 0.0]  # 赤

        centroid_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(centroids_ply_path, centroid_pcd)
        print(f"色付き重心点を保存しました: {centroids_ply_path}")

        # XYZ形式でも保存
        centroids_xyz_path = os.path.join(base_dir, f"{base_name}_centroids.xyz")
        with open(centroids_xyz_path, "w") as f:
            f.write("# Centroids with colors and labels\n")
            f.write("# X Y Z R G B Label\n")

            for i, centroid in enumerate(centroids_array):
                if i in keep_indices:
                    f.write(
                        f"{centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f} 0 255 0 keep\n"
                    )
                elif i in remove_indices:
                    f.write(
                        f"{centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f} 255 0 0 remove\n"
                    )
                else:
                    f.write(
                        f"{centroid[0]:.6f} {centroid[1]:.6f} {centroid[2]:.6f} 128 128 128 unknown\n"
                    )

        print(f"重心点XYZ形式で保存しました: {centroids_xyz_path}")


def main():
    """メイン実行関数"""
    # 設定
    input_folder = "/Users/yamasakit3a/Documents/lab/ply/ransac/field_spheres"
    output_folder = "/Users/yamasakit3a/Documents/lab/ply/result/"
    target_count = 35  # 残したい重心点の数

    # 可視化システムの初期化
    system = ThinningVisualizationSystem(input_folder)

    # Step 1: txtファイルの読み込みと重心計算
    print("Step 1: txtファイルの読み込みと重心計算")
    if not system.load_txt_files_and_centroids():
        print("Failed to load txt files")
        return

    # Step 2: 重心点の最適化
    print(f"\nStep 2: 重心点の最適化 (目標: {target_count}個)")
    try:
        keep_indices, remove_indices, experiment_info = system.run_optimization(
            target_count=target_count,
            max_generations=100,  # 計算時間を短縮
            population_size=50,
            output_folder=output_folder,  # 実験ログ保存先を指定
        )
    except Exception as e:
        print(f"Optimization failed: {e}")
        return

    # Step 3: 結果の可視化
    print("\nStep 3: 結果の可視化")
    output_viz_path = os.path.join(output_folder, "thinning_visualization.png")
    system.visualize_thinning_result(
        keep_indices=keep_indices,
        remove_indices=remove_indices,
        save_path=output_viz_path,
        show_centroids=True,
    )

    # Step 3.5: 重心点のサマリー可視化
    print("\nStep 3.5: 重心点のサマリー可視化")
    system.create_summary_visualization(
        keep_indices=keep_indices,
        remove_indices=remove_indices,
        save_path=output_viz_path,
    )

    # Step 4: 色付き点群の保存
    print("\nStep 4: 色付き点群の保存")
    colored_clouds_folder = os.path.join(output_folder, "colored_point_clouds")
    system.save_colored_point_clouds(
        keep_indices, remove_indices, colored_clouds_folder
    )

    # Step 5: 除去対象ファイルリストの出力
    print("\nStep 5: 除去対象ファイルリストの出力")
    removal_list_path = os.path.join(output_folder, "files_to_remove.json")
    system.export_removal_list(remove_indices, removal_list_path)

    # Step 6: 実験結果ログの保存
    print("\nStep 6: 実験結果ログの保存")
    system.save_experiment_log(
        keep_indices=keep_indices,
        remove_indices=remove_indices,
        method1_stats=system.calculate_statistics(
            system.method1_nearest_neighbor_removal()
        ),
        method2_stats=system.calculate_statistics(
            system.method2_combinatorial_optimization(
                max_generations=100, population_size=50
            )
        ),
        selected_method=(
            "手法1（近接実除去）"
            if len(remove_indices) < len(keep_indices)
            else "手法2（組合せ最適化）"
        ),
        target_count=target_count,
        max_generations=100,
        population_size=50,
        output_folder=output_folder,
    )

    print("\n処理完了!")
    print(f"除去対象ファイル数: {len(remove_indices)}")
    print(f"残すファイル数: {len(keep_indices)}")
    print(f"色付き点群保存先: {colored_clouds_folder}")


if __name__ == "__main__":
    main()
