#!/usr/bin/env python3
"""
重心点の最適化結果を用いて、不要なtxtファイルの点群を色付けして可視化するスクリプト
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


class Grape:
    """ぶどうの実を表すクラス（optimize.pyから簡略化）"""

    def __init__(self, id: int, position: tuple):
        self.id = id
        self.position = position

    def distance_to(self, other):
        """他の実との距離を計算"""
        p1 = np.array(self.position)
        p2 = np.array(other.position)
        return np.linalg.norm(p1 - p2)


class SimpleThinningSystem:
    """簡易摘粒システム"""

    def __init__(self, grapes: list, target_count: int = 36):
        self.grapes = grapes
        self.target_count = target_count

    def nearest_neighbor_removal(self):
        """距離の近い実を順に間引く手法"""
        remaining_grapes = self.grapes.copy()

        while len(remaining_grapes) > self.target_count:
            # 最も近い2つの実を見つける
            min_distance = float("inf")
            closest_pair = None

            for i in range(len(remaining_grapes)):
                for j in range(i + 1, len(remaining_grapes)):
                    distance = remaining_grapes[i].distance_to(remaining_grapes[j])
                    if distance < min_distance:
                        min_distance = distance
                        closest_pair = (i, j)

            if closest_pair is None:
                break

            # より多くの近隣を持つ方を除去
            i, j = closest_pair
            remaining_grapes.pop(max(i, j))  # インデックスの大きい方から削除

        return remaining_grapes


class ThinningVisualizationSystem:
    """摘粒結果可視化システム"""

    def __init__(self, input_folder: str):
        self.input_folder = input_folder
        self.txt_files = []
        self.centroids = []
        self.file_mapping = {}

    def load_txt_files_and_centroids(self):
        """txtファイルを読み込み、重心を計算"""
        txt_pattern = os.path.join(self.input_folder, "*.txt")
        txt_files = glob.glob(txt_pattern)

        if not txt_files:
            print(f"No txt files found in {self.input_folder}")
            return False

        # 最後のファイルを除外
        sorted_files = sorted(txt_files)[:-1]

        print(f"Found {len(txt_files)} txt files")
        print(f"Processing {len(sorted_files)} files")

        centroids = []

        for i, txt_file in enumerate(sorted_files):
            print(f"Processing: {os.path.basename(txt_file)}")

            points = self._read_point_cloud_txt(txt_file)

            if points is not None:
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

    def _read_point_cloud_txt(self, file_path: str):
        """txtファイルから点群データを読み込む"""
        try:
            data = np.loadtxt(file_path, skiprows=1)
            points = data[:, :3]
            return points
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def run_optimization(self, target_count: int = 25):
        """重心点に対して最適化を実行"""
        if not self.centroids:
            print("No centroids available")
            return [], []

        # 重心点をGrapeオブジェクトに変換
        grapes = []
        for i, centroid in enumerate(self.centroids):
            pos = (centroid[0], centroid[1], centroid[2])
            grapes.append(Grape(id=i, position=pos))

        # 摘粒システムの実行
        system = SimpleThinningSystem(grapes, target_count=target_count)

        print(f"重心点の最適化開始")
        print(f"重心点の数: {len(grapes)}")
        print(f"目標残し数: {target_count}")

        result_grapes = system.nearest_neighbor_removal()
        keep_ids = {grape.id for grape in result_grapes}

        keep_indices = list(keep_ids)
        all_indices = set(range(len(self.centroids)))
        remove_indices = list(all_indices - keep_ids)

        print(f"最適化完了 - 残し: {len(keep_indices)}, 除去: {len(remove_indices)}")

        return keep_indices, remove_indices

    def load_all_point_clouds(self):
        """すべてのtxtファイルの点群を読み込み、統合する"""
        all_points = []
        file_labels = []

        for i, txt_file in enumerate(self.txt_files):
            points = self._read_point_cloud_txt(txt_file)
            if points is not None:
                all_points.append(points)
                file_labels.extend([i] * len(points))

        if all_points:
            combined_points = np.vstack(all_points)
            file_labels = np.array(file_labels)
            return combined_points, file_labels
        else:
            return np.array([]), np.array([])

    def visualize_thinning_result(
        self, keep_indices, remove_indices, save_path=None, show_centroids=True
    ):
        """摘粒結果を可視化"""
        print("点群データを読み込み中...")
        all_points, file_labels = self.load_all_point_clouds()

        if len(all_points) == 0:
            print("No valid point clouds found")
            return

        print(f"総点数: {len(all_points)}")
        print(f"ファイル数: {len(self.txt_files)}")

        # 図の設定
        fig = plt.figure(figsize=(16, 10))

        # メイン可視化
        ax_main = fig.add_subplot(1, 2, 1, projection="3d")

        # 除去対象ファイルのセット
        removed_file_indices = set(remove_indices)

        # 残すファイルの点群（グレー）
        for i in range(len(self.txt_files)):
            if i not in removed_file_indices:
                mask = file_labels == i
                if np.any(mask):
                    points_subset = all_points[mask]
                    ax_main.scatter(
                        points_subset[:, 0],
                        points_subset[:, 1],
                        points_subset[:, 2],
                        c="lightgray",
                        s=20,
                        alpha=0.6,
                    )

        # 除去対象ファイルの点群（赤色）
        for i in removed_file_indices:
            mask = file_labels == i
            if np.any(mask):
                points_subset = all_points[mask]
                ax_main.scatter(
                    points_subset[:, 0],
                    points_subset[:, 1],
                    points_subset[:, 2],
                    c="red",
                    s=30,
                    alpha=0.8,
                    edgecolors="darkred",
                    linewidth=0.5,
                )

        # 重心点の表示
        if show_centroids and self.centroids:
            centroids_array = np.array(self.centroids)

            # 残す重心点（緑）
            if keep_indices:
                keep_centroids = centroids_array[keep_indices]
                ax_main.scatter(
                    keep_centroids[:, 0],
                    keep_centroids[:, 1],
                    keep_centroids[:, 2],
                    c="green",
                    s=100,
                    alpha=1.0,
                    edgecolors="darkgreen",
                    linewidth=2,
                    marker="o",
                    label="残す重心点",
                )

            # 除去する重心点（赤）
            if remove_indices:
                remove_centroids = centroids_array[remove_indices]
                ax_main.scatter(
                    remove_centroids[:, 0],
                    remove_centroids[:, 1],
                    remove_centroids[:, 2],
                    c="red",
                    s=100,
                    alpha=1.0,
                    edgecolors="darkred",
                    linewidth=2,
                    marker="X",
                    label="除去する重心点",
                )

        ax_main.set_title("摘粒結果: 除去対象ファイルの点群を赤色で表示", fontsize=14)
        ax_main.set_xlabel("X")
        ax_main.set_ylabel("Y")
        ax_main.set_zlabel("Z")
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)

        # 重心点のみの可視化
        if self.centroids:
            ax_centroid = fig.add_subplot(1, 2, 2, projection="3d")
            centroids_array = np.array(self.centroids)

            # 残す重心点
            if keep_indices:
                keep_centroids = centroids_array[keep_indices]
                ax_centroid.scatter(
                    keep_centroids[:, 0],
                    keep_centroids[:, 1],
                    keep_centroids[:, 2],
                    c="green",
                    s=80,
                    alpha=0.8,
                    edgecolors="darkgreen",
                    linewidth=1,
                    label=f"残す ({len(keep_indices)}個)",
                )

            # 除去する重心点
            if remove_indices:
                remove_centroids = centroids_array[remove_indices]
                ax_centroid.scatter(
                    remove_centroids[:, 0],
                    remove_centroids[:, 1],
                    remove_centroids[:, 2],
                    c="red",
                    s=80,
                    alpha=0.8,
                    edgecolors="darkred",
                    linewidth=1,
                    marker="X",
                    label=f"除去 ({len(remove_indices)}個)",
                )

            ax_centroid.set_title("重心点の分布", fontsize=12)
            ax_centroid.set_xlabel("X")
            ax_centroid.set_ylabel("Y")
            ax_centroid.set_zlabel("Z")
            ax_centroid.legend()
            ax_centroid.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"可視化結果を保存しました: {save_path}")

        plt.show()

    def export_removal_list(self, remove_indices, output_path):
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


def main():
    """メイン実行関数"""
    # 設定
    input_folder = "/Users/yamasakit3a/Documents/lab/ply/ransac/field_spheres"
    output_folder = "/Users/yamasakit3a/Documents/lab/ply/ransac"
    target_count = 25  # 残したい重心点の数

    # 可視化システムの初期化
    system = ThinningVisualizationSystem(input_folder)

    # Step 1: txtファイルの読み込みと重心計算
    print("Step 1: txtファイルの読み込みと重心計算")
    if not system.load_txt_files_and_centroids():
        print("Failed to load txt files")
        return

    # Step 2: 重心点の最適化
    print(f"\nStep 2: 重心点の最適化 (目標: {target_count}個)")
    keep_indices, remove_indices = system.run_optimization(target_count=target_count)

    if not keep_indices and not remove_indices:
        print("Optimization failed")
        return

    # Step 3: 結果の可視化
    print(f"\nStep 3: 結果の可視化")
    output_viz_path = os.path.join(output_folder, "thinning_visualization.png")
    system.visualize_thinning_result(
        keep_indices=keep_indices,
        remove_indices=remove_indices,
        save_path=output_viz_path,
        show_centroids=True,
    )

    # Step 4: 除去対象ファイルリストの出力
    print(f"\nStep 4: 除去対象ファイルリストの出力")
    removal_list_path = os.path.join(output_folder, "files_to_remove.json")
    system.export_removal_list(remove_indices, removal_list_path)

    print(f"\n処理完了!")
    print(f"除去対象ファイル数: {len(remove_indices)}")
    print(f"残すファイル数: {len(keep_indices)}")


if __name__ == "__main__":
    main()
