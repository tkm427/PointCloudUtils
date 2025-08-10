#!/usr/bin/env python3
"""
フォルダ内のすべてのtxtファイルの重心点を計算し、点群ファイルとして保存するスクリプト
"""

import glob
import os

import numpy as np


def read_point_cloud_txt(file_path):
    """
    txtファイルから点群データを読み込む

    Args:
        file_path (str): txtファイルのパス

    Returns:
        numpy.ndarray: 点群の座標データ (N x 3)
    """
    try:
        # ヘッダー行をスキップしてデータを読み込み
        data = np.loadtxt(file_path, skiprows=1)

        # X, Y, Z座標のみを抽出（最初の3列）
        points = data[:, :3]

        return points
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def calculate_centroid(points):
    """
    点群の重心を計算する

    Args:
        points (numpy.ndarray): 点群の座標データ (N x 3)

    Returns:
        numpy.ndarray: 重心座標 (3,)
    """
    if points is None or len(points) == 0:
        return None

    return np.mean(points, axis=0)


def save_centroids_as_xyz(centroids, output_path):
    """
    重心点群をXYZファイルとして保存する

    Args:
        centroids (list): 重心座標のリスト
        output_path (str): 出力ファイルパス
    """
    with open(output_path, "w") as f:
        f.write("# Centroid points calculated from txt files\n")
        f.write("# X Y Z\n")
        for centroid in centroids:
            if centroid is not None:
                x, y, z = centroid
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def save_centroids_as_ply(centroids, output_path):
    """
    重心点群をPLYファイルとして保存する

    Args:
        centroids (list): 重心座標のリスト
        output_path (str): 出力ファイルパス
    """
    # Noneを除去
    valid_centroids = [c for c in centroids if c is not None]
    n_points = len(valid_centroids)

    with open(output_path, "w") as f:
        # PLYヘッダー
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # 頂点データ（重心点を赤色で表示）
        for centroid in valid_centroids:
            x, y, z = centroid
            f.write(f"{x:.6f} {y:.6f} {z:.6f} 255 0 0\n")


def main():
    """
    メイン処理
    """
    # 入力フォルダパス
    input_folder = "/Users/yamasakit3a/Documents/lab/ply/ransac/field_spheres"

    # 出力フォルダ
    output_folder = "/Users/yamasakit3a/Documents/lab/ply/ransac/field_spheres"

    # txtファイルのパターン
    txt_pattern = os.path.join(input_folder, "*.txt")
    txt_files = glob.glob(txt_pattern)

    if not txt_files:
        print(f"No txt files found in {input_folder}")
        return

    # ファイルをソートして最後のファイルを除外
    sorted_files = sorted(txt_files)[:-1]  # 最後のファイルを除外

    print(f"Found {len(txt_files)} txt files")
    print(f"Processing {len(sorted_files)} files (excluding the last one)")

    centroids = []
    file_info = []

    # 各txtファイルを処理
    for txt_file in sorted_files:
        print(f"Processing: {os.path.basename(txt_file)}")

        # 点群データを読み込み
        points = read_point_cloud_txt(txt_file)

        if points is not None:
            # 重心を計算
            centroid = calculate_centroid(points)
            centroids.append(centroid)
            file_info.append(
                {
                    "file": os.path.basename(txt_file),
                    "num_points": len(points),
                    "centroid": centroid,
                }
            )

            x, y, z = centroid
            print(f"  Points: {len(points)}, " f"Centroid: ({x:.3f}, {y:.3f}, {z:.3f})")
        else:
            centroids.append(None)
            print("  Failed to read file")

    # 結果を保存
    if centroids:
        # XYZファイルとして保存
        xyz_output = os.path.join(output_folder, "centroids.xyz")
        save_centroids_as_xyz(centroids, xyz_output)
        print(f"Centroids saved as XYZ: {xyz_output}")

        # PLYファイルとして保存
        ply_output = os.path.join(output_folder, "centroids.ply")
        save_centroids_as_ply(centroids, ply_output)
        print(f"Centroids saved as PLY: {ply_output}")

        # 統計情報を保存
        stats_output = os.path.join(output_folder, "centroid_statistics.txt")
        with open(stats_output, "w") as f:
            f.write("Centroid Statistics\n")
            f.write("==================\n\n")

            valid_centroids = [c for c in centroids if c is not None]
            if valid_centroids:
                all_centroids = np.array(valid_centroids)

                f.write(f"Total files found: {len(txt_files)}\n")
                f.write(f"Files processed: {len(sorted_files)}\n")
                f.write(f"Valid centroids: {len(valid_centroids)}\n")
                failed_files = len(sorted_files) - len(valid_centroids)
                f.write(f"Failed files: {failed_files}\n\n")

                f.write("Overall centroid statistics:\n")
                f.write(f"Mean X: {np.mean(all_centroids[:, 0]):.6f}\n")
                f.write(f"Mean Y: {np.mean(all_centroids[:, 1]):.6f}\n")
                f.write(f"Mean Z: {np.mean(all_centroids[:, 2]):.6f}\n")
                f.write(f"Std X: {np.std(all_centroids[:, 0]):.6f}\n")
                f.write(f"Std Y: {np.std(all_centroids[:, 1]):.6f}\n")
                f.write(f"Std Z: {np.std(all_centroids[:, 2]):.6f}\n\n")

                f.write("Individual file information:\n")
                f.write("-" * 80 + "\n")
                for info in file_info:
                    if info["centroid"] is not None:
                        filename = info["file"]
                        num_pts = info["num_points"]
                        cx, cy, cz = info["centroid"]
                        f.write(
                            f"{filename:20} | Points: {num_pts:6} | "
                            f"Centroid: ({cx:8.3f}, {cy:8.3f}, "
                            f"{cz:8.3f})\n"
                        )

        print(f"Statistics saved: {stats_output}")
    else:
        print("No valid centroids calculated")


if __name__ == "__main__":
    main()
