import pathlib

import numpy as np
import pycolmap


def export_colmap_to_ply(colmap_path, output_ply_path):
    """
    ColmapのSfM結果をPLYファイルにエクスポート

    Args:
        colmap_path: Colmapの出力ディレクトリパス
        output_ply_path: 出力PLYファイルパス
    """
    # Colmapの再構成結果を読み込み
    reconstruction = pycolmap.Reconstruction(colmap_path)

    # 3Dポイントを取得
    points3D = reconstruction.points3D

    if len(points3D) == 0:
        print("No 3D points found in the reconstruction")
        return False

    # ポイントデータを準備
    vertices = []
    colors = []

    for point_id, point in points3D.items():
        # 3D座標
        xyz = point.xyz
        # 色情報（RGB）
        rgb = point.color

        vertices.append(xyz)
        colors.append(rgb)

    vertices = np.array(vertices)
    colors = np.array(colors, dtype=np.uint8)

    # PLYファイルとして保存
    write_ply(output_ply_path, vertices, colors)

    print(f"Exported {len(vertices)} points to {output_ply_path}")
    return True


def write_ply(filename, vertices, colors=None):
    """
    PLYファイルを書き出し

    Args:
        filename: 出力ファイル名
        vertices: 頂点座標 (N, 3)
        colors: 色情報 (N, 3), optional
    """
    with open(filename, "w") as f:
        # PLYヘッダー
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write("end_header\n")

        # 頂点データ
        for i, vertex in enumerate(vertices):
            if colors is not None:
                f.write(
                    f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} "
                    f"{colors[i][0]} {colors[i][1]} {colors[i][2]}\n"
                )
            else:
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")


def main():
    # パス設定
    colmap_path = pathlib.Path("colmap")
    output_ply_path = pathlib.Path("colmap/points3D.ply")

    # ColmapからPLYにエクスポート
    success = export_colmap_to_ply(colmap_path, output_ply_path)

    if success:
        print("Export completed successfully!")
    else:
        print("Export failed!")
        return 1

    return 0


if __name__ == "__main__":
    main()
