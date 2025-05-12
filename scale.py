import numpy as np
import open3d as o3d


def scale_ply_file(input_path, output_path, scale_factor=2.0):
    """
    PLYファイルを読み込み、点群のスケールを変更して保存する関数

    Parameters:
    -----------
    input_path : str
        入力PLYファイルのパス
    output_path : str
        出力PLYファイルのパス
    scale_factor : float
        スケール倍率（2.0なら2倍の大きさになる）
    """
    # PLYファイルを読み込む
    point_cloud = o3d.io.read_point_cloud(input_path)

    # 点群の座標を取得
    points = np.asarray(point_cloud.points)

    # 重心を計算（スケーリングの中心点）
    centroid = np.mean(points, axis=0)

    # 重心を原点とした座標系に変換してからスケーリング、その後元の位置に戻す
    scaled_points = (points - centroid) * scale_factor + centroid

    # スケールされた点群を元のポイントクラウドにセット
    point_cloud.points = o3d.utility.Vector3dVector(scaled_points)

    # 結果を保存
    o3d.io.write_point_cloud(output_path, point_cloud)

    print(f"スケーリングが完了しました。倍率: {scale_factor}倍")
    print(f"出力ファイル: {output_path}")

    return point_cloud


# 使用例
if __name__ == "__main__":
    input_file = "result/chair_30fps/3dgs/output.ply"  # 入力PLYファイルのパス
    output_file = "result/chair_30fps/3dgs/output_scaled.ply"  # 出力PLYファイルのパス
    scale = 5.0  # スケール倍率（ここでは5倍に設定）

    scaled_cloud = scale_ply_file(input_file, output_file, scale)

    # 結果を可視化する（必要に応じて）
    o3d.visualization.draw_geometries([scaled_cloud])
