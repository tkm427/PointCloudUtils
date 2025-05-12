import os

import cv2
import numpy as np


def create_fisheye_map(
    equirect_width, equirect_height, fov=180, output_width=800, output_height=800
):
    """
    equirectangular画像からfisheye画像への変換マップを作成

    Parameters:
    -----------
    equirect_width : int
        入力の360度画像の幅
    equirect_height : int
        入力の360度画像の高さ
    fov : float
        魚眼レンズの視野角（度）。通常180度
    output_width : int
        出力画像の幅
    output_height : int
        出力画像の高さ

    Returns:
    --------
    map_x : ndarray
        x座標の変換マップ
    map_y : ndarray
        y座標の変換マップ
    mask : ndarray
        有効な画素領域のマスク
    """
    # 出力画像の中心と半径
    cx = output_width / 2
    cy = output_height / 2
    radius = min(cx, cy)

    # 出力画像の各ピクセルに対する座標を生成
    y, x = np.indices((output_height, output_width))
    x = x - cx
    y = y - cy

    # 中心からの距離と角度を計算
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    # 有効な画素領域のマスクを作成（円形領域）
    mask = r <= radius

    # 距離を正規化（0～1の範囲に）
    r = r / radius

    # 魚眼レンズモデルに基づいて天頂角（縦方向の角度）を計算
    # 等距離射影方式（Equidistant Projection）を使用
    phi = r * np.radians(fov / 2)

    # 3D座標に変換
    x_3d = np.sin(phi) * np.cos(theta)
    y_3d = np.sin(phi) * np.sin(theta)
    z_3d = np.cos(phi)

    # 経度と緯度に変換
    longitude = np.arctan2(x_3d, z_3d)
    latitude = np.arcsin(y_3d)

    # equirectangular画像の座標に変換
    map_x = (longitude / (2 * np.pi) + 0.5) * equirect_width
    map_y = (latitude / np.pi + 0.5) * equirect_height

    # マスク外の領域を0に設定
    map_x = np.where(mask, map_x, 0)
    map_y = np.where(mask, map_y, 0)

    return map_x.astype(np.float32), map_y.astype(np.float32), mask


def convert_video(input_path, output_dir, fov=180, output_width=800, output_height=800):
    """
    equirectangular形式の動画をfisheye形式のフレーム画像に変換

    Parameters:
    -----------
    input_path : str
        入力動画のパス
    output_dir : str
        出力画像を保存するディレクトリのパス
    fov : float
        魚眼レンズの視野角（度）
    output_width : int
        出力画像の幅
    output_height : int
        出力画像の高さ
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # 入力動画を開く
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open input video")

    # 入力動画の情報を取得
    equirect_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    equirect_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 変換マップを作成（一度だけ計算）
    map_x, map_y, mask = create_fisheye_map(
        equirect_width, equirect_height, fov, output_width, output_height
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # equirectangular画像からfisheye画像に変換
        fisheye = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        # マスク外の領域を黒に設定
        fisheye[~mask] = 0

        # フレーム画像としてJPEG形式で保存
        output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(output_path, fisheye, [cv2.IMWRITE_JPEG_QUALITY, 95])

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    # リソースの解放
    cap.release()
    print(f"Completed! Total {frame_count} frames were processed.")


if __name__ == "__main__":
    # 使用例
    input_video = "raw_mov/room_8k3.MP4"  # 入力の360度動画
    output_video = "./output_frames/room_8k3_fisheye"  # 出力の魚眼動画

    convert_video(
        input_video,
        output_video,
        fov=180,  # 視野角（度）
        output_width=1920,
        output_height=1920,
    )
