import os

import cv2
import numpy as np


def create_perspective_map(
    equirect_width, equirect_height, fov_h=90, output_width=640, output_height=480
):
    """
    equirectangular画像からpinhole画像への変換マップを作成

    Parameters:
    -----------
    equirect_width : int
        入力の360度画像の幅
    equirect_height : int
        入力の360度画像の高さ
    fov_h : float
        水平視野角（度）
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
    """
    # 出力画像の中心
    cx = output_width / 2
    cy = output_height / 2

    # 焦点距離の計算（水平FOVから）
    f = cx / np.tan(np.radians(fov_h / 2))

    # 出力画像の各ピクセルに対応する3D座標を計算
    y, x = np.indices((output_height, output_width))
    x = x - cx
    y = y - cy

    # 3D座標を計算（z=1の平面に投影）
    z = np.full_like(x, f)
    norm = np.sqrt(x * x + y * y + z * z)

    # 正規化して単位ベクトルに
    x = x / norm
    y = y / norm
    z = z / norm

    # 球面座標に変換
    longitude = np.arctan2(x, z)  # 経度 (-π to π)
    latitude = np.arcsin(y)  # 緯度 (-π/2 to π/2)

    # equirectangular画像の座標に変換
    map_x = (longitude / (2 * np.pi) + 0.5) * equirect_width
    map_y = (latitude / np.pi + 0.5) * equirect_height

    return map_x.astype(np.float32), map_y.astype(np.float32)


def convert_video(
    input_path, output_dir, fov_h=90, output_width=640, output_height=480
):
    """
    equirectangular形式の動画をpinhole形式のフレーム画像に変換

    Parameters:
    -----------
    input_path : str
        入力動画のパス
    output_dir : str
        出力画像を保存するディレクトリのパス
    fov_h : float
        水平視野角（度）
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
    map_x, map_y = create_perspective_map(
        equirect_width, equirect_height, fov_h, output_width, output_height
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # equirectangular画像からpinhole画像に変換
        pinhole = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        # フレーム画像としてJPEG形式で保存
        output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(
            output_path, pinhole, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )  # 95は品質パラメータ（0-100）

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    # リソースの解放
    cap.release()
    print(f"Completed! Total {frame_count} frames were processed.")


if __name__ == "__main__":
    # 使用例
    input_video = "raw_mov/room_4k5.MP4"  # 入力の360度動画
    output_video = "./output_frames/room_4k5_test"  # 出力の通常動画

    convert_video(
        input_video,
        output_video,
        fov_h=120,  # 水平視野角（度）
        output_width=1920,
        output_height=1920,
    )
