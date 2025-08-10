import os
from pathlib import Path

import cv2
import numpy as np


def undistort_fisheye_images(input_folder, output_folder):
    """
    魚眼画像の歪み補正を行う関数

    Args:
        input_folder (str): 入力画像のフォルダパス
        output_folder (str): 出力画像の保存先フォルダパス
    """
    # カメラパラメータ
    K = np.array([[413.300345, 0, 5.18221], [0, 5.75903, 5.18221], [0, 0, 1]])

    # 歪み係数
    D = np.array([[-0.273251, 0.0852417, -0.014397, 0]])

    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)

    # 入力フォルダ内の画像を処理
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    for file_path in Path(input_folder).iterdir():
        if file_path.suffix.lower() in image_extensions:
            # 画像を読み込み
            img = cv2.imread(str(file_path))
            if img is None:
                print(f"Warning: Could not read image {file_path}")
                continue

            h, w = img.shape[:2]

            # 新しいカメラ行列を計算
            new_K = K.copy()
            new_K[0, 0] = K[0, 0]
            new_K[1, 1] = K[1, 1]

            # マップを計算
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, np.eye(3), new_K, (w, h), cv2.CV_32FC1
            )

            # 歪み補正を適用
            undistorted_img = cv2.remap(
                img,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            # 出力ファイル名を生成
            output_path = Path(output_folder) / f"undistorted_{file_path.name}"

            # 画像を保存
            cv2.imwrite(str(output_path), undistorted_img)
            print(f"Processed: {file_path.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fisheye image undistortion")
    parser.add_argument("input_folder", help="Input folder containing fisheye images")
    parser.add_argument(
        "--output_folder",
        default="undistorted_images",
        help="Output folder for undistorted images",
    )
    # python caliblation.py output_frames/room_8k3_fisheye --output_folder output_frames/undistorted_images/room_8k3
    args = parser.parse_args()

    undistort_fisheye_images(args.input_folder, args.output_folder)
