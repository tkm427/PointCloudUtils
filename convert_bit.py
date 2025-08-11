import glob
import os

from PIL import Image


def convert_images_for_colmap(input_folder, output_folder=None):
    """
    指定されたフォルダ内のすべての画像をColMap対応形式に変換する

    Args:
        input_folder: 入力画像フォルダのパス
        output_folder: 出力フォルダのパス（Noneの場合は入力フォルダに上書き）
    """

    # 出力フォルダが指定されていない場合は入力フォルダを使用
    if output_folder is None:
        output_folder = input_folder
    else:
        os.makedirs(output_folder, exist_ok=True)

    # サポートされている画像形式
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]

    # すべての画像ファイルを取得
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))

    print(f"Found {len(image_files)} images to convert")

    successful_conversions = 0
    failed_conversions = 0

    for i, image_path in enumerate(image_files):
        try:
            # 画像を開く
            with Image.open(image_path) as img:
                # RGBに変換（アルファチャンネルを削除）
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")

                # 出力ファイル名を決定
                filename = os.path.basename(image_path)
                name, _ = os.path.splitext(filename)
                output_path = os.path.join(output_folder, f"{name}.png")

                # 高品質のPNGとして保存
                img.save(output_path, "PNG", quality=95, optimize=True)

                successful_conversions += 1

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(image_files)} images")

        except Exception as e:
            print(f"Failed to convert {image_path}: {str(e)}")
            failed_conversions += 1

    print(f"\nConversion completed!")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")


if __name__ == "__main__":
    input_folder = "/Users/yamasakit3a/Documents/lab/ply/data/output_frames/before3"

    # 同じフォルダに上書き保存する場合
    # convert_images_for_colmap(input_folder)

    # 別のフォルダに保存する場合（コメントアウトを解除して使用）
    output_folder = (
        "/Users/yamasakit3a/Documents/lab/ply/data/output_frames/before3_converted"
    )
    convert_images_for_colmap(input_folder, output_folder)
