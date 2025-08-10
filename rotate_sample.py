import argparse
import os

from PIL import Image


def rotate_and_downsample(input_folder, output_folder, interval=1, rotation_angle=90):
    """
    フォルダー内の画像を回転させ、指定間隔でダウンサンプリングする

    Args:
        input_folder (str): 入力画像フォルダのパス
        output_folder (str): 出力画像フォルダのパス
        interval (int): サンプリング間隔（デフォルト: 1）
        rotation_angle (int): 回転角度（デフォルト: 90度右回転）
    """
    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)

    # 入力フォルダ内の画像ファイルを取得してソート
    image_files = [
        f
        for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    image_files.sort()

    print(f"総画像数: {len(image_files)}")
    print(f"サンプリング間隔: {interval}")
    print(f"回転角度: {rotation_angle}度")

    processed_count = 0

    # 指定間隔で画像を処理
    for i in range(0, len(image_files), interval):
        input_path = os.path.join(input_folder, image_files[i])

        try:
            # 画像を開く
            with Image.open(input_path) as img:
                # 右に90度回転（時計回り）
                rotated_img = img.rotate(-rotation_angle, expand=True)

                # 出力ファイル名を生成（連番を振り直し）
                output_filename = f"rotated_frame_{processed_count:06d}.jpg"
                output_path = os.path.join(output_folder, output_filename)

                # 回転した画像を保存
                rotated_img.save(output_path, "JPEG", quality=95)

                processed_count += 1
                print(f"処理完了: {image_files[i]} -> {output_filename}")

        except Exception as e:
            print(f"エラー: {image_files[i]} の処理中にエラーが発生しました: {e}")

    print(f"\n処理完了！ {processed_count}枚の画像を処理しました。")
    print(f"出力フォルダ: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="画像を回転させダウンサンプリングする")
    parser.add_argument(
        "--input",
        "-i",
        default="output_frames/field",
        help="入力フォルダのパス (デフォルト: output_frames/field)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output_frames/field_rotated2",
        help="出力フォルダのパス (デフォルト: output_frames/field_rotated)",
    )
    parser.add_argument(
        "--interval", "-n", type=int, default=1, help="サンプリング間隔 (デフォルト: 1)"
    )
    parser.add_argument(
        "--angle", "-a", type=int, default=90, help="回転角度 (デフォルト: 90)"
    )

    args = parser.parse_args()

    # 入力フォルダの存在確認
    if not os.path.exists(args.input):
        print(f"エラー: 入力フォルダが見つかりません: {args.input}")
        return

    rotate_and_downsample(args.input, args.output, args.interval, args.angle)


if __name__ == "__main__":
    main()
