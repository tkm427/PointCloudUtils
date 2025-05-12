import os
from datetime import datetime

import cv2


def extract_frames(video_path, output_dir=None, prefix="frame"):
    """
    動画ファイルからフレームを抽出し、個別の画像として保存します。

    Parameters:
    video_path (str): 入力動画ファイルのパス
    output_dir (str): 出力ディレクトリ（Noneの場合は現在時刻のフォルダを作成）
    prefix (str): 出力ファイル名のプレフィックス

    Returns:
    str: 保存したディレクトリのパス
    """
    # ビデオキャプチャオブジェクトを作成
    cap = cv2.VideoCapture(video_path)

    # ビデオが正常に開けたか確認
    if not cap.isOpened():
        raise ValueError("動画ファイルを開けませんでした")

    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = f"frames_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # フレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"総フレーム数: {total_frames}")
    print(f"FPS: {fps}")

    # フレームの読み込みと保存
    frame_count = 0
    while True:
        # フレームを読み込む
        ret, frame = cap.read()

        # フレームの読み込みに失敗したら終了
        if not ret:
            break

        # ファイル名を生成（例: frame_000001.jpg）
        filename = f"{prefix}_{str(frame_count).zfill(6)}.jpg"
        output_path = os.path.join(output_dir, filename)

        # フレームを保存
        cv2.imwrite(output_path, frame)

        # 進捗表示
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"進捗: {progress:.1f}% ({frame_count}/{total_frames})")

        frame_count += 1

    # 後処理
    cap.release()
    print(f"\n処理完了: {frame_count}フレームを保存しました")
    print(f"保存先ディレクトリ: {output_dir}")

    return output_dir


# 使用例
if __name__ == "__main__":
    # 動画ファイルのパスを指定して実行
    video_path = (
        "./raw_mov/iphone_60fps.MOV"  # 処理したい動画ファイルのパスに変更してください
    )
    extracted_dir = extract_frames(
        video_path,
        output_dir="./output_frames/iphone_60fps",  # 出力ディレクトリを指定（Noneの場合は自動生成）
    )
