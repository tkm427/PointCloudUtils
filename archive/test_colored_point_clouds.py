#!/usr/bin/env python3
"""
色付き点群保存機能のテスト用スクリプト
"""

import os

from visualize_thinning_result import ThinningVisualizationSystem


def test_colored_point_clouds():
    """色付き点群保存機能をテスト"""

    # フォルダパスを設定 (適宜変更してください)
    txt_folder = "/Users/yamasakit3a/Documents/lab/ply/sample_data"
    output_folder = "/Users/yamasakit3a/Documents/lab/ply/test_output"

    # 出力フォルダを作成
    os.makedirs(output_folder, exist_ok=True)

    print("色付き点群保存機能のテスト開始")
    print(f"入力フォルダ: {txt_folder}")
    print(f"出力フォルダ: {output_folder}")

    # システムを初期化
    system = ThinningVisualizationSystem()

    try:
        # Step 1: txtファイルと重心を読み込み
        print("\nStep 1: txtファイルと重心を読み込み")
        system.load_txt_files_and_centroids(txt_folder)

        if not hasattr(system, "centroids") or len(system.centroids) == 0:
            print("重心データがありません。txtファイルを確認してください。")
            return

        print(f"txtファイル数: {len(system.txt_files)}")
        print(f"重心数: {len(system.centroids)}")

        # Step 2: 遺伝的アルゴリズムによる最適化
        print("\nStep 2: 遺伝的アルゴリズムによる最適化")
        keep_indices, remove_indices = system.optimize_selection(
            population_size=20, generations=30  # テスト用に小さく設定
        )

        print(f"残すファイル数: {len(keep_indices)}")
        print(f"除去対象ファイル数: {len(remove_indices)}")

        # Step 3: 色付き点群を保存
        print("\nStep 3: 色付き点群を保存")
        colored_clouds_folder = os.path.join(output_folder, "colored_point_clouds")
        system.save_colored_point_clouds(
            keep_indices, remove_indices, colored_clouds_folder
        )

        # Step 4: 除去対象ファイルリストを保存
        print("\nStep 4: 除去対象ファイルリストを保存")
        removal_list_path = os.path.join(output_folder, "files_to_remove.json")
        system.export_removal_list(remove_indices, removal_list_path)

        print("\nテスト完了!")
        print(f"結果は {output_folder} に保存されました")
        print(f"色付き点群は {colored_clouds_folder} に保存されました")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_colored_point_clouds()
