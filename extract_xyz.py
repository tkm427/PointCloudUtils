def extract_xyz_coordinates(input_filename):
    """
    3Dポイントクラウドデータファイルからx, y, z座標を抽出する関数

    Parameters:
    input_filename (str): 入力ファイルのパス

    Returns:
    list: (x, y, z)の座標タプルのリスト
    """
    coordinates = []

    with open(input_filename, "r") as file:
        # ヘッダー行をスキップ
        for _ in range(2):
            next(file)

        # 各行を処理
        for line in file:
            # スペースで分割
            parts = line.strip().split()

            # インデックス1, 2, 3がx, y, z座標
            if len(parts) >= 4:  # 最低でも4つの値があることを確認
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    coordinates.append((x, y, z))
                except (ValueError, IndexError):
                    continue  # 数値への変換に失敗した行はスキップ

    return coordinates


def save_coordinates_to_txt(coordinates, output_filename):
    """
    座標データをtxtファイルに保存する関数

    Parameters:
    coordinates (list): (x, y, z)の座標タプルのリスト
    output_filename (str): 出力ファイルのパス
    """
    with open(output_filename, "w") as file:
        # 座標データを書き込む
        for x, y, z in coordinates:
            file.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def main():
    input_filename = "chair_30fps/sfm-output-highq/points3D.txt"  # 入力ファイル名
    output_filename = "chair_30fps/sfm-output-highq/3Dcoordinates.txt"  # 出力ファイル名

    # 座標を抽出
    coords = extract_xyz_coordinates(input_filename)

    # ファイルに保存
    save_coordinates_to_txt(coords, output_filename)

    # 結果の報告
    print(f"Coordinates extracted: {len(coords)} points")
    print(f"Data saved to: {output_filename}")

    # 最初の数点を表示して確認
    print("\nFirst 5 points of the extracted data:")
    with open(output_filename, "r") as file:
        for i, line in enumerate(file):
            if i == 0:  # ヘッダー行をスキップ
                continue
            if i > 5:  # 最初の5点のみ表示
                break
            print(line.strip())


if __name__ == "__main__":
    main()
