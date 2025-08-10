import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist


@dataclass
class Grape:
    """ぶどうの実を表すクラス"""

    id: int
    position: Tuple[float, float, float]  # (u, v, w) 3次元座標

    def distance_to(self, other: "Grape") -> float:
        """他の実との距離を計算"""
        p1 = np.array(self.position)
        p2 = np.array(other.position)
        return np.linalg.norm(p1 - p2)


class PointCloudLoader:
    """点群ファイルローダー"""

    @staticmethod
    def load_point_cloud(file_path: str) -> List[Grape]:
        """
        点群ファイルを読み込んでGrapeオブジェクトのリストを返す

        対応フォーマット:
        - PLY: Stanford Triangle Format
        - PCD: Point Cloud Data format
        - XYZ: 単純なXYZ座標テキスト
        - CSV: カンマ区切りファイル
        """
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".ply":
            return PointCloudLoader._load_ply(file_path)
        elif file_extension == ".pcd":
            return PointCloudLoader._load_pcd(file_path)
        elif file_extension == ".xyz":
            return PointCloudLoader._load_xyz(file_path)
        elif file_extension == ".csv":
            return PointCloudLoader._load_csv(file_path)
        elif file_extension == ".txt":
            return PointCloudLoader._load_txt(file_path)
        else:
            raise ValueError(f"サポートされていないファイル形式: {file_extension}")

    @staticmethod
    def _load_ply(file_path: str) -> List[Grape]:
        """PLYファイルを読み込み"""
        grapes = []
        with open(file_path, "r") as f:
            lines = f.readlines()

        # ヘッダー解析
        vertex_count = 0
        data_start = 0

        for i, line in enumerate(lines):
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            elif line.startswith("end_header"):
                data_start = i + 1
                break

        # データ読み込み
        for i in range(data_start, data_start + vertex_count):
            if i < len(lines):
                coords = lines[i].strip().split()
                if len(coords) >= 3:
                    x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                    grapes.append(Grape(id=i - data_start, position=(x, y, z)))

        return grapes

    @staticmethod
    def _load_pcd(file_path: str) -> List[Grape]:
        """PCDファイルを読み込み"""
        grapes = []
        with open(file_path, "r") as f:
            lines = f.readlines()

        # ヘッダー解析
        points_count = 0
        data_start = 0

        for i, line in enumerate(lines):
            if line.startswith("POINTS"):
                points_count = int(line.split()[1])
            elif line.startswith("DATA"):
                data_start = i + 1
                break

        # データ読み込み
        for i in range(data_start, min(data_start + points_count, len(lines))):
            coords = lines[i].strip().split()
            if len(coords) >= 3:
                x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                grapes.append(Grape(id=i - data_start, position=(x, y, z)))

        return grapes

    @staticmethod
    def _load_xyz(file_path: str) -> List[Grape]:
        """XYZファイルを読み込み"""
        grapes = []
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line and not line.startswith("#"):  # コメント行をスキップ
                    coords = line.split()
                    if len(coords) >= 3:
                        try:
                            x, y, z = (
                                float(coords[0]),
                                float(coords[1]),
                                float(coords[2]),
                            )
                            grapes.append(Grape(id=i, position=(x, y, z)))
                        except ValueError:
                            continue

        return grapes

    @staticmethod
    def _load_csv(file_path: str) -> List[Grape]:
        """CSVファイルを読み込み"""
        grapes = []
        df = pd.read_csv(file_path)

        # 座標列を自動検出
        coord_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(coord in col_lower for coord in ["x", "y", "z", "u", "v", "w"]):
                coord_columns.append(col)

        if len(coord_columns) < 3:
            # 最初の3列を座標として使用
            coord_columns = df.columns[:3].tolist()

        if len(coord_columns) >= 3:
            for i, row in df.iterrows():
                try:
                    x, y, z = (
                        float(row[coord_columns[0]]),
                        float(row[coord_columns[1]]),
                        float(row[coord_columns[2]]),
                    )
                    grapes.append(Grape(id=i, position=(x, y, z)))
                except (ValueError, TypeError):
                    continue

        return grapes

    @staticmethod
    def _load_txt(file_path: str) -> List[Grape]:
        """TXTファイルを読み込み（XYZと同じ形式）"""
        return PointCloudLoader._load_xyz(file_path)


class GrapeThinningSystem:
    """巨峰摘粒作業システム"""

    def __init__(self, grapes: List[Grape], target_count: int = 36):
        self.grapes = grapes
        self.target_count = target_count
        self.n_grapes = len(grapes)

    def method1_nearest_neighbor_removal(self) -> List[Grape]:
        """
        手法1: 距離の近い実を順に間引く手法
        """
        remaining_grapes = self.grapes.copy()

        print(
            f"手法1開始: {len(remaining_grapes)}個の実から{self.target_count}個に削減"
        )

        while len(remaining_grapes) > self.target_count:
            # 最も近い2つの実を見つける
            min_distance = float("inf")
            closest_pair = None

            for i in range(len(remaining_grapes)):
                for j in range(i + 1, len(remaining_grapes)):
                    distance = remaining_grapes[i].distance_to(remaining_grapes[j])
                    if distance < min_distance:
                        min_distance = distance
                        closest_pair = (i, j)

            if closest_pair is None:
                break

            i, j = closest_pair
            grape_a = remaining_grapes[i]
            grape_b = remaining_grapes[j]

            # それぞれの実について2番目に近い実との距離を計算
            distance_a_to_second = self._find_second_nearest_distance(
                grape_a, remaining_grapes
            )
            distance_b_to_second = self._find_second_nearest_distance(
                grape_b, remaining_grapes
            )

            # 2番目に近い実との距離が小さい方を間引く
            if distance_a_to_second <= distance_b_to_second:
                remaining_grapes.pop(i)
            else:
                remaining_grapes.pop(j)

            # 進捗表示
            if len(remaining_grapes) % 10 == 0:
                print(f"  残り実数: {len(remaining_grapes)}")

        print(f"手法1完了: {len(remaining_grapes)}個の実")
        return remaining_grapes

    def _find_second_nearest_distance(
        self, target_grape: Grape, grape_list: List[Grape]
    ) -> float:
        """指定された実の2番目に近い実との距離を計算"""
        distances = []
        for grape in grape_list:
            if grape.id != target_grape.id:
                distances.append(target_grape.distance_to(grape))

        if len(distances) < 2:
            return float("inf")

        distances.sort()
        return distances[1]  # 2番目に近い距離

    def method2_combinatorial_optimization(
        self, max_generations: int = 100, population_size: int = 50
    ) -> List[Grape]:
        """
        手法2: 組合せ最適化を用いた手法（遺伝的アルゴリズム）
        """
        print(
            f"手法2開始: 遺伝的アルゴリズム (世代数: {max_generations}, 個体数: {population_size})"
        )

        # 初期集団を生成
        population = self._initialize_population(population_size)
        best_fitness_history = []

        for generation in range(max_generations):
            # 各個体を評価
            fitness_scores = [
                self._evaluate_fitness(individual) for individual in population
            ]
            best_fitness = max(fitness_scores)
            best_fitness_history.append(best_fitness)

            # 進捗表示
            if generation % 10 == 0:
                print(f"  世代 {generation}: 最良適応度 = {best_fitness:.4f}")

            # 新しい世代を生成
            new_population = []

            # エリート選択（上位10%を保持）
            elite_count = max(1, population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # 残りを交叉と突然変異で生成
            while len(new_population) < population_size:
                # 親選択（トーナメント選択）
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # 交叉
                child = self._crossover(parent1, parent2)

                # 突然変異
                child = self._mutate(child)

                new_population.append(child)

            population = new_population

        # 最終的に最も良い個体を選択
        final_fitness = [
            self._evaluate_fitness(individual) for individual in population
        ]
        best_individual = population[np.argmax(final_fitness)]

        print(f"手法2完了: 最終適応度 = {max(final_fitness):.4f}")
        return self._individual_to_grapes(best_individual)

    def _initialize_population(self, population_size: int) -> List[List[int]]:
        """初期集団を生成"""
        population = []
        for _ in range(population_size):
            # ランダムに target_count 個の実を選択
            individual = [0] * self.n_grapes
            selected_indices = random.sample(
                range(self.n_grapes), min(self.target_count, self.n_grapes)
            )
            for idx in selected_indices:
                individual[idx] = 1
            population.append(individual)
        return population

    def _evaluate_fitness(self, individual: List[int]) -> float:
        """個体の適応度を評価"""
        selected_grapes = self._individual_to_grapes(individual)

        if len(selected_grapes) == 0:
            return -float("inf")

        # D(x): 実の個数の差
        d_x = abs(self.target_count - len(selected_grapes))

        # G(x): 実同士の重心距離の最小値
        if len(selected_grapes) < 2:
            g_x = 0
        else:
            min_distance = float("inf")
            for i in range(len(selected_grapes)):
                for j in range(i + 1, len(selected_grapes)):
                    distance = selected_grapes[i].distance_to(selected_grapes[j])
                    min_distance = min(min_distance, distance)
            g_x = min_distance

        # F(x) = G(x) - D(x)
        return g_x - d_x

    def _individual_to_grapes(self, individual: List[int]) -> List[Grape]:
        """個体（バイナリリスト）を実のリストに変換"""
        return [self.grapes[i] for i in range(len(individual)) if individual[i] == 1]

    def _tournament_selection(
        self,
        population: List[List[int]],
        fitness_scores: List[float],
        tournament_size: int = 3,
    ) -> List[int]:
        """トーナメント選択"""
        tournament_indices = random.sample(
            range(len(population)), min(tournament_size, len(population))
        )
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """単点交叉"""
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]

        # 選択された実の数を target_count に調整
        selected_count = sum(child)
        if selected_count != self.target_count:
            child = self._adjust_selection_count(child)

        return child

    def _mutate(self, individual: List[int], mutation_rate: float = 0.1) -> List[int]:
        """突然変異"""
        mutated = individual.copy()

        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]

        # 選択された実の数を target_count に調整
        selected_count = sum(mutated)
        if selected_count != self.target_count:
            mutated = self._adjust_selection_count(mutated)

        return mutated

    def _adjust_selection_count(self, individual: List[int]) -> List[int]:
        """選択された実の数を target_count に調整"""
        selected_count = sum(individual)
        adjusted = individual.copy()

        if selected_count < self.target_count:
            # 不足分を追加
            unselected_indices = [i for i in range(len(adjusted)) if adjusted[i] == 0]
            need_to_add = self.target_count - selected_count
            if len(unselected_indices) >= need_to_add:
                add_indices = random.sample(unselected_indices, need_to_add)
                for idx in add_indices:
                    adjusted[idx] = 1

        elif selected_count > self.target_count:
            # 余分を削除
            selected_indices = [i for i in range(len(adjusted)) if adjusted[i] == 1]
            need_to_remove = selected_count - self.target_count
            if len(selected_indices) >= need_to_remove:
                remove_indices = random.sample(selected_indices, need_to_remove)
                for idx in remove_indices:
                    adjusted[idx] = 0

        return adjusted

    def visualize_results(
        self,
        original_grapes: List[Grape],
        method1_result: List[Grape],
        method2_result: List[Grape],
        save_path: Optional[str] = None,
    ):
        """結果の可視化（残す実と間引く実を色分け表示）"""
        fig = plt.figure(figsize=(15, 10))

        # 元の実のIDセットを作成
        original_ids = {grape.id for grape in original_grapes}
        method1_ids = {grape.id for grape in method1_result}
        method2_ids = {grape.id for grape in method2_result}

        # 全ての実の位置を取得
        all_positions = np.array([grape.position for grape in original_grapes])

        # 手法1の可視化（3D）
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")

        # 残す実（緑）と間引く実（赤）に分類
        kept_positions_1 = []
        removed_positions_1 = []

        for grape in original_grapes:
            if grape.id in method1_ids:
                kept_positions_1.append(grape.position)
            else:
                removed_positions_1.append(grape.position)

        # プロット
        if kept_positions_1:
            kept_pos_1 = np.array(kept_positions_1)
            ax1.scatter(
                kept_pos_1[:, 0],
                kept_pos_1[:, 1],
                kept_pos_1[:, 2],
                c="green",
                s=80,
                alpha=0.8,
                label="残す実",
                edgecolors="black",
                linewidth=0.5,
            )

        if removed_positions_1:
            removed_pos_1 = np.array(removed_positions_1)
            ax1.scatter(
                removed_pos_1[:, 0],
                removed_pos_1[:, 1],
                removed_pos_1[:, 2],
                c="red",
                s=80,
                alpha=0.8,
                label="間引く実",
                edgecolors="black",
                linewidth=0.5,
            )

        ax1.set_title("a: 距離の近い実から順に間引く手法", fontsize=14, pad=20)
        ax1.set_xlabel("u", fontsize=12)
        ax1.set_ylabel("v", fontsize=12)
        ax1.set_zlabel("w", fontsize=12)
        ax1.legend(fontsize=12, loc="upper right")
        ax1.grid(True, alpha=0.3)

        # 手法2の可視化（3D）
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        # 残す実（緑）と間引く実（赤）に分類
        kept_positions_2 = []
        removed_positions_2 = []

        for grape in original_grapes:
            if grape.id in method2_ids:
                kept_positions_2.append(grape.position)
            else:
                removed_positions_2.append(grape.position)

        # プロット
        if kept_positions_2:
            kept_pos_2 = np.array(kept_positions_2)
            ax2.scatter(
                kept_pos_2[:, 0],
                kept_pos_2[:, 1],
                kept_pos_2[:, 2],
                c="green",
                s=80,
                alpha=0.8,
                label="残す実",
                edgecolors="black",
                linewidth=0.5,
            )

        if removed_positions_2:
            removed_pos_2 = np.array(removed_positions_2)
            ax2.scatter(
                removed_pos_2[:, 0],
                removed_pos_2[:, 1],
                removed_pos_2[:, 2],
                c="red",
                s=80,
                alpha=0.8,
                label="間引く実",
                edgecolors="black",
                linewidth=0.5,
            )

        ax2.set_title("b: 組合せ最適化を用いた手法", fontsize=14, pad=20)
        ax2.set_xlabel("u", fontsize=12)
        ax2.set_ylabel("v", fontsize=12)
        ax2.set_zlabel("w", fontsize=12)
        ax2.legend(fontsize=12, loc="upper right")
        ax2.grid(True, alpha=0.3)

        # 統計情報をタイトルに追加
        method1_stats = self.calculate_statistics(method1_result)
        method2_stats = self.calculate_statistics(method2_result)

        fig.suptitle(
            f"巨峰摘粒作業結果比較\n"
            + f'手法1: {len(method1_result)}個残し (最小距離: {method1_stats["min_distance"]:.2f}) | '
            + f'手法2: {len(method2_result)}個残し (最小距離: {method2_stats["min_distance"]:.2f})',
            fontsize=16,
            y=0.95,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"可視化結果を保存しました: {save_path}")

        plt.show()

    def calculate_min_distance(self, grapes: List[Grape]) -> float:
        """実同士の最小距離を計算"""
        if len(grapes) < 2:
            return float("inf")

        min_distance = float("inf")
        for i in range(len(grapes)):
            for j in range(i + 1, len(grapes)):
                distance = grapes[i].distance_to(grapes[j])
                min_distance = min(min_distance, distance)

        return min_distance

    def calculate_statistics(self, grapes: List[Grape]) -> Dict[str, float]:
        """統計情報を計算"""
        if len(grapes) < 2:
            return {"min_distance": float("inf"), "mean_distance": 0, "std_distance": 0}

        distances = []
        for i in range(len(grapes)):
            for j in range(i + 1, len(grapes)):
                distances.append(grapes[i].distance_to(grapes[j]))

        return {
            "min_distance": min(distances),
            "mean_distance": np.mean(distances),
            "std_distance": np.std(distances),
            "count": len(grapes),
        }

    def export_results(
        self,
        method1_result: List[Grape],
        method2_result: List[Grape],
        output_dir: str = "optimize_results",
    ):
        """結果をファイルに出力"""
        os.makedirs(output_dir, exist_ok=True)

        # 手法1の結果
        method1_positions = np.array([grape.position for grape in method1_result])
        np.savetxt(
            os.path.join(output_dir, "method1_result.xyz"),
            method1_positions,
            fmt="%.6f",
            header="X Y Z",
            comments="",
        )

        # 手法2の結果
        method2_positions = np.array([grape.position for grape in method2_result])
        np.savetxt(
            os.path.join(output_dir, "method2_result.xyz"),
            method2_positions,
            fmt="%.6f",
            header="X Y Z",
            comments="",
        )

        # 統計情報
        stats = {
            "original": self.calculate_statistics(self.grapes),
            "method1": self.calculate_statistics(method1_result),
            "method2": self.calculate_statistics(method2_result),
        }

        with open(os.path.join(output_dir, "statistics.json"), "w") as f:
            json.dump(stats, f, indent=2)

        print(f"結果を {output_dir} フォルダに保存しました")


# コマンドライン実行用の関数
def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="巨峰摘粒作業システム")
    parser.add_argument(
        "input_file", help="入力点群ファイル (.ply, .pcd, .xyz, .csv, .txt)"
    )
    parser.add_argument(
        "--target", type=int, default=36, help="目標実数 (デフォルト: 36)"
    )
    parser.add_argument(
        "--output",
        default="optimize_results",
        help="出力フォルダ (デフォルト: results)",
    )
    parser.add_argument(
        "--generations", type=int, default=100, help="GA世代数 (デフォルト: 100)"
    )
    parser.add_argument(
        "--population", type=int, default=50, help="GA個体数 (デフォルト: 50)"
    )
    parser.add_argument("--no-viz", action="store_true", help="可視化をスキップ")

    args = parser.parse_args()

    # 点群ファイルの読み込み
    print(f"点群ファイルを読み込み中: {args.input_file}")
    try:
        grapes = PointCloudLoader.load_point_cloud(args.input_file)
        print(f"読み込み完了: {len(grapes)}個の点")
    except Exception as e:
        print(f"エラー: ファイル読み込みに失敗しました - {e}")
        return

    if len(grapes) < args.target:
        print(f"警告: 入力点数({len(grapes)})が目標点数({args.target})より少ないです")
        args.target = len(grapes)

    # 摘粒システムの初期化
    system = GrapeThinningSystem(grapes, target_count=args.target)

    print(f"\n巨峰摘粒作業システム開始")
    print(f"元の実の数: {len(grapes)}")
    print(f"目標の実の数: {args.target}")
    print("=" * 50)

    # 手法1: 距離の近い実を順に間引く
    method1_result = system.method1_nearest_neighbor_removal()
    method1_stats = system.calculate_statistics(method1_result)

    print("\n手法2: 組合せ最適化を用いた手法")
    # 手法2: 組合せ最適化
    method2_result = system.method2_combinatorial_optimization(
        max_generations=args.generations, population_size=args.population
    )
    method2_stats = system.calculate_statistics(method2_result)

    # 結果の比較
    print("\n" + "=" * 50)
    print("結果の比較:")
    print(
        f"手法1 - 実の数: {method1_stats['count']}, 最小距離: {method1_stats['min_distance']:.3f}"
    )
    print(
        f"手法2 - 実の数: {method2_stats['count']}, 最小距離: {method2_stats['min_distance']:.3f}"
    )

    if method2_stats["min_distance"] > method1_stats["min_distance"]:
        print("→ 手法2（組合せ最適化）の方が良い結果を得られました。")
    else:
        print("→ 手法1（近接実除去）の方が良い結果を得られました。")

    # 結果の出力
    system.export_results(method1_result, method2_result, args.output)

    # 可視化
    if not args.no_viz:
        print("\n結果を可視化中...")
        viz_path = os.path.join(args.output, "visualization.png")
        system.visualize_results(grapes, method1_result, method2_result, viz_path)


# 使用例（サンプルデータ生成）
def generate_sample_data():
    """サンプルデータを生成"""

    # サンプル点群ファイルを生成
    n_points = 40
    np.random.seed(42)

    # ブドウ房のような形状を生成
    points = []
    for i in range(n_points):
        # 楕円体状の分布
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(0.5, 3.0)

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi) - 2  # 下向きに垂れ下がる

        points.append([x, y, z])

    # サンプルファイルを保存
    os.makedirs("sample_data", exist_ok=True)

    # XYZ形式
    np.savetxt("sample_data/grapes.xyz", points, fmt="%.6f")

    # CSV形式
    df = pd.DataFrame(points, columns=["x", "y", "z"])
    df.to_csv("sample_data/grapes.csv", index=False)

    print("サンプルデータを生成しました:")
    print("  - sample_data/grapes.xyz")
    print("  - sample_data/grapes.csv")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # 引数がない場合はサンプルデータで実行
        print("サンプルデータを生成して実行します...")
        generate_sample_data()

        # サンプルデータで実行
        grapes = PointCloudLoader.load_point_cloud("sample_data/grapes.xyz")
        system = GrapeThinningSystem(grapes, target_count=35)

        print(f"\n巨峰摘粒作業システムのデモンストレーション")
        print(f"元の実の数: {len(grapes)}")
        print(f"目標の実の数: {system.target_count}")
        print("=" * 50)

        # 手法1実行
        method1_result = system.method1_nearest_neighbor_removal()
        method1_stats = system.calculate_statistics(method1_result)

        # 手法2実行
        method2_result = system.method2_combinatorial_optimization(max_generations=50)
        method2_stats = system.calculate_statistics(method2_result)

        # 結果比較
        print("\n" + "=" * 50)
        print("結果の比較:")
        print(
            f"手法1 - 実の数: {method1_stats['count']}, 最小距離: {method1_stats['min_distance']:.3f}"
        )
        print(
            f"手法2 - 実の数: {method2_stats['count']}, 最小距離: {method2_stats['min_distance']:.3f}"
        )

        # 結果出力
        system.export_results(method1_result, method2_result)
        system.visualize_results(
            grapes, method1_result, method2_result, "visualization.png"
        )

        print("\n詳細比較も表示します...")
        system.visualize_detailed_comparison(
            grapes, method1_result, method2_result, "visualization.png"
        )
    else:
        # コマンドライン引数がある場合
        main()
