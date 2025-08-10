import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

# 日本語フォント設定
plt.rcParams["font.family"] = "Noto Sans CJK JP"  # Noto Sans CJK JPを使用

# japanize_matplotlibを明示的に適用
japanize_matplotlib.japanize()


class PointCloudCurvatureAnalyzer:
    def __init__(self, ply_file_path):
        """
        点群曲率解析クラス

        Args:
            ply_file_path (str): PLYファイルのパス
        """
        self.ply_file_path = ply_file_path
        self.points = None
        self.normals = None
        self.curvatures = None
        self.point_cloud = None

    def load_ply_file(self):
        """PLYファイルを読み込む"""
        try:
            # Open3Dを使用してPLYファイルを読み込み
            self.point_cloud = o3d.io.read_point_cloud(self.ply_file_path)
            self.points = np.asarray(self.point_cloud.points)

            print(f"点群データを読み込みました: {len(self.points)} 点")
            print(
                f"座標範囲: X[{self.points[:, 0].min():.3f}, {self.points[:, 0].max():.3f}], "
                f"Y[{self.points[:, 1].min():.3f}, {self.points[:, 1].max():.3f}], "
                f"Z[{self.points[:, 2].min():.3f}, {self.points[:, 2].max():.3f}]"
            )

            return True
        except Exception as e:
            print(f"PLYファイルの読み込みエラー: {e}")
            return False

    def estimate_normals(self, k_neighbors=30):
        """法線ベクトルを推定"""
        if self.point_cloud is None:
            print("先に点群データを読み込んでください")
            return False

        # Open3Dを使用して法線を計算
        self.point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
        )

        # 法線の向きを統一
        self.point_cloud.orient_normals_consistent_tangent_plane(k_neighbors)

        self.normals = np.asarray(self.point_cloud.normals)
        print(f"法線ベクトルを計算しました: {len(self.normals)} 個")
        return True

    def calculate_curvature_method1(self, k_neighbors=30):
        """
        方法1: 近傍点の法線ベクトル変化から曲率を計算
        """
        if self.normals is None:
            print("先に法線ベクトルを計算してください")
            return None

        # k-近傍法で近傍点を検索
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm="kd_tree")
        nbrs.fit(self.points)
        distances, indices = nbrs.kneighbors(self.points)

        curvatures = []

        for i in range(len(self.points)):
            # 近傍点のインデックス（自分自身を除く）
            neighbor_indices = indices[i][1:]

            # 中心点の法線
            center_normal = self.normals[i]

            # 近傍点の法線との差の平均を計算
            normal_variations = []
            for j in neighbor_indices:
                # 法線ベクトルの内積から角度差を計算
                dot_product = np.clip(np.dot(center_normal, self.normals[j]), -1.0, 1.0)
                angle_diff = np.arccos(np.abs(dot_product))
                normal_variations.append(angle_diff)

            # 曲率の近似値として平均角度変化を使用
            curvature = np.mean(normal_variations)
            curvatures.append(curvature)

        return np.array(curvatures)

    def calculate_curvature_method2(self, k_neighbors=30):
        """
        方法2: 主曲率を基にしたガウス曲率と平均曲率の計算
        """
        # Open3Dの曲率計算機能を使用
        try:
            # 点群の密度に基づいて半径を設定
            distances = self.point_cloud.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = avg_dist * 2.0

            # 曲率を計算（この方法は簡易的な近似）
            nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm="kd_tree")
            nbrs.fit(self.points)
            distances, indices = nbrs.kneighbors(self.points)

            curvatures = []
            for i in range(len(self.points)):
                # 近傍点でのローカル曲面を近似
                neighbor_points = self.points[indices[i]]

                # 共分散行列を計算
                centered_points = neighbor_points - np.mean(neighbor_points, axis=0)
                cov_matrix = np.cov(centered_points.T)

                # 固有値を計算（主成分分析）
                eigenvalues = np.linalg.eigvals(cov_matrix)
                eigenvalues = np.sort(eigenvalues)

                # 最小固有値を曲率の指標として使用
                curvature = eigenvalues[0] / (
                    eigenvalues[0] + eigenvalues[1] + eigenvalues[2] + 1e-10
                )
                curvatures.append(curvature)

            return np.array(curvatures)

        except Exception as e:
            print(f"曲率計算エラー: {e}")
            return None

    def analyze_curvature_distribution(self, method=1, k_neighbors=30):
        """曲率分布を解析"""
        print(f"曲率計算を開始（方法{method}、近傍点数: {k_neighbors}）...")

        if method == 1:
            self.curvatures = self.calculate_curvature_method1(k_neighbors)
        else:
            self.curvatures = self.calculate_curvature_method2(k_neighbors)

        if self.curvatures is None:
            return False

        # 統計情報を計算
        stats = {
            "平均": np.mean(self.curvatures),
            "標準偏差": np.std(self.curvatures),
            "最小値": np.min(self.curvatures),
            "最大値": np.max(self.curvatures),
            "中央値": np.median(self.curvatures),
            "25%分位": np.percentile(self.curvatures, 25),
            "75%分位": np.percentile(self.curvatures, 75),
        }

        print("\n=== 曲率分布統計 ===")
        for key, value in stats.items():
            print(f"{key}: {value:.6f}")

        return True

    def visualize_results(self, save_plots=True):
        """結果を可視化"""
        if self.curvatures is None:
            print("先に曲率を計算してください")
            return

        # 図のスタイル設定
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. 曲率分布のヒストグラム
        fig = plt.figure(figsize=(15, 12))

        # ヒストグラム
        plt.subplot(2, 3, 1)
        plt.hist(
            self.curvatures, bins=50, alpha=0.7, color="skyblue", edgecolor="black"
        )
        plt.xlabel("曲率")
        plt.ylabel("頻度")
        plt.title("曲率分布ヒストグラム")
        plt.grid(True, alpha=0.3)

        # ボックスプロット
        plt.subplot(2, 3, 2)
        plt.boxplot(self.curvatures)
        plt.ylabel("曲率")
        plt.title("曲率のボックスプロット")
        plt.grid(True, alpha=0.3)

        # 累積分布関数
        plt.subplot(2, 3, 3)
        sorted_curvatures = np.sort(self.curvatures)
        cumulative = np.arange(1, len(sorted_curvatures) + 1) / len(sorted_curvatures)
        plt.plot(sorted_curvatures, cumulative, linewidth=2)
        plt.xlabel("曲率")
        plt.ylabel("累積確率")
        plt.title("曲率の累積分布関数")
        plt.grid(True, alpha=0.3)

        # 3D点群（曲率でカラーマップ）
        ax = plt.subplot(2, 3, 4, projection="3d")
        scatter = ax.scatter(
            self.points[:, 0],
            self.points[:, 1],
            self.points[:, 2],
            c=self.curvatures,
            cmap="viridis",
            s=1,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("点群（曲率カラーマップ）")
        plt.colorbar(scatter, shrink=0.5, aspect=5)

        # Q-Qプロット（正規分布との比較）
        plt.subplot(2, 3, 5)
        from scipy import stats

        stats.probplot(self.curvatures, dist="norm", plot=plt)
        plt.title("Q-Q プロット（正規分布との比較）")
        plt.grid(True, alpha=0.3)

        # 曲率の2Dマップ（上面図）
        plt.subplot(2, 3, 6)
        scatter = plt.scatter(
            self.points[:, 0], self.points[:, 1], c=self.curvatures, cmap="viridis", s=1
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("曲率マップ（上面図）")
        plt.colorbar(scatter)

        plt.tight_layout()

        if save_plots:
            plt.savefig("curvature_analysis.png", dpi=300, bbox_inches="tight")

        plt.show()

    def save_results(self, output_file="curvature_results.txt"):
        """結果をファイルに保存"""
        if self.curvatures is None:
            print("先に曲率を計算してください")
            return

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("点群曲率解析結果\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"入力ファイル: {self.ply_file_path}\n")
            f.write(f"点群数: {len(self.points)}\n\n")

            f.write("曲率統計:\n")
            f.write(f"平均: {np.mean(self.curvatures):.6f}\n")
            f.write(f"標準偏差: {np.std(self.curvatures):.6f}\n")
            f.write(f"最小値: {np.min(self.curvatures):.6f}\n")
            f.write(f"最大値: {np.max(self.curvatures):.6f}\n")
            f.write(f"中央値: {np.median(self.curvatures):.6f}\n")
            f.write(f"25%分位: {np.percentile(self.curvatures, 25):.6f}\n")
            f.write(f"75%分位: {np.percentile(self.curvatures, 75):.6f}\n")

        print(f"結果を {output_file} に保存しました")


def main():
    """メイン関数 - 使用例"""
    # PLYファイルのパスを指定
    ply_file_path = "processed_ply/iphone_60fps/segment8/clusters/berry_cluster_4.ply"  # 実際のファイルパスに変更してください

    # 解析器を初期化
    analyzer = PointCloudCurvatureAnalyzer(ply_file_path)

    # PLYファイルを読み込み
    if not analyzer.load_ply_file():
        print("PLYファイルの読み込みに失敗しました")
        return

    # 法線ベクトルを推定
    if not analyzer.estimate_normals(k_neighbors=30):
        print("法線ベクトルの計算に失敗しました")
        return

    # 曲率分布を解析（方法1を使用）
    if not analyzer.analyze_curvature_distribution(method=1, k_neighbors=30):
        print("曲率計算に失敗しました")
        return

    # 結果を可視化
    analyzer.visualize_results(save_plots=True)

    # 結果をファイルに保存
    analyzer.save_results()


if __name__ == "__main__":
    main()


# 簡単な使用例（コメントアウト）
"""
# 基本的な使用方法：
analyzer = PointCloudCurvatureAnalyzer("your_file.ply")
analyzer.load_ply_file()
analyzer.estimate_normals()
analyzer.analyze_curvature_distribution()
analyzer.visualize_results()

# パラメータを調整した使用方法：
analyzer = PointCloudCurvatureAnalyzer("your_file.ply")
analyzer.load_ply_file()
analyzer.estimate_normals(k_neighbors=50)  # より多くの近傍点を使用
analyzer.analyze_curvature_distribution(method=2, k_neighbors=50)  # 方法2を使用
analyzer.visualize_results()
"""
