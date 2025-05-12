import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
from plyfile import PlyData
from sklearn.neighbors import KDTree


@dataclass
class SphereParams:
    center: np.ndarray
    radius: float
    score: int = 0
    inliers: List[int] = None


class SphereRANSAC:
    def __init__(
        self,
        distance_threshold: float = 0.02,
        normal_threshold: float = np.cos(np.pi / 6),  # 30度
        min_points: int = 100,
        max_iterations: int = 1000,
        early_stop_score: float = 0.8,
    ):
        """
        Parameters:
        -----------
        distance_threshold : float
            点と球面との距離の閾値
        normal_threshold : float
            法線方向の一致度の閾値（cos値）
        min_points : int
            検出する球体の最小点数
        max_iterations : int
            RANSACの最大反復回数
        early_stop_score : float
            早期終了のための閾値（0-1）
        """
        self.distance_threshold = distance_threshold
        self.normal_threshold = normal_threshold
        self.min_points = min_points
        self.max_iterations = max_iterations
        self.early_stop_score = early_stop_score

    def load_ply(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """PLYファイルから点群と法線を読み込む"""
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        if len(normals) == 0:
            # 法線が含まれていない場合は計算する
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            normals = np.asarray(pcd.normals)
        return points, normals

    def estimate_sphere_params(
        self, points: np.ndarray, normals: np.ndarray
    ) -> Optional[SphereParams]:
        """2点とその法線から球のパラメータを推定"""
        if len(points) != 2:
            return None

        p1, p2 = points
        n1, n2 = normals

        # 中心は2つの点から等距離にある必要がある
        mid_point = (p1 + p2) / 2
        direction = p2 - p1
        direction_norm = np.linalg.norm(direction)

        if direction_norm < 1e-10:  # 点が近すぎる場合
            return None

        # 中心は中点から法線方向に存在する
        try:
            # 連立方程式を解く
            # (center - p1) • n1 = radius
            # (center - p2) • n2 = radius
            # ||center - p1|| = ||center - p2|| = radius
            A = np.vstack([n1, n2])
            b = np.array([np.dot(mid_point, n1), np.dot(mid_point, n2)])
            center = np.linalg.solve(A, b)
            radius = np.linalg.norm(center - p1)

            return SphereParams(center=center, radius=radius)
        except np.linalg.LinAlgError:
            return None

    def compute_sphere_score(
        self, points: np.ndarray, normals: np.ndarray, sphere: SphereParams
    ) -> Tuple[int, List[int]]:
        """球面に対する点群のスコアを計算"""
        distances = np.linalg.norm(points - sphere.center, axis=1) - sphere.radius

        # 各点から中心への方向ベクトル
        directions = points - sphere.center
        directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

        # 法線との内積
        normal_angles = np.abs(np.sum(directions * normals, axis=1))

        # インライアの判定
        inliers = np.where(
            (np.abs(distances) < self.distance_threshold)
            & (normal_angles > self.normal_threshold)
        )[0]

        return len(inliers), inliers.tolist()

    def detect_sphere(
        self, points: np.ndarray, normals: np.ndarray
    ) -> Optional[SphereParams]:
        """RANSACによる球体検出"""
        best_sphere = None
        best_score = 0
        n_points = len(points)

        # KD木の構築（局所サンプリング用）
        kdtree = KDTree(points)

        for _ in range(self.max_iterations):
            # ランダムに1点目を選択
            idx1 = random.randrange(n_points)

            # 1点目の近傍から2点目を選択
            neighbors = kdtree.query_radius(
                [points[idx1]], r=self.distance_threshold * 10
            )[0]
            if len(neighbors) < 2:
                continue
            idx2 = neighbors[random.randrange(len(neighbors))]

            # 球パラメータの推定
            sphere = self.estimate_sphere_params(
                points=points[[idx1, idx2]], normals=normals[[idx1, idx2]]
            )
            if sphere is None:
                continue

            # スコア計算
            score, inliers = self.compute_sphere_score(points, normals, sphere)

            if score > best_score and score >= self.min_points:
                sphere.score = score
                sphere.inliers = inliers
                best_sphere = sphere
                best_score = score

                # 早期終了判定
                if score / n_points > self.early_stop_score:
                    break

        return best_sphere

    def visualize_result(
        self, points: np.ndarray, sphere: SphereParams, output_path: str = None
    ):
        """結果の可視化"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # インライアとアウトライアで色分け
        colors = np.zeros((len(points), 3))
        colors[sphere.inliers] = [1, 0, 0]  # インライアは赤
        colors[~np.isin(range(len(points)), sphere.inliers)] = [
            0.8,
            0.8,
            0.8,
        ]  # アウトライアは灰色
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 球のメッシュ作成
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere.radius)
        mesh_sphere.translate(sphere.center)
        mesh_sphere.paint_uniform_color([0, 1, 0])  # 緑色

        # 描画
        o3d.visualization.draw_geometries([pcd, mesh_sphere])

        if output_path:
            o3d.io.write_point_cloud(output_path, pcd)


def main():
    # 使用例
    ransac = SphereRANSAC(
        distance_threshold=0.2,
        normal_threshold=np.cos(np.pi / 6),
        min_points=100,
        max_iterations=1000,
        early_stop_score=0.8,
    )

    # PLYファイルから点群を読み込み
    points, normals = ransac.load_ply("result/room_360/output.ply")

    # 球体検出
    sphere = ransac.detect_sphere(points, normals)

    if sphere:
        print(f"Detected sphere:")
        print(f"Center: {sphere.center}")
        print(f"Radius: {sphere.radius}")
        print(f"Score: {sphere.score} points")

        # 結果の可視化
        ransac.visualize_result(points, sphere, "ransac_output.ply")
    else:
        print("No sphere detected.")


if __name__ == "__main__":
    main()
