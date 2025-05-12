import argparse
import random
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


class Sphere:
    """Sphere shape defined by center and radius."""

    def __init__(self, center=None, radius=None, min_radius=None, max_radius=None):
        self.center = center
        self.radius = radius
        self.min_radius = min_radius  # Minimum allowed radius
        self.max_radius = max_radius  # Maximum allowed radius

    def get_parameters(self):
        return {"center": self.center, "radius": self.radius}

    def estimate(self, points, normals):
        """Estimate sphere from four points."""
        if len(points) < 4:
            return False

        # Set up linear system to solve for sphere parameters
        A = np.zeros((4, 4))
        b = np.zeros(4)

        for i in range(4):
            A[i, 0:3] = points[i] * 2
            A[i, 3] = 1
            b[i] = np.sum(points[i] ** 2)

        try:
            # Solve linear system
            x = np.linalg.solve(A, b)

            # Extract center and radius
            center = x[0:3]
            radius = np.sqrt(np.sum(center**2) - x[3])

            # Check if radius is within allowed range
            if radius <= 0:
                return False

            if self.min_radius is not None and radius < self.min_radius:
                return False

            if self.max_radius is not None and radius > self.max_radius:
                return False

            # Verify normal consistency if normals are provided
            if normals is not None:
                for i, n in enumerate(normals):
                    v = points[i] - center
                    v_norm = np.linalg.norm(v)
                    if v_norm < 1e-10:
                        return False

                    # Normal should be radial
                    radial = v / v_norm
                    # CloudCompare uses a more relaxed constraint
                    if abs(np.dot(n, radial)) < 0.5:  # cos(60°) ≈ 0.5
                        return False

            self.center = center
            self.radius = radius
            return True
        except np.linalg.LinAlgError:
            return False

    def compute_score(self, points, normals, epsilon, alpha):
        """Compute score for this sphere using CloudCompare compatible approach."""
        if self.center is None or self.radius is None:
            return 0, np.zeros(len(points), dtype=bool)

        # Apply radius constraints
        if self.min_radius is not None and self.radius < self.min_radius:
            return 0, np.zeros(len(points), dtype=bool)

        if self.max_radius is not None and self.radius > self.max_radius:
            return 0, np.zeros(len(points), dtype=bool)

        # Vectors from center to points
        vectors = points - self.center

        # Distances to sphere surface
        distances_to_center = np.linalg.norm(vectors, axis=1)
        distances = np.abs(distances_to_center - self.radius)

        # CloudCompare uses an adaptive epsilon based on sphere size
        adaptive_epsilon = max(
            epsilon, self.radius * 0.02
        )  # 2% of radius or epsilon, whichever is larger

        # Check normal consistency if normals are provided
        if normals is not None:
            # Normalize vectors
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized_vectors = vectors / distances_to_center[:, np.newaxis]

            # Replace nan values with zeros
            normalized_vectors = np.nan_to_num(normalized_vectors)

            # Check if normals point radially (using CloudCompare's more relaxed constraint)
            dot_products = np.abs(np.sum(normalized_vectors * normals, axis=1))

            # CloudCompare uses a more relaxed constraint for larger spheres
            if self.radius > 1.0:
                # More relaxed constraint for larger spheres
                normal_constraint = 0.5  # cos(60°)
            else:
                normal_constraint = np.cos(alpha)

            normal_inliers = dot_products > normal_constraint
            inliers = (distances < adaptive_epsilon) & normal_inliers
        else:
            inliers = distances < adaptive_epsilon

        return np.sum(inliers), inliers

    def refine(self, points, normals, epsilon, alpha):
        """Refine sphere parameters using inlier points."""
        if len(points) < 4:
            return False

        # Use least squares to fit sphere to points
        A = np.zeros((len(points), 4))
        b = np.zeros(len(points))

        for i, point in enumerate(points):
            A[i, 0:3] = point * 2
            A[i, 3] = 1
            b[i] = np.sum(point**2)

        try:
            # Solve linear system using SVD for better stability
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

            # Extract center and radius
            center = x[0:3]
            radius = np.sqrt(np.sum(center**2) - x[3])

            # Check radius constraints
            if radius <= 0:
                return False

            if self.min_radius is not None and radius < self.min_radius:
                return False

            if self.max_radius is not None and radius > self.max_radius:
                return False

            self.center = center
            self.radius = radius
            return True
        except:
            return False

    def get_o3d_geometry(self, color=None):
        """Create a visualization for the sphere using Open3D."""
        if self.center is None or self.radius is None:
            return None

        if color is None:
            color = [0.0, 1.0, 0.0]  # Default green

        # Create a sphere mesh
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=self.radius, resolution=20
        )
        mesh_sphere.compute_vertex_normals()
        mesh_sphere.paint_uniform_color(color)

        # Translate to center
        mesh_sphere.translate(self.center)

        return mesh_sphere


def preprocess_point_cloud(
    pcd,
    voxel_size=None,
    estimate_normals=True,
    normal_neighbors=30,
    remove_outliers=True,
):
    """
    CloudCompare互換の前処理を行う関数

    Args:
        pcd: Open3Dの点群オブジェクト
        voxel_size: ダウンサンプリングのボクセルサイズ（None=処理なし）
        estimate_normals: 法線を推定するかどうか
        normal_neighbors: 法線推定に使用する近傍点数
        remove_outliers: 外れ値除去を行うかどうか

    Returns:
        処理済みの点群オブジェクト
    """
    # 点群のコピーを作成
    processed_pcd = o3d.geometry.PointCloud(pcd)

    # 外れ値除去（CloudCompareの処理に類似）
    if remove_outliers:
        print("Removing outliers...")
        # 統計的外れ値除去
        cl, ind = processed_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        processed_pcd = processed_pcd.select_by_index(ind)
        print(f"  Removed {len(pcd.points) - len(processed_pcd.points)} outlier points")

    # ダウンサンプリング（CloudCompareの処理に類似）
    if voxel_size is not None:
        print(f"Downsampling with voxel size {voxel_size}...")
        processed_pcd = processed_pcd.voxel_down_sample(voxel_size)
        print(f"  Points after downsampling: {len(processed_pcd.points)}")

    # 法線推定（CloudCompareの処理に類似）
    if estimate_normals:
        print(f"Estimating normals (neighbors: {normal_neighbors})...")
        processed_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_neighbors)
        )
        # 法線の向きを一貫させる（CloudCompareと同様）
        processed_pcd.orient_normals_consistent_tangent_plane(k=normal_neighbors)

    return processed_pcd


def cc_sphere_sampling(
    points,
    normals,
    indices,
    num_samples,
    max_radius=None,
    min_radius=None,
    tree=None,
    bbox_diagonal=None,
):
    """
    CloudCompareの球体検出サンプリングに合わせた実装
    """
    if len(indices) < num_samples:
        return indices

    if num_samples == 1:
        return [random.choice(indices)]

    # 最初の点をランダムに選択
    sample_indices = [random.choice(indices)]

    # サンプリング半径の設定
    if bbox_diagonal is None:
        bbox_diagonal = np.linalg.norm(np.ptp(points, axis=0))

    # CloudCompareスタイルのサンプリング半径
    if max_radius is not None:
        # 小さな球体検出の場合でも十分な範囲をカバー
        sampling_radius = max(max_radius * 3.0, bbox_diagonal * 0.05)
    else:
        sampling_radius = bbox_diagonal * 0.1

    # 残りの点を選択
    remaining = set(indices) - set(sample_indices)

    # 法線情報があれば活用
    if normals is not None:
        # 最初の点の法線
        first_normal = normals[sample_indices[0]]

        # 法線が類似している点を優先
        normal_similarity = []
        for idx in list(remaining)[: min(1000, len(remaining))]:  # 効率化のため上限設定
            similarity = abs(np.dot(first_normal, normals[idx]))
            normal_similarity.append((idx, similarity))

        # 類似度の高い順にソート
        normal_similarity.sort(key=lambda x: x[1], reverse=True)

        # 上位30%の点を候補とする
        candidate_count = max(int(len(normal_similarity) * 0.3), 10)
        candidates = [x[0] for x in normal_similarity[:candidate_count]]

        # 候補から2点目をランダムに選択
        if candidates:
            next_idx = random.choice(candidates)
            sample_indices.append(next_idx)
            remaining.remove(next_idx)

    # KD-treeの初期化
    if tree is None:
        tree = cKDTree(points)

    # 3点目と4点目を選択
    while len(sample_indices) < num_samples and remaining:
        # 既にサンプリングした点からランダムに選ぶ
        seed_idx = random.randint(0, len(sample_indices) - 1)
        seed_point = points[sample_indices[seed_idx]]

        # 一定半径内の点を見つける
        nearby_indices = tree.query_ball_point(seed_point, sampling_radius)

        # まだ使っていない点だけをフィルタリング
        nearby_valid = [idx for idx in nearby_indices if idx in remaining]

        if nearby_valid:
            # 既存の点からできるだけ離れた点を選ぶ
            if len(sample_indices) >= 2:
                farthest_dist = -1
                best_idx = None

                for idx in nearby_valid[: min(50, len(nearby_valid))]:
                    p = points[idx]
                    min_dist = min(
                        [np.linalg.norm(p - points[i]) for i in sample_indices]
                    )

                    if min_dist > farthest_dist:
                        farthest_dist = min_dist
                        best_idx = idx

                if best_idx is not None:
                    sample_indices.append(best_idx)
                    remaining.remove(best_idx)
                    continue

            # デフォルト：近傍からランダムに選ぶ
            next_idx = random.choice(nearby_valid)
            sample_indices.append(next_idx)
            remaining.remove(next_idx)
        else:
            # 近くに点がない場合はランダムに選ぶ
            next_idx = random.choice(list(remaining))
            sample_indices.append(next_idx)
            remaining.remove(next_idx)

    return sample_indices


class SphereDetector:
    """CloudCompare互換の球体検出クラス"""

    def __init__(
        self,
        points,
        normals=None,
        epsilon=0.01,
        alpha=np.radians(20),
        min_points=30,
        probability=0.99,
        min_radius=None,
        max_radius=None,
    ):
        """
        Initialize the detector.

        Args:
            points: point coordinates [n, 3]
            normals: point normals [n, 3] or None
            epsilon: distance threshold for inliers
            alpha: angle threshold in radians for normal consistency
            min_points: minimum points for a valid sphere
            probability: desired detection probability
            min_radius: minimum radius for sphere detection
            max_radius: maximum radius for sphere detection
        """
        self.points = np.asarray(points)
        self.normals = normals
        self.epsilon = epsilon
        self.alpha = alpha
        self.min_points = min_points
        self.probability = probability
        self.min_radius = min_radius
        self.max_radius = max_radius

        # Build KD-tree for efficient spatial queries
        self.tree = cKDTree(points)

        # Initialize data structures for shape detection
        self.spheres = []
        self.sphere_points = []
        self.remaining_points = np.ones(len(points), dtype=bool)

        # Compute diagonal of bounding box for radius scaling
        self.bbox_diagonal = np.linalg.norm(np.ptp(points, axis=0))

    def estimate_normals(self, k_neighbors=30):
        """Estimate normals if they weren't provided."""
        print("Estimating point normals...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
        )
        pcd.orient_normals_consistent_tangent_plane(k=k_neighbors)
        self.normals = np.asarray(pcd.normals)
        return self.normals

    def detect(self, max_iterations=1000, min_score=None):
        """
        Detect spheres in the point cloud.

        Args:
            max_iterations: maximum number of iterations to run
            min_score: minimum score required for a valid sphere

        Returns:
            List of detected spheres
        """
        print("Detecting spheres...")

        if min_score is None:
            min_score = max(
                10, int(0.01 * len(self.points))
            )  # At least 1% of points or 10 points

        # Initialize remaining points (all points are initially remaining)
        self.remaining_points = np.ones(len(self.points), dtype=bool)

        # Main loop
        iteration = 0
        while (
            np.sum(self.remaining_points) > self.min_points
            and iteration < max_iterations
        ):
            # Sample points only from remaining points
            remaining_indices = np.where(self.remaining_points)[0]

            # Break if too few points remain
            if len(remaining_indices) < self.min_points:
                break

            # Try to detect a new sphere
            best_sphere = None
            best_score = 0
            best_inliers = None

            # Compute the number of trials needed for the current probability
            num_trials = self.compute_num_trials(
                0.01 if best_score == 0 else best_score / len(remaining_indices)
            )
            max_trials = min(1000, num_trials)  # Limit trials for efficiency

            trial = 0
            while trial < max_trials:
                # Create a new sphere with radius constraints
                sphere = Sphere(min_radius=self.min_radius, max_radius=self.max_radius)

                # Draw minimal sample using CloudCompare compatible sampling
                sample_indices = cc_sphere_sampling(
                    self.points,
                    self.normals,
                    remaining_indices,
                    4,  # 4 points needed for sphere
                    max_radius=self.max_radius,
                    min_radius=self.min_radius,
                    tree=self.tree,
                    bbox_diagonal=self.bbox_diagonal,
                )

                if len(sample_indices) < 4:
                    trial += 1
                    continue

                # Get points and normals for sample
                sample_points = self.points[sample_indices]
                sample_normals = (
                    None if self.normals is None else self.normals[sample_indices]
                )

                # Try to estimate sphere parameters
                if not sphere.estimate(sample_points, sample_normals):
                    trial += 1
                    continue

                # Compute score using CloudCompare compatible method
                score, inliers = sphere.compute_score(
                    self.points[remaining_indices],
                    None if self.normals is None else self.normals[remaining_indices],
                    self.epsilon,
                    self.alpha,
                )

                if score > best_score:
                    best_score = score
                    best_sphere = sphere
                    best_inliers = inliers

                    # If we have a very good sphere (>50% inliers), refine and break
                    if score > 0.5 * len(remaining_indices):
                        break

                trial += 1

            # If no sphere was found, break
            if best_sphere is None:
                break

            # Refine the sphere with all its inliers
            inlier_global_indices = remaining_indices[best_inliers]

            # Only refine if we have enough points
            if len(inlier_global_indices) >= self.min_points:
                refinement_success = best_sphere.refine(
                    self.points[inlier_global_indices],
                    (
                        None
                        if self.normals is None
                        else self.normals[inlier_global_indices]
                    ),
                    self.epsilon,
                    self.alpha,
                )

                # If refinement fails (e.g., due to radius constraints), skip this sphere
                if not refinement_success:
                    iteration += 1
                    continue

                # Recalculate score with refined sphere
                score, inliers = best_sphere.compute_score(
                    self.points[remaining_indices],
                    None if self.normals is None else self.normals[remaining_indices],
                    self.epsilon,
                    self.alpha,
                )

                inlier_global_indices = remaining_indices[inliers]

                # Make sure we still have enough inliers
                if len(inlier_global_indices) >= self.min_points and score >= min_score:
                    # Add sphere to the list of detected spheres
                    self.spheres.append(best_sphere)
                    self.sphere_points.append(inlier_global_indices)

                    # Remove inliers from the remaining points
                    self.remaining_points[inlier_global_indices] = False

                    print(
                        f"Iteration {iteration}: Found sphere with {len(inlier_global_indices)} points"
                    )

            iteration += 1

        print(f"Detected {len(self.spheres)} spheres")
        return self.spheres

    def compute_num_trials(self, inlier_ratio):
        """Compute the number of trials needed for the current probability."""
        if inlier_ratio < 1e-6:
            return 10000  # Large number for very low inlier ratio

        # Probability of selecting all inliers in a minimal sample (4 points for sphere)
        success_prob = (inlier_ratio) ** 4

        # Handle very low probabilities
        if success_prob < 1e-8:
            return 10000

        # Formula from the RANSAC paper: T ≥ ln(1 - p_t) / ln(1 - P(n))
        num_trials = np.log(1.0 - self.probability) / np.log(1.0 - success_prob)

        return max(100, int(np.ceil(num_trials)))


def load_point_cloud(filename):
    """Load point cloud from file (txt or ply)."""
    if filename.endswith(".ply"):
        # Load PLY file using Open3D
        pcd = o3d.io.read_point_cloud(filename)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    else:
        # Assume text file with coordinates
        try:
            data = np.loadtxt(filename, dtype=np.float32)

            # Check shape of data
            if data.ndim == 1:
                # Single row - reshape based on assumptions
                if len(data) % 3 == 0:  # If divisible by 3, assume just points
                    points = data.reshape(-1, 3)
                    normals = None
                elif len(data) % 6 == 0:  # If divisible by 6, assume points+normals
                    n_points = len(data) // 6
                    points = data[: n_points * 3].reshape(-1, 3)
                    normals = data[n_points * 3 :].reshape(-1, 3)
                else:
                    raise ValueError(f"Cannot parse data of length {len(data)}")
            else:
                # Multiple rows
                if data.shape[1] == 3:  # Just points
                    points = data
                    normals = None
                elif data.shape[1] == 6:  # Points and normals
                    points = data[:, 0:3]
                    normals = data[:, 3:6]
                else:
                    raise ValueError(f"Unexpected data shape: {data.shape}")
        except Exception as e:
            raise ValueError(f"Error loading point cloud from {filename}: {e}")

    return points, normals


def visualize_results(points, spheres, sphere_points=None, remaining_points=None):
    """Visualize detection results using Open3D."""
    print("Visualizing results with Open3D...")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()

    # Add remaining points (gray)
    geometries = []
    if remaining_points is not None and np.sum(remaining_points) > 0:
        unassigned_pcd = o3d.geometry.PointCloud()
        unassigned_pcd.points = o3d.utility.Vector3dVector(points[remaining_points])
        unassigned_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
        geometries.append(unassigned_pcd)

    # Colors for different spheres
    colors = [
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 0.0, 0.0],  # Red
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
        [1.0, 0.5, 0.0],  # Orange
        [0.5, 0.0, 1.0],  # Purple
    ]

    # Add each sphere's points and geometry
    for i, (sphere, pts_idx) in enumerate(zip(spheres, sphere_points)):
        color = colors[i % len(colors)]

        # Add sphere's points
        shape_pcd = o3d.geometry.PointCloud()
        shape_pcd.points = o3d.utility.Vector3dVector(points[pts_idx])
        shape_pcd.paint_uniform_color(color)
        geometries.append(shape_pcd)

        # Add sphere geometry
        sphere_geo = sphere.get_o3d_geometry(color)
        if sphere_geo is not None:
            geometries.append(sphere_geo)

    # Add coordinate system
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coord_frame)

    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name="CloudCompare Compatible Sphere Detection",
        width=1280,
        height=720,
        point_show_normal=False,
    )


def main():
    """Main function to run the shape detection algorithm."""
    parser = argparse.ArgumentParser(
        description="CloudCompare Compatible Sphere Detector"
    )
    parser.add_argument("input_file", help="Input point cloud file (txt or ply)")
    parser.add_argument(
        "--epsilon", type=float, default=0.01, help="Distance threshold for inliers"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=20,
        help="Angle threshold in degrees for normal consistency",
    )
    parser.add_argument(
        "--min-points", type=int, default=30, help="Minimum points for a valid sphere"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=1000, help="Maximum iterations"
    )
    parser.add_argument(
        "--min-radius", type=float, help="Minimum radius for sphere detection"
    )
    parser.add_argument(
        "--max-radius", type=float, help="Maximum radius for sphere detection"
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Preprocess the point cloud"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        help="Voxel size for downsampling during preprocessing",
    )
    parser.add_argument("--output", help="Output file for remaining points (ply)")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    args = parser.parse_args()

    try:
        # Load point cloud
        points, normals = load_point_cloud(args.input_file)
        print(f"Loaded {len(points)} points")

        # Preprocess the point cloud if requested
        if args.preprocess:
            print("Preprocessing point cloud...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(normals)

            # Preprocess the point cloud
            processed_pcd = preprocess_point_cloud(
                pcd,
                voxel_size=args.voxel_size,
                estimate_normals=True,
                normal_neighbors=30,
                remove_outliers=True,
            )

            # Get processed points and normals
            points = np.asarray(processed_pcd.points)
            normals = np.asarray(processed_pcd.normals)
            print(f"After preprocessing: {len(points)} points")

        # Print radius constraints if specified
        radius_info = []
        if args.min_radius is not None:
            radius_info.append(f"min_radius={args.min_radius}")
        if args.max_radius is not None:
            radius_info.append(f"max_radius={args.max_radius}")
        if radius_info:
            print(f"Sphere constraints: {', '.join(radius_info)}")

        # Estimate normals if required and not provided
        if normals is None:
            detector = SphereDetector(points)
            normals = detector.estimate_normals()
            print(f"Estimated {len(normals)} normals")

        # Initialize detector
        detector = SphereDetector(
            points,
            normals,
            epsilon=args.epsilon,
            alpha=np.radians(args.alpha),
            min_points=args.min_points,
            probability=0.99,
            min_radius=args.min_radius,
            max_radius=args.max_radius,
        )

        # Detect spheres
        start_time = time.time()
        spheres = detector.detect(max_iterations=args.max_iterations)
        end_time = time.time()

        print(f"Detection completed in {end_time - start_time:.2f} seconds")

        # Print results
        for i, sphere in enumerate(spheres):
            params = sphere.get_parameters()
            points_idx = detector.sphere_points[i]
            print(f"Sphere {i+1}: with {len(points_idx)} points")
            print(f"  center: [{', '.join(f'{v:.4f}' for v in params['center'])}]")
            print(f"  radius: {params['radius']:.4f}")

        # Visualize if requested
        if args.visualize:
            visualize_results(
                points, spheres, detector.sphere_points, detector.remaining_points
            )

        # Save remaining points if output file is specified
        if args.output and args.output.endswith(".ply"):
            # Create point cloud with remaining points
            remaining_pcd = o3d.geometry.PointCloud()
            remaining_points = points[detector.remaining_points]
            remaining_pcd.points = o3d.utility.Vector3dVector(remaining_points)

            # Add normals if available
            if normals is not None:
                remaining_normals = normals[detector.remaining_points]
                remaining_pcd.normals = o3d.utility.Vector3dVector(remaining_normals)

            # Write to file
            o3d.io.write_point_cloud(args.output, remaining_pcd)
            print(
                f"Saved {np.sum(detector.remaining_points)} remaining points to {args.output}"
            )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    """
    uv run ransac.py result/room_4k5/output.ply --max-radius 0.9 --min-points 100 --visualize\
        --epsilon 0.05 --alpha 30 --max-iterations 1000 
    """
    sys.exit(main())
