import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d


@dataclass
class OctreeCell:
    """Octree cell for level-weighted sampling"""

    level: int
    points: np.ndarray
    normals: np.ndarray
    score: float = 0.0


class ScoreStatistics:
    """Keep track of score statistics for each level"""

    def __init__(self):
        self.level_scores: Dict[int, float] = {}
        self.level_counts: Dict[int, int] = {}

    def update(self, level: int, score: float):
        if level not in self.level_scores:
            self.level_scores[level] = 0.0
            self.level_counts[level] = 0
        self.level_scores[level] += score
        self.level_counts[level] += 1

    def get_weight(self, level: int) -> float:
        if level not in self.level_scores or self.level_counts[level] == 0:
            return 0.0
        return self.level_scores[level] / self.level_counts[level]


class Octree:
    """Octree structure for level-weighted sampling"""

    def __init__(self, points: np.ndarray, normals: np.ndarray, min_points: int = 10):
        self.points = points
        self.normals = normals
        self.min_points = min_points
        self.cells: List[OctreeCell] = []
        self.stats = ScoreStatistics()
        self._build_octree()

    def _build_octree(self):
        """Build octree structure recursively"""
        self._recursive_split(self.points, self.normals, 0)

    def _recursive_split(self, points: np.ndarray, normals: np.ndarray, level: int):
        if len(points) < self.min_points:
            return

        # Create cell
        cell = OctreeCell(level, points, normals)
        self.cells.append(cell)

        if len(points) < self.min_points * 2:
            return

        # Split space into octants
        center = np.mean(points, axis=0)
        mask_list = []
        for i in range(8):
            mask = np.ones(len(points), dtype=bool)
            for j in range(3):
                if (i >> j) & 1:
                    mask &= points[:, j] >= center[j]
                else:
                    mask &= points[:, j] < center[j]
            mask_list.append(mask)

            if np.sum(mask) >= self.min_points:
                self._recursive_split(points[mask], normals[mask], level + 1)

    def sample_cell(self) -> Optional[OctreeCell]:
        """Sample cell based on level weights"""
        if not self.cells:
            return None

        # Calculate sampling probabilities
        weights = np.array([self.stats.get_weight(cell.level) for cell in self.cells])
        total_weight = np.sum(weights)
        if total_weight == 0:
            # Initial case: uniform sampling
            weights = np.ones(len(self.cells)) / len(self.cells)
        else:
            weights /= total_weight

        # Sample cell
        return random.choices(self.cells, weights=weights)[0]


def compute_bitmap_resolution(points: np.ndarray) -> float:
    """Compute optimal bitmap resolution based on point density"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Sample points to estimate average distance
    sample_size = min(1000, len(points))
    distances = []
    for i in random.sample(range(len(points)), sample_size):
        _, idx, dist = kdtree.search_knn_vector_3d(points[i], 2)
        distances.append(np.sqrt(dist[1]))  # Distance to nearest neighbor

    return np.median(distances)


def compute_score(
    points: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    radius: float,
    distance_threshold: float,
    normal_threshold: float,
    bitmap_resolution: float,
) -> Tuple[float, np.ndarray]:
    """
    Compute score for a sphere candidate following the paper's methodology
    Returns both score and inlier indices
    """
    # Compute distances and normals
    distances = np.abs(np.linalg.norm(points - center.reshape(1, 3), axis=1) - radius)

    directions = points - center.reshape(1, 3)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    normal_deviations = np.abs(np.sum(directions * normals, axis=1) - 1)

    # Find potential inliers
    potential_inliers = np.where(
        (distances < distance_threshold) & (normal_deviations < normal_threshold)
    )[0]

    if len(potential_inliers) < 3:
        return 0.0, np.array([], dtype=int)

    # Find connected components in parameter space
    inlier_points = points[potential_inliers]
    connected_mask = find_largest_connected_component(inlier_points, bitmap_resolution)
    final_inliers = potential_inliers[connected_mask]

    if len(final_inliers) < 3:
        return 0.0, np.array([], dtype=int)

    # Compute score based on number of inliers and their quality
    distance_scores = 1 - distances[final_inliers] / distance_threshold
    normal_scores = 1 - normal_deviations[final_inliers] / normal_threshold
    point_scores = distance_scores * normal_scores

    # Final score is average point score
    score = np.mean(point_scores)

    return score, final_inliers


def refit_sphere(
    points: np.ndarray,
    normals: np.ndarray,
    initial_center: np.ndarray,
    initial_radius: float,
    max_iterations: int = 10,
) -> Tuple[np.ndarray, float]:
    """
    Refit sphere parameters to minimize error using all inlier points
    """
    center = initial_center.copy()
    radius = initial_radius

    for _ in range(max_iterations):
        # Compute vectors from center to points
        vectors = points - center.reshape(1, 3)
        distances = np.linalg.norm(vectors, axis=1)

        # Update center
        center_update = np.mean(vectors * (distances - radius).reshape(-1, 1), axis=0)

        # Update radius
        radius = np.mean(distances)

        # Check convergence
        if np.linalg.norm(center_update) < 1e-6:
            break

        center += center_update

    return center, radius


def load_points_with_normals(
    filename: str, k_neighbors: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load point cloud and compute normals

    Args:
        filename: Path to point cloud file (.txt or .ply)
        k_neighbors: Number of neighbors to use for normal estimation

    Returns:
        points: Nx3 array of point coordinates
        normals: Nx3 array of normal vectors
    """
    # Load points
    if filename.endswith(".txt"):
        points = np.loadtxt(filename)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    elif filename.endswith(".ply"):
        pcd = o3d.io.read_point_cloud(filename)
        points = np.asarray(pcd.points)
    else:
        raise ValueError("Unsupported file format. Use .txt or .ply")

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors),
        fast_normal_computation=False,
    )
    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k_neighbors)

    normals = np.asarray(pcd.normals)
    return points, normals


def compute_sphere_from_points_and_normals(
    points: np.ndarray, normals: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Compute sphere parameters from two points and their normals
    following the paper's approach

    Args:
        points: 2x3 array containing two points
        normals: 2x3 array containing corresponding normal vectors

    Returns:
        center: 3D center point of sphere
        radius: radius of sphere
    """
    p1, p2 = points
    n1, n2 = normals

    # Normalize normals to ensure they are unit vectors
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    # Find the intersection of the two lines defined by points and normals
    # Use the method described in the paper

    # Direction between points
    v = p2 - p1

    # Check if points are too close or normals are parallel
    if np.linalg.norm(v) < 1e-10 or abs(np.dot(n1, n2)) > 0.999:
        return None, None

    # Compute center using the midpoint of the shortest line segment
    # between the two lines defined by points and normals

    # Matrix A for solving the parameters
    A = np.array([n1, -n2]).T
    b = p2 - p1

    try:
        # Solve for parameters
        t = np.linalg.solve(A.T @ A, A.T @ b)

        # Compute closest points on both lines
        c1 = p1 + t[0] * n1
        c2 = p2 + t[1] * n2

        # Center is midpoint between closest points
        center = (c1 + c2) / 2

        # Radius is distance from center to either point
        radius = np.linalg.norm(center - p1)

        return center, radius
    except np.linalg.LinAlgError:
        return None, None


def find_largest_connected_component(
    points: np.ndarray, threshold: float = 0.1
) -> np.ndarray:
    """
    Find the largest connected component in a point cloud.

    Args:
        points: Nx3 array of point coordinates
        threshold: Distance threshold for considering points connected

    Returns:
        Boolean mask indicating points in the largest component
    """
    N = len(points)
    if N == 0:
        return np.array([], dtype=bool)

    # Create a point cloud for nearest neighbor search
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Initialize variables for connected components
    visited = np.zeros(N, dtype=bool)
    components = []

    # Find connected components
    for i in range(N):
        if visited[i]:
            continue

        # Start new component
        component = {i}
        queue = [i]
        visited[i] = True

        # Grow component
        while queue:
            current = queue.pop(0)
            # Find neighbors within threshold
            _, idx, dist = kdtree.search_radius_vector_3d(points[current], threshold)

            # Add unvisited neighbors to component
            for j, d in zip(idx, dist):
                if not visited[j]:
                    visited[j] = True
                    component.add(j)
                    queue.append(j)

        components.append(component)

    # Find largest component
    if not components:
        return np.zeros(N, dtype=bool)

    largest = max(components, key=len)
    mask = np.zeros(N, dtype=bool)
    mask[list(largest)] = True

    return mask


def select_close_points(
    points: np.ndarray, normals: np.ndarray, k_neighbors: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select two close points using k-nearest neighbors search

    Args:
        points: Nx3 array of point coordinates
        normals: Nx3 array of normal vectors
        k_neighbors: Number of neighbors to consider

    Returns:
        selected_points: 2x3 array containing two selected points
        selected_normals: 2x3 array containing corresponding normals
    """
    # Create point cloud for kd-tree search
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # Select first point randomly
    idx1 = random.randrange(len(points))
    p1 = points[idx1]

    # Find k nearest neighbors of the first point
    _, idx_list, _ = kdtree.search_knn_vector_3d(p1, k_neighbors)

    # Remove the first point itself from neighbors
    neighbors = [i for i in idx_list if i != idx1]

    if not neighbors:
        return None, None

    # Select second point randomly from neighbors
    idx2 = random.choice(neighbors)

    selected_points = points[[idx1, idx2]]
    selected_normals = normals[[idx1, idx2]]

    return selected_points, selected_normals


def ransac_sphere_detection_with_weights(
    points: np.ndarray,
    normals: np.ndarray,
    distance_threshold: float = 0.01,
    normal_threshold: float = 0.1,
    min_points: int = 10,
    max_iterations: int = 1000,
    confidence: float = 0.99,
) -> List[Tuple[np.ndarray, float, np.ndarray]]:
    """
    Improved RANSAC implementation with all paper features
    """
    # Initialize octree structure
    octree = Octree(points, normals, min_points)
    bitmap_resolution = compute_bitmap_resolution(points)

    remaining_points = points.copy()
    remaining_normals = normals.copy()
    spheres = []

    while len(remaining_points) >= min_points:
        best_score = 0.0
        best_inliers = None
        best_center = None
        best_radius = None

        for _ in range(max_iterations):
            # Sample cell based on level weights
            cell = octree.sample_cell()
            if cell is None:
                break

            # Sample points from cell
            if len(cell.points) < 2:
                continue

            sample_indices = random.sample(range(len(cell.points)), 2)
            sample_points = cell.points[sample_indices]
            sample_normals = cell.normals[sample_indices]

            # Compute candidate sphere
            center, radius = compute_sphere_from_points_and_normals(
                sample_points, sample_normals
            )
            if center is None:
                continue

            # Compute score and find inliers
            score, inliers = compute_score(
                remaining_points,
                remaining_normals,
                center,
                radius,
                distance_threshold,
                normal_threshold,
                bitmap_resolution,
            )

            # Update statistics
            octree.stats.update(cell.level, score)

            if score > best_score and len(inliers) >= min_points:
                best_score = score
                best_inliers = inliers
                best_center = center
                best_radius = radius

                # Early termination check
                inlier_ratio = len(inliers) / len(remaining_points)
                num_iterations = np.log(1 - confidence) / np.log(1 - inlier_ratio**2)
                max_iterations = min(max_iterations, int(num_iterations))

        if best_inliers is not None and len(best_inliers) >= min_points:
            # Refit sphere parameters
            final_center, final_radius = refit_sphere(
                remaining_points[best_inliers],
                remaining_normals[best_inliers],
                best_center,
                best_radius,
            )

            # Add to results
            spheres.append((final_center, final_radius, remaining_points[best_inliers]))

            # Remove inliers
            mask = np.ones(len(remaining_points), dtype=bool)
            mask[best_inliers] = False
            remaining_points = remaining_points[mask]
            remaining_normals = remaining_normals[mask]

            # Update octree
            octree = Octree(remaining_points, remaining_normals, min_points)
        else:
            break

    return spheres


def visualize_results(
    points: np.ndarray, spheres: List[Tuple[np.ndarray, float, np.ndarray]]
):
    """Visualize detected spheres using Open3D"""
    # Create point cloud for visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])  # Gray color for all points

    # Create visualization elements for each sphere
    geometries = [pcd]
    for i, (center, radius, inliers) in enumerate(spheres):
        # Create mesh for sphere
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_sphere.translate(center)
        mesh_sphere.paint_uniform_color([1, 0, 0])  # Red color for spheres
        mesh_sphere.compute_vertex_normals()

        # Create point cloud for inliers with different color
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(inliers)
        inlier_pcd.paint_uniform_color([0, 1, 0])  # Green color for inliers

        geometries.extend([mesh_sphere, inlier_pcd])

    # Visualize
    o3d.visualization.draw_geometries(geometries)


def main(filename: str):
    """Main function to run sphere detection with normals"""
    # Load points and compute normals
    points, normals = load_points_with_normals(filename)

    # Detect spheres
    spheres = ransac_sphere_detection_with_weights(
        points,
        normals,
        distance_threshold=0.1,
        normal_threshold=0.5,
        min_points=100,
        max_iterations=10000,
        confidence=0.5,
    )

    # Print results and visualize
    print(f"Detected {len(spheres)} spheres:")
    for i, (center, radius, inliers) in enumerate(spheres):
        print(f"Sphere {i+1}:")
        print(f"  Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        print(f"  Radius: {radius:.3f}")
        print(f"  Number of inliers: {len(inliers)}")

    visualize_results(points, spheres)


if __name__ == "__main__":
    filename = "result/room_360/output.ply"  # or "point_cloud.ply"
    main(filename)
