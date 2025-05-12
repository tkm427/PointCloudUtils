import os
import random

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares


def load_point_cloud(filename):
    """点群データを読み込み、デバッグ情報を表示"""
    file_extension = os.path.splitext(filename)[1].lower()

    try:
        if file_extension == ".ply":
            print("PLYファイルを読み込みます...")
            pcd = o3d.io.read_point_cloud(filename)
            print(f"点群データの情報:")
            print(f"- 点の数: {len(pcd.points)}")
            print(f"- 法線の有無: {pcd.has_normals()}")
            print(f"- 色情報の有無: {pcd.has_colors()}")

            # 点群データの最初の数点を表示
            points = np.asarray(pcd.points)
            print("\n最初の5点のデータ:")
            print(points[:5])

            return points, pcd

        elif file_extension == ".txt":
            print("TEXTファイルを読み込みます...")
            points = np.loadtxt(filename)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            return points, pcd

        else:
            raise ValueError(f"未対応のファイル形式です: {file_extension}")

    except Exception as e:
        print(f"ファイル読み込み中にエラーが発生しました:")
        print(f"エラーの種類: {type(e).__name__}")
        print(f"エラーメッセージ: {str(e)}")
        raise


def distance_to_sphere(point, sphere_params):
    """点と球面との距離を計算"""
    center = sphere_params[:3]
    radius = sphere_params[3]
    return abs(np.linalg.norm(point - center) - radius)


def estimate_sphere_from_points(points):
    """4点から球体のパラメータを推定"""
    if len(points) < 4:
        return None

    A = np.zeros((4, 4))
    b = np.zeros(4)

    for i in range(4):
        x, y, z = points[i]
        A[i] = [2 * x, 2 * y, 2 * z, -1]
        b[i] = x * x + y * y + z * z

    try:
        solution = np.linalg.solve(A, b)
        center = solution[:3]
        radius = np.sqrt(np.sum(center * center) - solution[3])
        return np.array([*center, radius])
    except np.linalg.LinAlgError:
        return None


def optimize_sphere(points, initial_params):
    """最小二乗法で球体パラメータを最適化"""

    def objective(params):
        return [distance_to_sphere(point, params) for point in points]

    result = least_squares(objective, initial_params)
    return result.x


def detect_sphere_ransac(
    points, distance_threshold=0.5, max_iterations=1000, min_inliers=100
):
    """RANSACパラメータを調整した球体検出"""
    best_sphere = None
    max_inliers_count = 0

    # スケールに基づいて距離閾値を調整
    points_scale = np.max(np.abs(points)) * 0.05  # データスケールの5%を閾値に
    distance_threshold = max(distance_threshold, points_scale)

    print(f"RANSAC パラメータ:")
    print(f"- 距離閾値: {distance_threshold}")
    print(f"- 最大イテレーション数: {max_iterations}")
    print(f"- 最小インライア数: {min_inliers}")

    # データをサブサンプリング（処理速度向上のため）
    if len(points) > 10000:
        indices = np.random.choice(len(points), 10000, replace=False)
        points_subset = points[indices]
    else:
        points_subset = points

    for i in range(max_iterations):
        if i % 100 == 0:
            print(f"RANSAC進捗: {i}/{max_iterations}")

        try:
            # 4点をランダムに選択
            sample_indices = random.sample(range(len(points_subset)), 4)
            sample_points = points_subset[sample_indices]

            # 球体パラメータを推定
            sphere_params = estimate_sphere_from_points(sample_points)
            if sphere_params is None:
                continue

            # インライアをカウント（全点群に対して）
            inliers = []
            inlier_indices = []
            for i, point in enumerate(points):
                if distance_to_sphere(point, sphere_params) < distance_threshold:
                    inliers.append(point)
                    inlier_indices.append(i)

            inliers_count = len(inliers)
            if inliers_count > max_inliers_count and inliers_count >= min_inliers:
                max_inliers_count = inliers_count
                optimized_params = optimize_sphere(np.array(inliers), sphere_params)
                best_sphere = (optimized_params, inliers, inlier_indices)
                print(f"新しい最適解を発見: インライア数 = {inliers_count}")

        except Exception as e:
            print(f"イテレーション中のエラー: {str(e)}")
            continue

    if best_sphere is not None:
        print(f"最終的なインライア数: {len(best_sphere[1])}")

    return best_sphere


def create_sphere_mesh(center, radius, resolution=20):
    """球体のメッシュを生成"""
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius, resolution=resolution
    )
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color([0, 1, 0])  # 緑色
    mesh_sphere.compute_vertex_normals()
    return mesh_sphere


def visualize_results(original_pcd, sphere_params=None, inlier_indices=None):
    """改良された可視化関数"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 元の点群の色を設定
    original_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # グレー

    if sphere_params is not None and inlier_indices is not None:
        # インライアを赤色で表示
        colors = np.asarray(original_pcd.colors)
        colors[inlier_indices] = [1, 0, 0]  # 赤色
        original_pcd.colors = o3d.utility.Vector3dVector(colors)

        # 検出された球体をメッシュで表示
        center = sphere_params[:3]
        radius = sphere_params[3]
        mesh_sphere = create_sphere_mesh(center, radius)
        mesh_sphere.compute_vertex_normals()
        vis.add_geometry(mesh_sphere)

    # 点群の追加
    vis.add_geometry(original_pcd)

    # レンダリングオプションの設定
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.asarray([0, 0, 0])  # 黒背景
    opt.light_on = True

    # 座標軸の追加
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=max(sphere_params[3], 1.0) if sphere_params is not None else 1.0
    )
    vis.add_geometry(coord_frame)

    # カメラの位置設定
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])

    vis.run()
    vis.destroy_window()


def main():
    # コマンドライン引数でファイル名を指定できるようにする
    import sys

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "points.txt"  # デフォルトファイル名

    try:

        # 点群データの読み込み
        points, pcd = load_point_cloud(filename)

        # 球体検出
        result = detect_sphere_ransac(points)

        if result is not None:
            sphere_params, inliers, inlier_indices = result
            center = sphere_params[:3]
            radius = sphere_params[3]
            print(f"検出された球体:")
            print(f"中心座標: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
            print(f"半径: {radius:.3f}")
            print(f"インライア数: {len(inliers)}")

            # 結果の可視化
            visualize_results(pcd, sphere_params, inlier_indices)
        else:
            print("球体は検出されませんでした。")
            visualize_results(pcd)

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")


if __name__ == "__main__":
    main()
