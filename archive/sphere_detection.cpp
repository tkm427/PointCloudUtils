#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/pca.h>
#include <pcl/common/centroid.h>
#include <Eigen/Dense>
#include <thread>
#include <chrono>
#include <vector>
#include <fstream>
#include <cmath>

// 検出モデルのタイプ
enum class ModelType
{
    SPHERE,
    ELLIPSOID
};

// 検出された物体の構造体
struct DetectedObject
{
    // 中心座標
    float x, y, z;

    // 球体の場合の半径、楕円体の場合は最大半径
    float max_radius;

    // 楕円体の主軸の長さ (x, y, z方向の半径)
    float radius_x;
    float radius_y;
    float radius_z;

    // 楕円体の回転行列 (3x3)
    Eigen::Matrix3f rotation;

    // 物体に属する点群
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;

    // 評価指標
    double confidence;   // 信頼度スコア (0-1)
    double sphericity;   // 球形度 (0-1)、1に近いほど球形
    double compactness;  // コンパクト度 (0-1)、点の分布の均一性
    double isolation;    // 孤立度 (0-1)、他の物体との距離
    double volume;       // 体積
    double surface_area; // 表面積
    double density;      // 密度（点数/体積）

    // モデルタイプ
    ModelType model_type;

    // コンストラクタ
    DetectedObject() : x(0), y(0), z(0),
                       max_radius(0),
                       radius_x(0), radius_y(0), radius_z(0),
                       confidence(0), sphericity(0), compactness(0), isolation(0),
                       volume(0), surface_area(0), density(0),
                       model_type(ModelType::SPHERE)
    {
        cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
        rotation = Eigen::Matrix3f::Identity();
    }
};

// 楕円体の体積を計算
double calculateEllipsoidVolume(double a, double b, double c)
{
    return (4.0 / 3.0) * M_PI * a * b * c;
}

// 楕円体の表面積を計算（近似）
double calculateEllipsoidSurfaceArea(double a, double b, double c)
{
    // Knud Thomsen の近似式
    double p = 1.6075;
    double term = pow(a * b, p) + pow(a * c, p) + pow(b * c, p);
    return 4.0 * M_PI * pow(term / 3.0, 1.0 / p);
}

// 点群から主軸方向を計算し、楕円体パラメータを推定する
void estimateEllipsoidParameters(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, DetectedObject &object)
{
    // 重心を計算
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    object.x = centroid[0];
    object.y = centroid[1];
    object.z = centroid[2];

    // PCA (主成分分析) を適用
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud);

    // 主軸と固有値を取得
    Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
    Eigen::Vector3f eigen_values = pca.getEigenValues();

    // 固有値から楕円体の半径を計算（固有値の平方根がスケール）
    object.radius_x = std::sqrt(eigen_values[0]) * 2.5;
    object.radius_y = std::sqrt(eigen_values[1]) * 2.5;
    object.radius_z = std::sqrt(eigen_values[2]) * 2.5;

    // 最大半径
    object.max_radius = std::max({object.radius_x, object.radius_y, object.radius_z});

    // 回転行列を保存
    object.rotation = eigen_vectors;

    // 体積と表面積を計算
    object.volume = calculateEllipsoidVolume(object.radius_x, object.radius_y, object.radius_z);
    object.surface_area = calculateEllipsoidSurfaceArea(object.radius_x, object.radius_y, object.radius_z);

    // 密度（点の数/体積）
    object.density = cloud->size() / object.volume;

    // 球形度を計算 (三つの半径がどれだけ近いかを測定)
    double min_radius = std::min({object.radius_x, object.radius_y, object.radius_z});
    object.sphericity = min_radius / object.max_radius;
}

// 楕円体モデルから点までの距離を計算
double pointToEllipsoidDistance(const pcl::PointXYZ &point, const DetectedObject &ellipsoid)
{
    // 点を楕円体の座標系に変換
    Eigen::Vector3f p(point.x - ellipsoid.x, point.y - ellipsoid.y, point.z - ellipsoid.z);

    // 楕円体の主軸座標系に変換
    p = ellipsoid.rotation.transpose() * p;

    // 楕円体表面上の対応する点への方向ベクトル
    double rx2 = ellipsoid.radius_x * ellipsoid.radius_x;
    double ry2 = ellipsoid.radius_y * ellipsoid.radius_y;
    double rz2 = ellipsoid.radius_z * ellipsoid.radius_z;

    // 厳密な距離計算は複雑なので、近似解を使用
    // 点を正規化した座標で表現
    double x_norm = p[0] / ellipsoid.radius_x;
    double y_norm = p[1] / ellipsoid.radius_y;
    double z_norm = p[2] / ellipsoid.radius_z;
    double norm = std::sqrt(x_norm * x_norm + y_norm * y_norm + z_norm * z_norm);

    // 点が楕円体の内部にあるか外部にあるかを判定
    if (norm <= 1.0)
    {
        // 内部: 最短の半径方向までの距離を概算
        Eigen::Vector3f closest_surface_point;
        closest_surface_point[0] = p[0] / norm * ellipsoid.radius_x;
        closest_surface_point[1] = p[1] / norm * ellipsoid.radius_y;
        closest_surface_point[2] = p[2] / norm * ellipsoid.radius_z;

        // 点から表面までの距離
        double distance = (closest_surface_point - p).norm();
        return -distance; // 内部なので負の距離
    }
    else
    {
        // 外部: 同様の方法で距離を概算
        Eigen::Vector3f closest_surface_point;
        closest_surface_point[0] = p[0] / norm * ellipsoid.radius_x;
        closest_surface_point[1] = p[1] / norm * ellipsoid.radius_y;
        closest_surface_point[2] = p[2] / norm * ellipsoid.radius_z;

        // 点から表面までの距離
        double distance = (closest_surface_point - p).norm();
        return distance; // 外部なので正の距離
    }
}

// コンパクト度を計算 (点の分布の均一性)
double calculateCompactness(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const DetectedObject &object)
{
    double total_deviation = 0.0;
    double max_deviation = 0.0;

    // 各点から楕円体表面までの距離の偏差を計算
    for (const auto &point : *cloud)
    {
        double distance = std::abs(pointToEllipsoidDistance(point, object));
        total_deviation += distance;
        max_deviation = std::max(max_deviation, distance);
    }

    // 平均偏差
    double avg_deviation = total_deviation / cloud->size();

    // 平均偏差が小さいほどコンパクト度が高い (1に近い)
    return 1.0 - std::min(1.0, avg_deviation / (object.max_radius / 2.0));
}

// 検出された物体間の距離を計算
double calculateIsolation(const DetectedObject &obj, const std::vector<DetectedObject> &all_objects)
{
    if (all_objects.size() <= 1)
        return 1.0; // 他の物体がなければ完全に孤立

    double min_distance = std::numeric_limits<double>::max();
    Eigen::Vector3f center1(obj.x, obj.y, obj.z);

    for (const auto &other : all_objects)
    {
        // 自分自身はスキップ
        if (&obj == &other)
            continue;

        Eigen::Vector3f center2(other.x, other.y, other.z);
        double distance = (center1 - center2).norm() - obj.max_radius - other.max_radius;
        min_distance = std::min(min_distance, distance);
    }

    // 距離が大きいほど孤立度が高い (1に近い)
    return std::min(1.0, min_distance / (obj.max_radius * 5.0));
}

// 楕円体モデルを可視化関数
void visualizeEllipsoid(pcl::visualization::PCLVisualizer::Ptr viewer, const DetectedObject &ellipsoid,
                        const std::string &name, double r, double g, double b)
{
    // 楕円体の中心
    pcl::PointXYZ center(ellipsoid.x, ellipsoid.y, ellipsoid.z);

    // 主軸方向のベクトル
    Eigen::Vector3f axis1 = ellipsoid.rotation.col(0) * ellipsoid.radius_x;
    Eigen::Vector3f axis2 = ellipsoid.rotation.col(1) * ellipsoid.radius_y;
    Eigen::Vector3f axis3 = ellipsoid.rotation.col(2) * ellipsoid.radius_z;

    // 主軸を可視化
    pcl::PointXYZ p1(center.x + axis1[0], center.y + axis1[1], center.z + axis1[2]);
    pcl::PointXYZ p2(center.x + axis2[0], center.y + axis2[1], center.z + axis2[2]);
    pcl::PointXYZ p3(center.x + axis3[0], center.y + axis3[1], center.z + axis3[2]);

    viewer->addLine(center, p1, r, g, b, name + "_axis1");
    viewer->addLine(center, p2, r, g, b, name + "_axis2");
    viewer->addLine(center, p3, r, g, b, name + "_axis3");

    // 楕円体を近似する球体（視覚的な表現のため）
    viewer->addSphere(center, ellipsoid.max_radius * 0.3, r, g, b, name + "_sphere");

    // 楕円体のワイヤーフレーム表現を生成（点を使用）
    pcl::PointCloud<pcl::PointXYZ>::Ptr ellipsoid_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // 経度と緯度のステップ数
    int steps = 20;

    // 楕円体の表面上の点を生成
    for (int i = 0; i <= steps; ++i)
    {
        double phi = M_PI * i / steps; // 0から180度
        for (int j = 0; j <= steps; ++j)
        {
            double theta = 2 * M_PI * j / steps; // 0から360度

            // 通常の球面座標から点を生成
            Eigen::Vector3f point;
            point[0] = std::sin(phi) * std::cos(theta);
            point[1] = std::sin(phi) * std::sin(theta);
            point[2] = std::cos(phi);

            // 楕円体の形状に合わせてスケーリング
            point[0] *= ellipsoid.radius_x;
            point[1] *= ellipsoid.radius_y;
            point[2] *= ellipsoid.radius_z;

            // 回転を適用
            point = ellipsoid.rotation * point;

            // 中心を移動
            point[0] += ellipsoid.x;
            point[1] += ellipsoid.y;
            point[2] += ellipsoid.z;

            // 点を追加
            pcl::PointXYZ pcl_point;
            pcl_point.x = point[0];
            pcl_point.y = point[1];
            pcl_point.z = point[2];
            ellipsoid_cloud->push_back(pcl_point);
        }
    }

    // 点群を使用してワイヤーフレームを表示
    if (!ellipsoid_cloud->empty())
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler(ellipsoid_cloud, r * 255, g * 255, b * 255);
        viewer->addPointCloud<pcl::PointXYZ>(ellipsoid_cloud, handler, name + "_wireframe");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name + "_wireframe");
    }
}

int main(int argc, char **argv)
{
    // ファイル名をコマンドライン引数から取得
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " input.ply [options]" << std::endl;
        std::cerr << "Options:" << std::endl;
        std::cerr << "  --model [sphere|ellipsoid] : 検出モデル (default: ellipsoid)" << std::endl;
        std::cerr << "  --max_objects N         : 検出する最大物体数 (default: 20)" << std::endl;
        std::cerr << "  --min_radius R          : 最小半径 (default: 0.01)" << std::endl;
        std::cerr << "  --max_radius R          : 最大半径 (default: 0.5)" << std::endl;
        std::cerr << "  --distance_threshold D  : 点とモデルの許容距離 (default: 0.02)" << std::endl;
        std::cerr << "  --min_points N          : 検出に必要な最小点数 (default: 50)" << std::endl;
        std::cerr << "  --probability P         : 良いサンプルを見つける確率 (default: 0.95)" << std::endl;
        std::cerr << "  --max_iterations N      : 最大イテレーション数 (default: 1000)" << std::endl;
        std::cerr << "  --normal_distance_weight W: 法線距離の重み (default: 0.05)" << std::endl;
        std::cerr << "  --filter_outliers       : 外れ値フィルタリングを有効化" << std::endl;
        std::cerr << "  --outlier_mean_k N      : 外れ値検出の平均K値 (default: 50)" << std::endl;
        std::cerr << "  --outlier_stddev M      : 外れ値の標準偏差乗数 (default: 1.0)" << std::endl;
        std::cerr << "  --downsample            : ダウンサンプリングを有効化" << std::endl;
        std::cerr << "  --leaf_size S           : ボクセルグリッドのリーフサイズ (default: 0.005)" << std::endl;
        std::cerr << "  --normal_k N            : 法線推定のK近傍点数 (default: 30)" << std::endl;
        std::cerr << "  --cluster_tolerance D   : クラスタリングの距離閾値 (default: 0.02)" << std::endl;
        std::cerr << "  --cluster_min_size N    : クラスタの最小サイズ (default: 50)" << std::endl;
        std::cerr << "  --cluster_max_size N    : クラスタの最大サイズ (default: 10000)" << std::endl;
        std::cerr << "  --use_clustering        : クラスタリング法を使用" << std::endl;
        return -1;
    }

    std::string filename = argv[1];

    // デフォルトパラメータ
    ModelType model_type = ModelType::ELLIPSOID; // デフォルトで楕円体モデルを使用
    int max_objects = 20;                        // 検出する最大物体数
    float min_radius = 0.01f;                    // 最小半径
    float max_radius = 0.4f;                     // 最大半径
    float distance_threshold = 0.05f;            // 点とモデルの許容距離
    int min_points = 50;                         // 検出に必要な最小点数
    double probability = 0.95;                   // 良いサンプルを見つける確率
    int max_iterations = 1000;                   // 最大イテレーション数
    double normal_distance_weight = 0.05;        // 法線距離の重み
    bool filter_outliers = false;                // 外れ値フィルタリング
    int outlier_mean_k = 50;                     // 外れ値検出のK値
    double outlier_stddev = 1.0;                 // 外れ値の標準偏差乗数
    bool downsample = false;                     // ダウンサンプリング
    float leaf_size = 0.005f;                    // ボクセルサイズ
    int normal_k = 30;                           // 法線推定のK値
    float cluster_tolerance = 0.02f;             // クラスタリングの距離閾値
    int cluster_min_size = 50;                   // クラスタの最小サイズ
    int cluster_max_size = 10000;                // クラスタの最大サイズ
    bool use_clustering = false;                 // クラスタリング使用フラグ

    // コマンドライン引数の解析
    for (int i = 2; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--model" && i + 1 < argc)
        {
            std::string model_name = argv[++i];
            if (model_name == "sphere")
            {
                model_type = ModelType::SPHERE;
            }
            else if (model_name == "ellipsoid")
            {
                model_type = ModelType::ELLIPSOID;
            }
        }
        else if (arg == "--max_objects" && i + 1 < argc)
        {
            max_objects = std::stoi(argv[++i]);
        }
        else if (arg == "--min_radius" && i + 1 < argc)
        {
            min_radius = std::stof(argv[++i]);
        }
        else if (arg == "--max_radius" && i + 1 < argc)
        {
            max_radius = std::stof(argv[++i]);
        }
        else if (arg == "--distance_threshold" && i + 1 < argc)
        {
            distance_threshold = std::stof(argv[++i]);
        }
        else if (arg == "--min_points" && i + 1 < argc)
        {
            min_points = std::stoi(argv[++i]);
        }
        else if (arg == "--probability" && i + 1 < argc)
        {
            probability = std::stod(argv[++i]);
        }
        else if (arg == "--max_iterations" && i + 1 < argc)
        {
            max_iterations = std::stoi(argv[++i]);
        }
        else if (arg == "--normal_distance_weight" && i + 1 < argc)
        {
            normal_distance_weight = std::stod(argv[++i]);
        }
        else if (arg == "--filter_outliers")
        {
            filter_outliers = true;
        }
        else if (arg == "--outlier_mean_k" && i + 1 < argc)
        {
            outlier_mean_k = std::stoi(argv[++i]);
        }
        else if (arg == "--outlier_stddev" && i + 1 < argc)
        {
            outlier_stddev = std::stod(argv[++i]);
        }
        else if (arg == "--downsample")
        {
            downsample = true;
        }
        else if (arg == "--leaf_size" && i + 1 < argc)
        {
            leaf_size = std::stof(argv[++i]);
        }
        else if (arg == "--normal_k" && i + 1 < argc)
        {
            normal_k = std::stoi(argv[++i]);
        }
        else if (arg == "--cluster_tolerance" && i + 1 < argc)
        {
            cluster_tolerance = std::stof(argv[++i]);
        }
        else if (arg == "--cluster_min_size" && i + 1 < argc)
        {
            cluster_min_size = std::stoi(argv[++i]);
        }
        else if (arg == "--cluster_max_size" && i + 1 < argc)
        {
            cluster_max_size = std::stoi(argv[++i]);
        }
        else if (arg == "--use_clustering")
        {
            use_clustering = true;
        }
    }

    // 点群データの格納用
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

    // PLYファイルの読み込み
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(filename, *cloud) == -1)
    {
        PCL_ERROR("ファイル %s の読み込みに失敗しました。\n", filename.c_str());
        return -1;
    }

    std::cout << "Point cloud loaded: " << cloud->size() << " points." << std::endl;

    // 前処理: 外れ値の除去（オプション）
    if (filter_outliers)
    {
        std::cout << "外れ値除去フィルタを適用中..." << std::endl;
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(outlier_mean_k);
        sor.setStddevMulThresh(outlier_stddev);
        sor.filter(*cloud_filtered);
        std::cout << "  外れ値除去後の点数: " << cloud_filtered->size() << " 点" << std::endl;
    }
    else
    {
        *cloud_filtered = *cloud;
    }

    // 前処理: ダウンサンプリング（オプション）
    if (downsample)
    {
        std::cout << "ダウンサンプリングを適用中..." << std::endl;
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
        vg.setInputCloud(cloud_filtered);
        vg.setLeafSize(leaf_size, leaf_size, leaf_size);
        vg.filter(*cloud_downsampled);
        *cloud_filtered = *cloud_downsampled;
        std::cout << "  ダウンサンプリング後の点数: " << cloud_filtered->size() << " 点" << std::endl;
    }

    // 法線の計算
    std::cout << "法線を計算中..." << std::endl;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud_filtered);
    ne.setKSearch(normal_k);
    ne.compute(*normals);
    std::cout << "  法線計算完了: " << normals->size() << " 法線ベクトル" << std::endl;

    // 検出した物体を保存する配列
    std::vector<DetectedObject> detected_objects;

    // 可視化の準備
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // 元の点群を白で表示
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> white_color(cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, white_color, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

    // クラスタリング使用時の処理
    if (use_clustering)
    {
        std::cout << "ユークリッド距離クラスタリングを実行中..." << std::endl;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr cluster_tree(new pcl::search::KdTree<pcl::PointXYZ>);
        cluster_tree->setInputCloud(cloud_filtered);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(cluster_tolerance);
        ec.setMinClusterSize(cluster_min_size);
        ec.setMaxClusterSize(cluster_max_size);
        ec.setSearchMethod(cluster_tree);
        ec.setInputCloud(cloud_filtered);
        ec.extract(cluster_indices);

        std::cout << "  検出されたクラスタ数: " << cluster_indices.size() << std::endl;

        // 各クラスタに対して処理
        int object_idx = 0;
        for (const auto &indices : cluster_indices)
        {
            if (object_idx >= max_objects)
                break;

            // クラスタの点を抽出
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::Normal>::Ptr cluster_normals(new pcl::PointCloud<pcl::Normal>);

            for (const auto &idx : indices.indices)
            {
                cluster_cloud->push_back((*cloud_filtered)[idx]);
                cluster_normals->push_back((*normals)[idx]);
            }

            // 点数が少なすぎる場合はスキップ
            if (cluster_cloud->size() < min_points)
            {
                continue;
            }

            DetectedObject object;
            object.cloud = cluster_cloud;
            object.model_type = model_type;

            if (model_type == ModelType::SPHERE)
            {
                // 球体モデルの場合
                pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

                pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
                seg.setOptimizeCoefficients(true);
                seg.setModelType(pcl::SACMODEL_SPHERE);
                seg.setMethodType(pcl::SAC_RANSAC);
                seg.setDistanceThreshold(distance_threshold);
                seg.setMaxIterations(max_iterations);
                seg.setRadiusLimits(min_radius, max_radius);
                seg.setProbability(probability);
                seg.setNormalDistanceWeight(normal_distance_weight);
                seg.setInputCloud(cluster_cloud);
                seg.setInputNormals(cluster_normals);
                seg.segment(*inliers, *coefficients);

                // 検出成功の場合
                if (inliers->indices.size() >= min_points)
                {
                    object.x = coefficients->values[0];
                    object.y = coefficients->values[1];
                    object.z = coefficients->values[2];
                    object.radius_x = object.radius_y = object.radius_z = coefficients->values[3];
                    object.max_radius = coefficients->values[3];

                    // 回転行列はデフォルトで単位行列（球体なので方向はない）
                    object.rotation = Eigen::Matrix3f::Identity();

                    // 球の体積と表面積
                    object.volume = (4.0 / 3.0) * M_PI * std::pow(object.max_radius, 3);
                    object.surface_area = 4.0 * M_PI * std::pow(object.max_radius, 2);

                    // 完全な球の場合の球形度は1.0
                    object.sphericity = 1.0;

                    // 信頼度スコア（インライア率）を計算
                    object.confidence = static_cast<double>(inliers->indices.size()) / cluster_cloud->size();
                }
                else
                {
                    // 球体検出失敗の場合はクラスタ全体を使い、中心と平均半径を計算
                    Eigen::Vector4f centroid;
                    pcl::compute3DCentroid(*cluster_cloud, centroid);
                    object.x = centroid[0];
                    object.y = centroid[1];
                    object.z = centroid[2];

                    // 平均半径を計算
                    double avg_radius = 0.0;
                    for (const auto &point : *cluster_cloud)
                    {
                        double dx = point.x - object.x;
                        double dy = point.y - object.y;
                        double dz = point.z - object.z;
                        avg_radius += std::sqrt(dx * dx + dy * dy + dz * dz);
                    }
                    avg_radius /= cluster_cloud->size();

                    object.radius_x = object.radius_y = object.radius_z = object.max_radius = avg_radius;
                    object.confidence = 0.5; // 中程度の信頼度
                    object.sphericity = 0.8; // 仮定の値
                }
            }
            else
            {
                // 楕円体モデルの場合（PCAベースの推定）
                estimateEllipsoidParameters(cluster_cloud, object);

                // 楕円体モデルの適合度を計算
                double total_deviation = 0.0;
                int inside_points = 0;

                for (const auto &point : *cluster_cloud)
                {
                    double distance = pointToEllipsoidDistance(point, object);
                    total_deviation += std::abs(distance);
                    if (distance <= 0)
                        inside_points++; // 内部の点をカウント
                }

                double avg_deviation = total_deviation / cluster_cloud->size();
                object.confidence = 1.0 - std::min(1.0, avg_deviation / (object.max_radius / 2.0));

                // 点群の内部点の割合も考慮
                double inside_ratio = static_cast<double>(inside_points) / cluster_cloud->size();
                object.confidence = object.confidence * 0.7 + inside_ratio * 0.3;
            }

            // 密度を計算
            object.density = cluster_cloud->size() / object.volume;

            // コンパクト度を計算（点分布の均一性）
            object.compactness = calculateCompactness(cluster_cloud, object);

            // 検出したオブジェクトを配列に追加
            detected_objects.push_back(object);
            object_idx++;
        }

        // 孤立度の計算（他のオブジェクトとの相対的な距離）
        for (auto &object : detected_objects)
        {
            object.isolation = calculateIsolation(object, detected_objects);
        }
    }
    else
    {
        // 反復的な検出（クラスタリングを使わない場合）
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_remaining(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::Normal>::Ptr normals_remaining(new pcl::PointCloud<pcl::Normal>);
        *cloud_remaining = *cloud_filtered;
        *normals_remaining = *normals;

        for (int object_idx = 0; object_idx < max_objects; ++object_idx)
        {
            DetectedObject object;
            object.model_type = model_type;

            if (model_type == ModelType::SPHERE)
            {
                // 球体モデルの場合
                pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

                pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
                seg.setOptimizeCoefficients(true);
                seg.setModelType(pcl::SACMODEL_SPHERE);
                seg.setMethodType(pcl::SAC_RANSAC);
                seg.setDistanceThreshold(distance_threshold);
                seg.setMaxIterations(max_iterations);
                seg.setRadiusLimits(min_radius, max_radius);
                seg.setProbability(probability);
                seg.setNormalDistanceWeight(normal_distance_weight);
                seg.setInputCloud(cloud_remaining);
                seg.setInputNormals(normals_remaining);
                seg.segment(*inliers, *coefficients);

                // 検出失敗または点数が少なすぎる場合は終了
                if (inliers->indices.size() < min_points)
                {
                    std::cout << "これ以上の物体は検出できませんでした（点数不足）。" << std::endl;
                    break;
                }

                // 検出した物体の情報を保存
                object.x = coefficients->values[0];
                object.y = coefficients->values[1];
                object.z = coefficients->values[2];
                object.radius_x = object.radius_y = object.radius_z = coefficients->values[3];
                object.max_radius = coefficients->values[3];
                object.cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

                // 回転行列はデフォルトで単位行列（球体なので方向はない）
                object.rotation = Eigen::Matrix3f::Identity();

                // 球の体積と表面積
                object.volume = (4.0 / 3.0) * M_PI * std::pow(object.max_radius, 3);
                object.surface_area = 4.0 * M_PI * std::pow(object.max_radius, 2);

                // 完全な球の場合の球形度は1.0
                object.sphericity = 1.0;

                // 検出された物体の点を抽出
                pcl::ExtractIndices<pcl::PointXYZ> extract_points;
                extract_points.setInputCloud(cloud_remaining);
                extract_points.setIndices(inliers);
                extract_points.setNegative(false);
                extract_points.filter(*(object.cloud));

                // 対応する法線も抽出
                pcl::PointCloud<pcl::Normal>::Ptr object_normals(new pcl::PointCloud<pcl::Normal>);
                pcl::ExtractIndices<pcl::Normal> extract_normals;
                extract_normals.setInputCloud(normals_remaining);
                extract_normals.setIndices(inliers);
                extract_normals.setNegative(false);
                extract_normals.filter(*object_normals);

                // 信頼度スコア（インライア率）を計算
                object.confidence = static_cast<double>(inliers->indices.size()) / cloud_remaining->size();

                // 検出した物体の点を残りの点群から除去
                extract_points.setNegative(true);
                extract_points.filter(*cloud_remaining);

                // 対応する法線も除去
                extract_normals.setNegative(true);
                extract_normals.filter(*normals_remaining);
            }
            else
            {
                // 楕円体モデルの場合、まず球体で大まかに検出してから楕円体としてフィット
                pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

                pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
                seg.setOptimizeCoefficients(true);
                seg.setModelType(pcl::SACMODEL_SPHERE);
                seg.setMethodType(pcl::SAC_RANSAC);
                seg.setDistanceThreshold(distance_threshold * 1.5); // 緩めの閾値で検出
                seg.setMaxIterations(max_iterations);
                seg.setRadiusLimits(min_radius, max_radius);
                seg.setProbability(probability);
                seg.setNormalDistanceWeight(normal_distance_weight);
                seg.setInputCloud(cloud_remaining);
                seg.setInputNormals(normals_remaining);
                seg.segment(*inliers, *coefficients);

                // 検出失敗または点数が少なすぎる場合は終了
                if (inliers->indices.size() < min_points)
                {
                    std::cout << "これ以上の物体は検出できませんでした（点数不足）。" << std::endl;
                    break;
                }

                // 検出された物体の点を抽出
                object.cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::ExtractIndices<pcl::PointXYZ> extract_points;
                extract_points.setInputCloud(cloud_remaining);
                extract_points.setIndices(inliers);
                extract_points.setNegative(false);
                extract_points.filter(*(object.cloud));

                // 抽出した点群に対して楕円体パラメータを推定
                estimateEllipsoidParameters(object.cloud, object);

                // 楕円体モデルの適合度を計算
                double total_deviation = 0.0;
                int inside_points = 0;

                for (const auto &point : *(object.cloud))
                {
                    double distance = pointToEllipsoidDistance(point, object);
                    total_deviation += std::abs(distance);
                    if (distance <= 0)
                        inside_points++; // 内部の点をカウント
                }

                double avg_deviation = total_deviation / object.cloud->size();
                object.confidence = 1.0 - std::min(1.0, avg_deviation / (object.max_radius / 2.0));

                // 点群の内部点の割合も考慮
                double inside_ratio = static_cast<double>(inside_points) / object.cloud->size();
                object.confidence = object.confidence * 0.7 + inside_ratio * 0.3;

                // 対応する法線も抽出
                pcl::PointCloud<pcl::Normal>::Ptr object_normals(new pcl::PointCloud<pcl::Normal>);
                pcl::ExtractIndices<pcl::Normal> extract_normals;
                extract_normals.setInputCloud(normals_remaining);
                extract_normals.setIndices(inliers);
                extract_normals.setNegative(false);
                extract_normals.filter(*object_normals);

                // 検出した物体の点を残りの点群から除去
                extract_points.setNegative(true);
                extract_points.filter(*cloud_remaining);

                // 対応する法線も除去
                extract_normals.setNegative(true);
                extract_normals.filter(*normals_remaining);
            }

            // 密度を計算
            object.density = object.cloud->size() / object.volume;

            // コンパクト度を計算（点分布の均一性）
            object.compactness = calculateCompactness(object.cloud, object);

            // 検出した物体の情報を表示
            std::cout << "物体 #" << object_idx + 1 << " を検出しました:" << std::endl;
            std::cout << "  中心点: (" << object.x << ", " << object.y << ", " << object.z << ")" << std::endl;
            if (model_type == ModelType::SPHERE)
            {
                std::cout << "  半径: " << object.max_radius << std::endl;
            }
            else
            {
                std::cout << "  半径 (X,Y,Z): (" << object.radius_x << ", " << object.radius_y << ", "
                          << object.radius_z << ")" << std::endl;
            }
            std::cout << "  点数: " << object.cloud->size() << " 点" << std::endl;
            std::cout << "  信頼度: " << object.confidence << std::endl;
            std::cout << "  球形度: " << object.sphericity << std::endl;
            std::cout << "  コンパクト度: " << object.compactness << std::endl;
            std::cout << "  体積: " << object.volume << std::endl;
            std::cout << "  密度: " << object.density << std::endl;

            // 検出した物体を保存
            detected_objects.push_back(object);

            std::cout << "  残りの点数: " << cloud_remaining->size() << " 点" << std::endl;
        }

        // 孤立度の計算（他のオブジェクトとの相対的な距離）
        for (auto &object : detected_objects)
        {
            object.isolation = calculateIsolation(object, detected_objects);
        }
    }

    // 検出結果を表示
    std::cout << "合計 " << detected_objects.size() << " 個の物体を検出しました。" << std::endl;

    // 検出物体の可視化
    for (size_t i = 0; i < detected_objects.size(); ++i)
    {
        const auto &object = detected_objects[i];

        // オブジェクトごとに異なる色を用意
        int r = 50 + (200 * i) % 205;
        int g = 50 + (130 * i) % 205;
        int b = 50 + (90 * i) % 205;

        // 点群の表示
        std::string cloud_name = "object_cloud_" + std::to_string(i);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(object.cloud, r, g, b);
        viewer->addPointCloud<pcl::PointXYZ>(object.cloud, color_handler, cloud_name);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_name);

        // 物体モデルの可視化
        if (object.model_type == ModelType::SPHERE)
        {
            // 球体の表示
            std::string sphere_name = "object_sphere_" + std::to_string(i);
            viewer->addSphere(pcl::PointXYZ(object.x, object.y, object.z), object.max_radius,
                              r / 255.0, g / 255.0, b / 255.0, sphere_name);
        }
        else
        {
            // 楕円体の表示
            std::string ellipsoid_name = "object_ellipsoid_" + std::to_string(i);
            visualizeEllipsoid(viewer, object, ellipsoid_name, r / 255.0, g / 255.0, b / 255.0);
        }

        // 情報テキストの表示
        std::string text_name = "text_" + std::to_string(i);
        std::string model_type_str = (object.model_type == ModelType::SPHERE) ? "球" : "楕円体";
        std::string info_text = std::to_string(i + 1) + " (" + model_type_str + ")";
        viewer->addText3D(info_text, pcl::PointXYZ(object.x, object.y, object.z),
                          0.02, 1.0, 1.0, 1.0, text_name);
    }

    // 検出結果をCSVファイルに保存
    std::string output_filename = filename.substr(0, filename.find_last_of('.')) + "_objects.csv";
    std::ofstream outfile(output_filename);
    outfile << "ID,Model,X,Y,Z,Radius_X,Radius_Y,Radius_Z,Max_Radius,Volume,Surface_Area,Points,Confidence,Sphericity,Compactness,Isolation,Density" << std::endl;

    for (size_t i = 0; i < detected_objects.size(); ++i)
    {
        const auto &object = detected_objects[i];
        outfile << i + 1 << ","
                << (object.model_type == ModelType::SPHERE ? "Sphere" : "Ellipsoid") << ","
                << object.x << ","
                << object.y << ","
                << object.z << ","
                << object.radius_x << ","
                << object.radius_y << ","
                << object.radius_z << ","
                << object.max_radius << ","
                << object.volume << ","
                << object.surface_area << ","
                << object.cloud->size() << ","
                << object.confidence << ","
                << object.sphericity << ","
                << object.compactness << ","
                << object.isolation << ","
                << object.density << std::endl;
    }
    outfile.close();
    std::cout << "検出結果を " << output_filename << " に保存しました。" << std::endl;

    // 座標系と表示設定
    viewer->addCoordinateSystem(0.1);
    viewer->initCameraParameters();

    // ビューアを実行
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}