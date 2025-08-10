#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <sys/stat.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/console/parse.h>

// 可視化のためのヘッダー（オプション）
#ifdef USE_VISUALIZATION
#include <pcl/visualization/pcl_visualizer.h>
#endif

int min_cluster_size = 30;
int max_cluster_size = 1000;
int num_neighbors = 15;
float smoothness_threshold = 10.0;
float curvature_threshold = 2.0; // 曲率の閾値
int k_search = 30;
int save_min_size = 30;
// コンパイルコマンド例
// mkdir build && cd build
// cmake ..
// make
//./region_growing ../processed_ply/iphone_60fps/output_smoothed.ply ../processed_ply/iphone_60fps/segment24/berry_

// ディレクトリを作成する関数
void createDirectory(const std::string &path)
{
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

// ハイパーパラメータをファイルに保存する関数
void saveParametersToFile(const std::string &output_prefix, const std::string &input_file)
{
    // 現在の日時を取得
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::stringstream filename;
    filename << output_prefix << "parameters.txt";

    std::ofstream param_file(filename.str());
    if (!param_file.is_open())
    {
        std::cerr << "Warning: Could not create parameter file: " << filename.str() << std::endl;
        return;
    }

    param_file << "=== Region Growing Segmentation Parameters ===" << std::endl;
    param_file << "Date: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << std::endl;
    param_file << "Input file: " << input_file << std::endl;
    param_file << std::endl;

    param_file << "--- Hyperparameters ---" << std::endl;
    param_file << "min_cluster_size = " << min_cluster_size << std::endl;
    param_file << "max_cluster_size = " << max_cluster_size << std::endl;
    param_file << "num_neighbors = " << num_neighbors << std::endl;
    param_file << "smoothness_threshold = " << smoothness_threshold << " (degrees)" << std::endl;
    param_file << "curvature_threshold = " << curvature_threshold << std::endl;
    param_file << "k_search = " << k_search << std::endl;
    param_file << "save_min_size = " << save_min_size << std::endl;
    param_file << std::endl;

    param_file << "--- Technical Details ---" << std::endl;
    param_file << "smoothness_threshold (radians) = " << (smoothness_threshold / 180.0 * M_PI) << std::endl;
    param_file << std::endl;

    param_file.close();
    std::cout << "Parameters saved to: " << filename.str() << std::endl;
}

// 結果をパラメータファイルに追記する関数
void appendResultsToFile(const std::string &output_prefix, size_t total_points,
                         size_t num_clusters, size_t classified_points)
{
    std::stringstream filename;
    filename << output_prefix << "parameters.txt";

    std::ofstream param_file(filename.str(), std::ios::app);
    if (!param_file.is_open())
    {
        std::cerr << "Warning: Could not append to parameter file: " << filename.str() << std::endl;
        return;
    }

    param_file << "--- Segmentation Results ---" << std::endl;
    param_file << "Total points: " << total_points << std::endl;
    param_file << "Total clusters: " << num_clusters << std::endl;
    param_file << "Classified points: " << classified_points << std::endl;
    param_file << "Unclassified points: " << (total_points - classified_points) << std::endl;
    param_file << "Classification rate: " << std::fixed << std::setprecision(2)
               << (100.0 * classified_points / total_points) << "%" << std::endl;
    param_file << std::endl;

    param_file << "--- Cluster Size Distribution ---" << std::endl;
    param_file.close();
}

// クラスターサイズ分布をパラメータファイルに追記する関数
void appendClusterInfo(const std::string &output_prefix,
                       const std::vector<pcl::PointIndices> &clusters)
{
    std::stringstream filename;
    filename << output_prefix << "parameters.txt";

    std::ofstream param_file(filename.str(), std::ios::app);
    if (!param_file.is_open())
    {
        return;
    }

    for (size_t i = 0; i < clusters.size(); i++)
    {
        param_file << "Cluster " << i << ": " << clusters[i].indices.size() << " points" << std::endl;
    }
    param_file << std::endl;

    param_file.close();
}

int main(int argc, char **argv)
{
    // コマンドライン引数チェック
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input.ply> [output_prefix]" << std::endl;
        return -1;
    }

    std::string input_file = argv[1];
    std::string output_prefix = "segmented_";
    if (argc >= 3)
    {
        output_prefix = argv[2];
    }

    // パラメータをファイルに保存
    saveParametersToFile(output_prefix, input_file);

    // clustersディレクトリを作成
    std::string clusters_dir = output_prefix + "clusters/";
    createDirectory(clusters_dir);
    std::cout << "Created clusters directory: " << clusters_dir << std::endl;

    // 点群データの読み込み
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(input_file, *cloud) == -1)
    {
        std::cerr << "Failed to load PLY file: " << input_file << std::endl;
        return -1;
    }
    std::cout << "Loaded " << cloud->size() << " points from " << input_file << std::endl;

    // 法線の推定
    std::cout << "Estimating normals..." << std::endl;
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud);
    normal_estimator.setKSearch(k_search); // 近傍点数
    normal_estimator.compute(*normals);
    std::cout << "Normal estimation completed." << std::endl;

    // Region Growing Segmentationの設定
    std::cout << "Starting region growing segmentation..." << std::endl;
    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(min_cluster_size); // 最小クラスターサイズ
    reg.setMaxClusterSize(max_cluster_size); // 最大クラスターサイズ
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(num_neighbors); // 近傍点数
    reg.setInputCloud(cloud);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(smoothness_threshold / 180.0 * M_PI); // 3度（ラジアン）
    reg.setCurvatureThreshold(curvature_threshold);                  // 曲率閾値

    // セグメンテーション実行
    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);

    std::cout << "Number of clusters: " << clusters.size() << std::endl;

    // 結果の可視化準備（各クラスターに色を割り当て）
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    colored_cloud->width = cloud->width;
    colored_cloud->height = cloud->height;
    colored_cloud->resize(cloud->size());

    // 初期化（全て黒）
    for (size_t i = 0; i < cloud->size(); i++)
    {
        colored_cloud->points[i].x = cloud->points[i].x;
        colored_cloud->points[i].y = cloud->points[i].y;
        colored_cloud->points[i].z = cloud->points[i].z;
        colored_cloud->points[i].r = 0;
        colored_cloud->points[i].g = 0;
        colored_cloud->points[i].b = 0;
    }

    // 各クラスターに色を割り当て
    for (size_t i = 0; i < clusters.size(); i++)
    {
        // ランダムな色を生成
        uint8_t r = rand() % 256;
        uint8_t g = rand() % 256;
        uint8_t b = rand() % 256;

        std::cout << "Cluster " << i << " has " << clusters[i].indices.size() << " points." << std::endl;

        // クラスター内の点に色を割り当て
        for (const auto &idx : clusters[i].indices)
        {
            colored_cloud->points[idx].r = r;
            colored_cloud->points[idx].g = g;
            colored_cloud->points[idx].b = b;
        }

        // 各クラスターを個別のPLYファイルに保存（オプション）
        if (clusters[i].indices.size() > 50)
        { // 100点以上のクラスターのみ保存
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto &idx : clusters[i].indices)
            {
                cluster_cloud->points.push_back(cloud->points[idx]);
            }
            cluster_cloud->width = cluster_cloud->points.size();
            cluster_cloud->height = 1;
            cluster_cloud->is_dense = true;

            std::stringstream ss;
            ss << output_prefix << "clusters/cluster_" << i << ".ply";
            pcl::io::savePLYFile(ss.str(), *cluster_cloud);
            std::cout << "Saved cluster " << i << " to " << ss.str() << std::endl;
        }
    }

    // 着色された点群を保存
    std::string colored_output = output_prefix + "colored.ply";
    pcl::io::savePLYFile(colored_output, *colored_cloud);
    std::cout << "Saved colored cloud to " << colored_output << std::endl;

    // 統計情報の出力
    std::cout << "\n=== Segmentation Statistics ===" << std::endl;
    std::cout << "Total points: " << cloud->size() << std::endl;
    std::cout << "Total clusters: " << clusters.size() << std::endl;

    // 未分類の点をカウント
    size_t classified_points = 0;
    for (const auto &cluster : clusters)
    {
        classified_points += cluster.indices.size();
    }
    std::cout << "Classified points: " << classified_points << std::endl;
    std::cout << "Unclassified points: " << (cloud->size() - classified_points) << std::endl;

    // 統計情報をファイルに保存
    appendResultsToFile(output_prefix, cloud->size(), clusters.size(), classified_points);
    appendClusterInfo(output_prefix, clusters);

#ifdef USE_VISUALIZATION
    // PCLVisualizerを使用した可視化（メインスレッドで実行）
    std::cout << "\nStarting visualization..." << std::endl;
    std::cout << "Press 'q' in the viewer window to exit." << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Region Growing Segmentation"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZRGB>(colored_cloud, "colored cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "colored cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }
#else
    std::cout << "\nVisualization disabled. To enable, compile with -DUSE_VISUALIZATION flag." << std::endl;
    std::cout << "View the results in the saved PLY files:" << std::endl;
    std::cout << "  - Colored cloud: " << colored_output << std::endl;
    std::cout << "  - Individual clusters: " << output_prefix << "cluster_*.ply" << std::endl;
#endif

    return 0;
}
