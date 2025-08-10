#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/surface/mls.h>
#include <pcl/common/common.h>
#include <pcl/console/parse.h>
#include <pcl/search/kdtree.h>
#include <cmath>

//./mls ../processed_ply/iphone_60fps/output-Cloud.ply  ../processed_ply/iphone_60fps/output_smoothed.ply -radius 0.03 -polynomial 2
int main(int argc, char **argv)
{
    // コマンドライン引数の確認
    if (argc < 3)
    {
        std::cout << "使用方法: " << argv[0] << " <入力PLYファイル> <出力PLYファイル> [オプション]" << std::endl;
        std::cout << "オプション:" << std::endl;
        std::cout << "  -radius <値>     : 検索半径 (デフォルト: 0.03)" << std::endl;
        std::cout << "  -polynomial <次数> : 多項式次数 (デフォルト: 2)" << std::endl;
        std::cout << "  -upsampling      : アップサンプリングを有効にする" << std::endl;
        std::cout << "  -normal          : 法線も出力する" << std::endl;
        return -1;
    }

    // パラメータの設定
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    double search_radius = 0.03;
    int polynomial_order = 2;
    bool compute_normals = false;
    bool use_upsampling = false;

    // コマンドライン引数の解析
    pcl::console::parse_argument(argc, argv, "-radius", search_radius);
    pcl::console::parse_argument(argc, argv, "-polynomial", polynomial_order);
    compute_normals = pcl::console::find_switch(argc, argv, "-normal");
    use_upsampling = pcl::console::find_switch(argc, argv, "-upsampling");

    // 点群データの定義
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);

    // PLYファイルの読み込み
    std::cout << "PLYファイルを読み込み中: " << input_file << std::endl;
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(input_file, *cloud) == -1)
    {
        PCL_ERROR("PLYファイルの読み込みに失敗しました。\n");
        return -1;
    }
    std::cout << "読み込み完了: " << cloud->size() << " 点" << std::endl;

    // Moving Least Squares (MLS) の設定
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;

    // 入力点群の設定
    mls.setInputCloud(cloud);

    // 検索方法の設定（KdTreeを使用）
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    mls.setSearchMethod(tree);

    // パラメータの設定
    mls.setSearchRadius(search_radius);       // 検索半径
    mls.setPolynomialOrder(polynomial_order); // 多項式の次数
    mls.setComputeNormals(compute_normals);   // 法線計算の有無

    // アップサンプリングの設定（オプション）
    if (use_upsampling)
    {
        mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::SAMPLE_LOCAL_PLANE);
        mls.setUpsamplingRadius(search_radius);
        mls.setUpsamplingStepSize(search_radius * 0.5);
        std::cout << "アップサンプリングを有効にしました" << std::endl;
    }

    // MLS処理の実行
    std::cout << "Moving Least Squares処理を実行中..." << std::endl;
    std::cout << "パラメータ:" << std::endl;
    std::cout << "  検索半径: " << search_radius << std::endl;
    std::cout << "  多項式次数: " << polynomial_order << std::endl;
    std::cout << "  法線計算: " << (compute_normals ? "有効" : "無効") << std::endl;

    mls.process(*cloud_with_normals);

    std::cout << "処理完了: " << cloud_with_normals->size() << " 点" << std::endl;

    // 結果の保存
    std::cout << "結果を保存中: " << output_file << std::endl;

    if (compute_normals)
    {
        // 法線付きで保存
        if (pcl::io::savePLYFile(output_file, *cloud_with_normals) == -1)
        {
            PCL_ERROR("PLYファイルの保存に失敗しました。\n");
            return -1;
        }
    }
    else
    {
        // 点のみで保存（法線なし）
        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*cloud_with_normals, *output_cloud);
        if (pcl::io::savePLYFile(output_file, *output_cloud) == -1)
        {
            PCL_ERROR("PLYファイルの保存に失敗しました。\n");
            return -1;
        }
    }

    std::cout << "保存完了!" << std::endl;

    // 統計情報の表示
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);

    // 対角線長を手動で計算
    double dx = max_pt.x - min_pt.x;
    double dy = max_pt.y - min_pt.y;
    double dz = max_pt.z - min_pt.z;
    double diagonal = std::sqrt(dx * dx + dy * dy + dz * dz);

    std::cout << "\n=== 処理結果 ===" << std::endl;
    std::cout << "入力点数: " << cloud->size() << std::endl;
    std::cout << "出力点数: " << cloud_with_normals->size() << std::endl;
    std::cout << "点群の対角線長: " << diagonal << std::endl;
    std::cout << "使用した検索半径: " << search_radius << " (" << (search_radius / diagonal * 100) << "% of diagonal)" << std::endl;

    return 0;
}

// 両方のプログラムをビルドできるCMakeLists.txt:
/*
cmake_minimum_required(VERSION 3.5)
project(pcl_processing)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Region Growing実行ファイル（既存）
add_executable(region_growing region_growing.cpp)
target_link_libraries(region_growing ${PCL_LIBRARIES})

# MLS Smoothing実行ファイル（新規）
add_executable(mls_smoothing mls_smoothing.cpp)
target_link_libraries(mls_smoothing ${PCL_LIBRARIES})
*/