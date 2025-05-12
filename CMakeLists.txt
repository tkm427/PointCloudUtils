cmake_minimum_required(VERSION 3.5)
project(sphere_detection)

# C++11を有効化
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# PCLを見つける
find_package(PCL 1.8 REQUIRED)

# インクルードディレクトリを追加
include_directories(${PCL_INCLUDE_DIRS})

# リンクディレクトリを追加
link_directories(${PCL_LIBRARY_DIRS})

# 定義を追加
add_definitions(${PCL_DEFINITIONS})

# 実行ファイルの作成
add_executable(sphere_detection sphere_detection.cpp)

# PCLライブラリとリンク
target_link_libraries(sphere_detection ${PCL_LIBRARIES})