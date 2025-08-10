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
add_executable(mls mls.cpp)
target_link_libraries(mls ${PCL_LIBRARIES})