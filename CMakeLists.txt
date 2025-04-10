cmake_minimum_required(VERSION 3.10)
project(MosaicGenerator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(src)

add_executable(main
    src/main.cpp
    src/kmeans_mosaic.cpp
    src/statistical_features.cpp
)

target_link_libraries(main PRIVATE ${OpenCV_LIBS})
