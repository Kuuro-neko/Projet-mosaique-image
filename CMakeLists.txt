# cmake_minimum_required(VERSION 3.10)
# project(MosaicGenerator)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(OpenCV REQUIRED)

# include_directories(src)

# add_executable(main
#     src/main.cpp
#     src/kmeans_mosaic.cpp
#     src/statistical_features.cpp
# )

# target_link_libraries(main PRIVATE ${OpenCV_LIBS})









# cmake_minimum_required(VERSION 3.5)
# project( ImageMosaic )
# find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} )

# set(SRC ${CMAKE_SOURCE_DIR}/src)

# set(SOURCES
#     ${SRC}/main.cpp
# )

# add_executable( main ${SOURCES} )
# target_link_libraries( main ${OpenCV_LIBS} )
cmake_minimum_required(VERSION 3.5)
project( ImageMosaic )

find_package( OpenCV REQUIRED )
find_package( FLTK REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/includes)
include_directories( ${OpenCV_INCLUDE_DIRS} ${FLTK_INCLUDE_DIR} )

set(SRC ${CMAKE_SOURCE_DIR}/src)
set(INCLUDES ${CMAKE_SOURCE_DIR}/src/includes)

set(SOURCES
    ${SRC}/main.cpp
    ${SRC}/kmeans_mosaic.cpp
    ${SRC}/statistical_features.cpp
    ${INCLUDES}/alignmentMosaic.cpp
    ${INCLUDES}/meanFeatureMosaic.cpp
)

add_executable( main ${SOURCES} )
target_link_libraries( main PRIVATE ${OpenCV_LIBS} ${FLTK_LIBRARIES} )
