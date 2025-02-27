cmake_minimum_required(VERSION 2.8)
project( ImageMosaic )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SRC ${CMAKE_SOURCE_DIR}/src)

set(SOURCES
    ${SRC}/main.cpp
)

add_executable( main ${SOURCES} )
target_link_libraries( main ${OpenCV_LIBS} )