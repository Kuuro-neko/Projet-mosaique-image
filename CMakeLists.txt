cmake_minimum_required(VERSION 3.5)
project( ImageMosaic )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/includes)

set(SRC ${CMAKE_SOURCE_DIR}/src)
set(INCLUDES ${CMAKE_SOURCE_DIR}/src/includes)

set(SOURCES
    ${SRC}/main.cpp
    ${INCLUDES}/alignmentMosaic.cpp
)

add_executable( main ${SOURCES} )
target_link_libraries( main ${OpenCV_LIBS} )