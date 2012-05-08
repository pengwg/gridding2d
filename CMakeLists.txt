project(Gridding2D)

cmake_minimum_required(VERSION 2.6.0)

set(CMAKE_BUILD_TYPE Debug)

find_package(Qt4 COMPONENTS QtCore REQUIRED)
include(${QT_USE_FILE})

set(SRC_FILES
)

# include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(gridding2d ${SRC_FILES})
target_link_libraries(gridding2d ${QT_LIBRARIES})
