project(Gridding2D)

cmake_minimum_required(VERSION 2.6.0)

set(CMAKE_BUILD_TYPE Debug)

find_package(Qt4 COMPONENTS QtCore QtGui REQUIRED)
include(${QT_USE_FILE})

# set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
# find_package(GSL REQUIRED)

set(SRC_FILES
    GriddingMain.cpp
    ConvKernel.cpp
    GridLut.cpp
)

# include_directories(${GSL_INCLUDE_DIRS})
# link_directories(${GSL_INCLUDE_DIRS})

add_executable(gridding ${SRC_FILES})
target_link_libraries(gridding ${QT_LIBRARIES} gsl blas)
