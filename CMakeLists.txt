project(Gridding2D)

cmake_minimum_required(VERSION 2.6.0)

set(CMAKE_BUILD_TYPE Debug)

find_package(Qt4 COMPONENTS QtCore QtGui REQUIRED)
include(${QT_USE_FILE})

set(SRC_FILES
    GriddingMain.cpp
    ConvKernel.cpp
    GridLut.cpp
)

add_executable(gridding ${SRC_FILES})
target_link_libraries(gridding ${QT_LIBRARIES} gsl blas)
