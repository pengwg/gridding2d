project(Gridding2D)

cmake_minimum_required(VERSION 2.6.0)

add_definitions( -std=c++0x )

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

find_package(Qt4 COMPONENTS QtCore QtGui REQUIRED)
include(${QT_USE_FILE})

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(FFTW3 COMPONENTS single REQUIRED)

set(SRC_FILES
    Main.cpp
    ConvKernel.cpp
    Grid.cpp
    GridLut.cpp
    FFT2D.cpp
)

find_package(CUDA REQUIRED)
include_directories(
    ${FFTW3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/GPU
)

set(CUDA_NVCC_FLAGS -arch=compute_20)

add_subdirectory(GPU)
add_executable(gridding ${SRC_FILES})
target_link_libraries(gridding gridGpu ${QT_LIBRARIES} gsl blas ${FFTW3_LIBRARIES})
