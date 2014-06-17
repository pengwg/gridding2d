project(Gridding2D)

cmake_minimum_required(VERSION 2.8.9)

add_definitions( -std=c++11 )

find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(FFTW3 COMPONENTS single REQUIRED)

set(SRC_FILES
    Main.cpp
    ConvKernel.cpp
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


add_subdirectory(GPU)
add_executable(gridding ${SRC_FILES})
target_link_libraries(gridding gridGpu gsl blas ${FFTW3_LIBRARIES})
qt5_use_modules(gridding Core Widgets)
