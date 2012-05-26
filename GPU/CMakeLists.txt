
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

set(GRID_GPU_SRC_FILES
    GridGpu.cpp
    GridGpu.cu
    FFTGpu.cpp
)

set(CUDA_NVCC_FLAGS -gencode;arch=compute_20,code=sm_20;--ptxas-options=-v)
cuda_add_library(gridGpu ${GRID_GPU_SRC_FILES} SHARED)

target_link_libraries(gridGpu ${QT_LIBRARIES})
cuda_add_cufft_to_target(gridGpu)
