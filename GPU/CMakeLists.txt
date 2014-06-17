
set(GRID_GPU_SRC_FILES
    GridGpu.cpp
    GridGpu.cu
    FFTGpu.cpp
)

cuda_include_directories(gridGpu ${Qt5Core_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS -gencode;arch=compute_20,code=sm_20;--ptxas-options=-v)
cuda_add_library(gridGpu ${GRID_GPU_SRC_FILES} SHARED)
cuda_add_cufft_to_target(gridGpu)
qt5_use_modules(gridGpu Core)

