find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

set(GRID_GPU_SRC_FILES
    GridGpu.cpp
    GridGpu.cu
)

cuda_add_library(gridGpu ${GRID_GPU_SRC_FILES} SHARED)
target_link_libraries(gridGpu ${QT_LIBRARIES})
