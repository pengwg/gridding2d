#ifndef GRIDGPU_H
#define GRIDGPU_H

#include "Grid.h"

typedef struct {
    kTraj*  trajData;
    size_t pitchTraj;
    int trajWidth;
} TrajGpu;

typedef struct __align__(8) {
    float real;
    float imag;
} complexGpu;

class GridGpu : public Grid
{
public:
    GridGpu(int gridSize, ConvKernel &kernel);
    ~GridGpu();

    void gridding(QVector<kTraj> &trajData, complexVector &kData, complexVector &gData);
    void gridding(complexVector &kData);
    cudaError_t prepareGPU(QVector<kTraj> &trajData);
    cudaError_t retrieveData(complexVector &gData);

private:
    void createTrajBlocks(QVector<kTraj> &trajData);
    cudaError_t copyKernelData();
    cudaError_t copyTrajBlocks();
    cudaError_t mallocGpu();
    cudaError_t kernelCall(complexVector &kData);

    const int m_threadsPerBlock;
    const int m_gpuGridSize;
    QVector< QVector<kTraj> > m_trajBlocks;
    int m_kSize;

    TrajGpu m_d_Traj;
    complexGpu *m_d_kData;
    complexGpu *m_d_gData;
    int m_sharedSize;
};


#endif // GRIDGPU_H
