#ifndef GRIDGPU_H
#define GRIDGPU_H

#include "GridLut.h"
#include <cuda_runtime.h>

typedef struct __align__(16)
{
    TrajPoint trajPoint;
} TrajPointGpu;

typedef struct {
    TrajPointGpu*  trajPoints;
    size_t pitchTraj;
    int trajWidth;
} TrajBlocksGpu;

typedef struct __align__(8) {
    float real;
    float imag;
} complexGpu;

class GridGpu : public GridLut
{
public:
    GridGpu(int gridSize, ConvKernel &kernel);
    ~GridGpu();

    void gridding(QVector<TrajPoint> &trajPoints, complexVector &trajData, complexVector &gData);
    void gridding(complexVector &trajData);
    void gridding();

    cudaError_t prepareGPU(QVector<TrajPoint> &trajPoints);
    cudaError_t transferData(complexVector &trajData);
    cudaError_t retrieveData(complexVector &gData);
    complexGpu *getDevicePointer() { return m_d_gData; }

private:
    void createTrajBlocks(QVector<TrajPoint> &trajPoints);
    cudaError_t copyKernelData();
    cudaError_t copyTrajBlocks();
    cudaError_t mallocGpu();
    cudaError_t kernelCall();

    const int m_threadsPerBlock;
    const int m_gpuGridSize;
    QVector< QVector<TrajPointGpu> > m_trajBlocks;
    int m_kSize;

    TrajBlocksGpu m_d_trajBlocks;
    complexGpu *m_d_trajData;
    complexGpu *m_d_gData;
    int m_sharedSize;
};


#endif // GRIDGPU_H
