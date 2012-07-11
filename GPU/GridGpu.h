#ifndef GRIDGPU_H
#define GRIDGPU_H

#include "GridLut.h"

typedef struct {
    Traj*  trajPoints;
    size_t pitchTraj;
    int trajWidth;
} TrajGpu;

typedef struct __align__(8) {
    float real;
    float imag;
} complexGpu;

class GridGpu : public GridLut
{
public:
    GridGpu(int gridSize, ConvKernel &kernel);
    ~GridGpu();

    void gridding(QVector<Traj> &trajPoints, complexVector &trajData, complexVector &gData);
    void gridding(complexVector &trajData);
    void gridding();

    cudaError_t prepareGPU(QVector<Traj> &trajPoints);
    cudaError_t transferData(complexVector &trajData);
    cudaError_t retrieveData(complexVector &gData);
    complexGpu *getDevicePointer() { return m_d_gData; }

private:
    void createTrajBlocks(QVector<Traj> &trajPoints);
    cudaError_t copyKernelData();
    cudaError_t copyTrajBlocks();
    cudaError_t mallocGpu();
    cudaError_t kernelCall();

    const int m_threadsPerBlock;
    const int m_gpuGridSize;
    QVector< QVector<Traj> > m_trajBlocks;
    int m_kSize;

    TrajGpu m_d_traj;
    complexGpu *m_d_trajData;
    complexGpu *m_d_gData;
    int m_sharedSize;
};


#endif // GRIDGPU_H
