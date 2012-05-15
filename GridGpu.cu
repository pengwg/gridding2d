#include <QVector>
#include <QDebug>

#include <cuda_runtime.h>

#include "Grid.h"
#include "GridGpu.h"


int threadsPerBlock;
int gpuGridSize;


TrajGpu devTraj;
float *devKData;
float *devGData;


__constant__ float Kernel[256];


__global__ void griddingKernel(float *devKDataSet, float *devGDataSet, int gridSize)
{
    devGDataSet[0] = 1;
}

cudaError_t copyKernel(const QVector<float> &kernelData)
{
    Q_ASSERT(kernelData.size() == 256);
    // Copy gridding kernel data
    cudaMemcpyToSymbol(Kernel, kernelData.data(), kernelData.size() * sizeof(float));

    return cudaGetLastError();
}


cudaError_t copyTraj(const QVector< QVector<kTraj> > &trajPartition)
{
    // Copy gridding k-trajectory data
    int maxP = 0;
    for (int i = 0; i < trajPartition.size(); i++) {
        if (trajPartition[i].size() > maxP) maxP = trajPartition[i].size();
        // qWarning() << "Partition" << i << trajPartition[i].size();
    }
    devTraj.trajWidth = maxP;

    cudaMallocPitch(&devTraj.trajData, &devTraj.pitchTraj, maxP * sizeof(kTraj), trajPartition.size());
    qWarning() << "Partition pitch:" << devTraj.pitchTraj;

    for (int i = 0; i < trajPartition.size(); i++) {
        char *row = (char *)devTraj.trajData + i * devTraj.pitchTraj;
        cudaMemcpy(row, trajPartition[i].data(), trajPartition[i].size() * sizeof(kTraj), cudaMemcpyHostToDevice);
    }

    return cudaGetLastError();
}

cudaError_t mallocGpu(int kSize, int gSize)
{
    // Malloc k-space and gridding matrix data
    cudaMalloc(&devKData, kSize * sizeof(float));
    cudaMalloc(&devGData, gSize * sizeof(float));

    return cudaGetLastError();
}

cudaError_t griddingGpu(complexVector &kData, complexVector &gData, int gridSize)
{
    qWarning() << "In gridding GPU";
    cudaMemcpy(devKData, kData.data(), kData.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 GridSize(gpuGridSize, gpuGridSize);
    griddingKernel<<<GridSize, threadsPerBlock>>>(devKData, devGData, gridSize);

    cudaMemcpy(gData.data(), devGData, gData.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return cudaGetLastError();
}
