#include <QVector>
#include <QDebug>

#include <cuda_runtime.h>

#include "Grid.h"
#include "GridGpu.h"


int threadsPerBlock;
int gpuGridSize;


TrajGpu devTraj;
complexGpu *devKData;
complexGpu *devGData;


__constant__ float Kernel[256];


__global__ void griddingKernel(TrajGpu devTraj, complexGpu *devKData, complexGpu *devGData, int gridSize)
{
    int blockWidth = ceilf((float)gridSize / gridDim.x);
    int blockHeight = ceilf((float)gridSize / gridDim.y);

    int blockStartX = blockWidth * blockIdx.x;
    int blockEndX = blockStartX + blockWidth;
    if (blockEndX > gridSize) blockEndX = gridSize;

    int blockStartY = blockHeight * blockIdx.y;
    int blockEndY = blockStartY + blockHeight;
    if (blockEndY > gridSize) blockEndY = gridSize;

    extern __shared__ complexGpu local_block[];

    int blockSize = blockWidth * blockHeight;
    for (int i = threadIdx.x; i < blockSize; i += blockDim.x) {
        local_block[i].real = 0;
        local_block[i].imag = 0;
    }
    __syncthreads();

    int kHW = 2;
    int klength = 256;

    int blockID = blockIdx.y * gridDim.x + blockIdx.x;
    kTraj *pTraj = (kTraj *)((char *)devTraj.trajData + devTraj.pitchTraj * blockID);

    for (int i = threadIdx.x; i < devTraj.trajWidth; i += blockDim.x) {
        float xCenter = (0.5f + pTraj[i].kx) * gridSize; // kx in (-0.5, 0.5)
        int xStart = ceilf(xCenter - kHW);
        int xEnd = floorf(xCenter + kHW);

        float yCenter = (0.5f + pTraj[i].ky) * gridSize; // ky in (-0.5, 0.5)
        int yStart = ceilf(yCenter - kHW);
        int yEnd = floorf(yCenter + kHW);

        if (xStart < blockStartX) xStart = blockStartX;
        if (xEnd > blockEndX - 1) xEnd = blockEndX - 1;

        if (yStart < blockStartY) yStart = blockStartY;
        if (yEnd > blockEndY - 1) yEnd = blockEndY - 1;

        int n = (yStart - blockStartY) * blockWidth + xStart - blockStartX;
        int dn = blockWidth - (xEnd - xStart) - 1;

        complexGpu data = devKData[pTraj[i].idx];

        for (int y = yStart; y <= yEnd; y++) {
            float dy = y - yCenter;

            for (int x = xStart; x <= xEnd; x++) {
                float dx = x - xCenter;
                float dk = sqrt(dy * dy + dx * dx);

                if (dk < kHW) {
                    int ki = rintf(dk / kHW * (klength - 1));
                    local_block[n].real += Kernel[ki] * pTraj[i].dcf * data.real;
                    local_block[n].imag += Kernel[ki] * pTraj[i].dcf * data.imag;
                }
                n++;
            }
            n += dn;
        }
    }

    __syncthreads();


    for (int i = threadIdx.x; i < blockSize; i += blockDim.x) {
        int x = i % blockWidth + blockStartX;
        int y = i / blockWidth + blockStartY;
        if (x < blockEndX && y < blockEndY) {
            int idx = y  * gridSize + x;
            devGData[idx].real = local_block[i].real;
            devGData[idx].imag = local_block[i].imag;
        }
    }

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
    cudaMemset(devTraj.trajData, 0, devTraj.pitchTraj * trajPartition.size());
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
    cudaMalloc(&devKData, kSize * sizeof(complexGpu));
    cudaMalloc(&devGData, gSize * sizeof(complexGpu));

    return cudaGetLastError();
}

cudaError_t griddingGpu(complexVector &kData, complexVector &gData, int gridSize)
{
    cudaMemcpy(devKData, kData.data(), kData.size() * sizeof(complexGpu), cudaMemcpyHostToDevice);

    int sharedSize = powf(ceilf((float)gridSize / gpuGridSize), 2) * sizeof(complexGpu);
    qWarning() << " Shared mem size:" << sharedSize;

    dim3 GridSize(gpuGridSize, gpuGridSize);
    griddingKernel<<<GridSize, threadsPerBlock, sharedSize>>>(devTraj, devKData, devGData, gridSize);

    cudaMemcpy(gData.data(), devGData, gData.size() * sizeof(complexGpu), cudaMemcpyDeviceToHost);
    std::complex<float> *p = gData.data();
    int a = p[0].real();

    return cudaGetLastError();
}
