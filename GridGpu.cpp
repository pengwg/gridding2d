#include <QDebug>

#include "GridGpu.h"


cudaError_t copyKernel(const QVector<float> &kernelData);
cudaError_t copyTraj(const QVector< QVector<kTraj> > &trajPartition);
cudaError_t mallocGpu(int kSize, int gSize);
cudaError_t griddingGpu(complexVector &trajaSet, complexVector &gDataSet, int gridSize);

extern int threadsPerBlock;
extern int gpuGridSize;

extern TrajGpu devTraj;
extern complexGpu *devKData;
extern complexGpu *devGData;


GridGpu::GridGpu(int gridSize, ConvKernel &kernel)
    : Grid(gridSize, kernel), m_threadsPerBlock(256), m_gpuGridSize(16)
{
    threadsPerBlock = m_threadsPerBlock;
    gpuGridSize = m_gpuGridSize;
}

GridGpu::~GridGpu()
{
    cudaFree(devTraj.trajData);
    cudaFree(devKData);
    cudaFree(devGData);
}

void GridGpu::gridding(QVector<kTraj> &trajData, complexVector &kData, complexVector &gData)
{
    prepare(trajData);
    gridding(kData, gData);
}


void GridGpu::gridding(complexVector &kData, complexVector &gData)
{
    cudaError_t status = griddingGpu(kData, gData, m_gridSize);
    if (status != cudaSuccess)
        qWarning() << cudaGetErrorString(status);
}


void GridGpu::prepare(QVector<kTraj> &trajData)
{
    float kBlockSize = 1.0 / m_gpuGridSize;

    QVector< QVector<kTraj> > trajPartition(m_gpuGridSize * m_gpuGridSize);
    float kHW = m_kernel.getKernelWidth() / 2 / m_gridSize;

    for (auto &traj : trajData) {
        int blockX = (traj.kx + 0.5) / kBlockSize;
        Q_ASSERT(blockX < m_gpuGridSize);

        int blockY = (traj.ky + 0.5) / kBlockSize;
        Q_ASSERT(blockY < m_gpuGridSize);

        trajPartition[blockY * m_gpuGridSize + blockX].append(traj);

        int lbx = (traj.kx + 0.5 - kHW) / kBlockSize;
        int ubx = (traj.kx + 0.5 + kHW) / kBlockSize;
        int lby = (traj.ky + 0.5 - kHW) / kBlockSize;
        int uby = (traj.ky + 0.5 + kHW) / kBlockSize;

        if (lbx == blockX - 1 && lbx >= 0) {
            trajPartition[blockY * m_gpuGridSize + lbx].append(traj);
            if (lby == blockY - 1 && lby >= 0)
                trajPartition[lby * m_gpuGridSize + lbx].append(traj);
            if (uby == blockY + 1 && uby < m_gpuGridSize)
                trajPartition[uby * m_gpuGridSize + lbx].append(traj);
        }


        if (ubx == blockX + 1 && ubx < m_gpuGridSize) {
            trajPartition[blockY * m_gpuGridSize + ubx].append(traj);
            if (lby == blockY - 1 && lby >= 0)
                trajPartition[lby * m_gpuGridSize + ubx].append(traj);
            if (uby == blockY + 1 && uby < m_gpuGridSize)
                trajPartition[uby * m_gpuGridSize + ubx].append(traj);
        }

        if (lby == blockY - 1 && lby >= 0)
            trajPartition[lby * m_gpuGridSize + blockX].append(traj);
        if (uby == blockY + 1 && uby < m_gpuGridSize)
            trajPartition[uby * m_gpuGridSize + blockX].append(traj);
    }

    copyKernel(m_kernel.getKernelData());
    copyTraj(trajPartition);
    mallocGpu(trajData.size(), m_gridSize * m_gridSize);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        qWarning() << cudaGetErrorString(status);
}
