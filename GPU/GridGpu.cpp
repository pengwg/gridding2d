#include <QDebug>

#include "GridGpu.h"


GridGpu::GridGpu(int gridSize, ConvKernel &kernel)
    : GridLut(gridSize, kernel), m_threadsPerBlock(256), m_gpuGridSize(16),
      m_d_trajData(nullptr), m_d_gData(nullptr), m_trajBlocks(m_gpuGridSize * m_gpuGridSize)
{
    m_d_trajBlocks.trajPoints = nullptr;
}

GridGpu::~GridGpu()
{
    if (m_d_trajBlocks.trajPoints)
        cudaFree(m_d_trajBlocks.trajPoints);

    if (m_d_trajData)
        cudaFree(m_d_trajData);

    if (m_d_gData)
        cudaFree(m_d_gData);
}

void GridGpu::gridding(QVector<TrajPoint> &trajPoints, complexVector &trajData, complexVector &gData)
{
    prepareGPU(trajPoints);
    gridding(trajData);
}


void GridGpu::gridding(complexVector &trajData)
{
    if (trajData.size() != m_kSize)
        qCritical() << "Size of k-space data not equal to the size of trajactory.";

    transferData(trajData);
    cudaError_t status = kernelCall();
    if (status != cudaSuccess)
        qWarning() << cudaGetErrorString(status);
}

void GridGpu::gridding()
{
    cudaError_t status = kernelCall();
    if (status != cudaSuccess)
        qWarning() << cudaGetErrorString(status);
}

void GridGpu::createTrajBlocks(QVector<TrajPoint> &trajPoints)
{
    float kBlockSize = ceilf((float)m_gridSize / m_gpuGridSize);

    m_trajBlocks.resize(m_gpuGridSize * m_gpuGridSize);
    float kHW = m_kernel.getKernelWidth() / 2;

    for (auto &traj : trajPoints) {
        int blockX = ((traj.kx + 0.5) * m_gridSize) / kBlockSize;
        Q_ASSERT(blockX < m_gpuGridSize);

        int blockY = (traj.ky + 0.5) * m_gridSize / kBlockSize;
        Q_ASSERT(blockY < m_gpuGridSize);

        TrajPointGpu trajGpu;
        trajGpu.trajPoint = traj;

        m_trajBlocks[blockY * m_gpuGridSize + blockX].append(trajGpu);

        int lbx = ((traj.kx + 0.5) * m_gridSize - kHW) / kBlockSize;
        int ubx = ((traj.kx + 0.5) * m_gridSize + kHW) / kBlockSize;
        int lby = ((traj.ky + 0.5) * m_gridSize - kHW) / kBlockSize;
        int uby = ((traj.ky + 0.5) * m_gridSize + kHW) / kBlockSize;

        if (lbx == blockX - 1 && lbx >= 0) {
            m_trajBlocks[blockY * m_gpuGridSize + lbx].append(trajGpu);
            if (lby == blockY - 1 && lby >= 0) {
                m_trajBlocks[lby * m_gpuGridSize + lbx].append(trajGpu);
                // qWarning() << trajGpu.kx << trajGpu.ky;
            }
            if (uby == blockY + 1 && uby < m_gpuGridSize)
                m_trajBlocks[uby * m_gpuGridSize + lbx].append(trajGpu);
        }

        if (ubx == blockX + 1 && ubx < m_gpuGridSize) {
            m_trajBlocks[blockY * m_gpuGridSize + ubx].append(trajGpu);
            if (lby == blockY - 1 && lby >= 0)
                m_trajBlocks[lby * m_gpuGridSize + ubx].append(trajGpu);
            if (uby == blockY + 1 && uby < m_gpuGridSize)
                m_trajBlocks[uby * m_gpuGridSize + ubx].append(trajGpu);
        }

        if (lby == blockY - 1 && lby >= 0)
            m_trajBlocks[lby * m_gpuGridSize + blockX].append(trajGpu);
        if (uby == blockY + 1 && uby < m_gpuGridSize)
            m_trajBlocks[uby * m_gpuGridSize + blockX].append(trajGpu);
    }

    m_kSize = trajPoints.size();
}


cudaError_t GridGpu::copyTrajBlocks()
{
    // Copy gridding k-trajectory data
    int maxP = 0;
    for (int i = 0; i < m_trajBlocks.size(); i++) {
        if (m_trajBlocks[i].size() > maxP) maxP = m_trajBlocks[i].size();
    }
    m_d_trajBlocks.trajWidth = maxP;

    cudaMallocPitch(&m_d_trajBlocks.trajPoints, &m_d_trajBlocks.pitchTraj, maxP * sizeof(TrajPointGpu), m_trajBlocks.size());
    cudaMemset(m_d_trajBlocks.trajPoints, 0, m_d_trajBlocks.pitchTraj * m_trajBlocks.size());
    qWarning() << "Max traj points per block:" << maxP;

    for (int i = 0; i < m_trajBlocks.size(); i++) {
        char *row = (char *)m_d_trajBlocks.trajPoints + i * m_d_trajBlocks.pitchTraj;
        cudaMemcpy(row, m_trajBlocks[i].data(), m_trajBlocks[i].size() * sizeof(TrajPointGpu), cudaMemcpyHostToDevice);
    }

    return cudaGetLastError();
}

cudaError_t GridGpu::mallocGpu()
{
    int gSize = m_gridSize * m_gridSize;

    // Malloc k-space and gridding matrix data
    cudaMalloc(&m_d_trajData, m_kSize * sizeof(complexGpu));
    cudaMalloc(&m_d_gData, gSize * sizeof(complexGpu));

    return cudaGetLastError();
}

cudaError_t GridGpu::prepareGPU(QVector<TrajPoint> &trajPoints)
{
    createTrajBlocks(trajPoints);
    copyKernelData();
    copyTrajBlocks();
    mallocGpu();
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    m_sharedSize = powf(ceilf((float)m_gridSize / m_gpuGridSize), 2) * sizeof(complexGpu);
    qWarning() << "Shared mem size:" << m_sharedSize;

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        qWarning() << cudaGetErrorString(status);
}

cudaError_t GridGpu::transferData(complexVector &trajData)
{
    cudaMemcpy(m_d_trajData, trajData.data(), trajData.size() * sizeof(complexGpu), cudaMemcpyHostToDevice);
    return cudaGetLastError();
}

cudaError_t GridGpu::retrieveData(complexVector &gData)
{
    gData.resize(m_gridSize * m_gridSize);
    cudaMemcpy(gData.data(), m_d_gData, gData.size() * sizeof(complexGpu), cudaMemcpyDeviceToHost);
    return cudaGetLastError();
}
