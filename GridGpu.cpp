#include <QDebug>

#include "GridGpu.h"


GridGpu::GridGpu(int gridSize, ConvKernel &kernel)
    : Grid(gridSize, kernel), m_threadsPerBlock(256), m_gpuGridSize(16),
      m_d_kData(nullptr), m_d_gData(nullptr), m_trajBlocks(m_gpuGridSize * m_gpuGridSize)
{
    m_d_Traj.trajData = nullptr;
}

GridGpu::~GridGpu()
{
    if (m_d_Traj.trajData)
        cudaFree(m_d_Traj.trajData);

    if (m_d_kData)
        cudaFree(m_d_kData);

    if (m_d_gData)
        cudaFree(m_d_gData);
}

void GridGpu::gridding(QVector<kTraj> &trajData, complexVector &kData, complexVector &gData)
{
    prepareGPU(trajData);
    gridding(kData);
}


void GridGpu::gridding(complexVector &kData)
{
    if (kData.size() != m_kSize)
        qCritical() << "Size of k-space data not equal to the size of trajactory.";

    cudaError_t status = kernelCall(kData);
    if (status != cudaSuccess)
        qWarning() << cudaGetErrorString(status);
}


void GridGpu::createTrajBlocks(QVector<kTraj> &trajData)
{
    float kBlockSize = ceilf((float)m_gridSize / m_gpuGridSize);

    m_trajBlocks.resize(m_gpuGridSize * m_gpuGridSize);
    float kHW = m_kernel.getKernelWidth() / 2;

    for (auto &traj : trajData) {
        int blockX = ((traj.kx + 0.5) * m_gridSize) / kBlockSize;
        Q_ASSERT(blockX < m_gpuGridSize);

        int blockY = (traj.ky + 0.5) * m_gridSize / kBlockSize;
        Q_ASSERT(blockY < m_gpuGridSize);

        m_trajBlocks[blockY * m_gpuGridSize + blockX].append(traj);

        int lbx = ((traj.kx + 0.5) * m_gridSize - kHW) / kBlockSize;
        int ubx = ((traj.kx + 0.5) * m_gridSize + kHW) / kBlockSize;
        int lby = ((traj.ky + 0.5) * m_gridSize - kHW) / kBlockSize;
        int uby = ((traj.ky + 0.5) * m_gridSize + kHW) / kBlockSize;

        if (lbx == blockX - 1 && lbx >= 0) {
            m_trajBlocks[blockY * m_gpuGridSize + lbx].append(traj);
            if (lby == blockY - 1 && lby >= 0) {
                m_trajBlocks[lby * m_gpuGridSize + lbx].append(traj);
                // qWarning() << traj.kx << traj.ky;
            }
            if (uby == blockY + 1 && uby < m_gpuGridSize)
                m_trajBlocks[uby * m_gpuGridSize + lbx].append(traj);
        }

        if (ubx == blockX + 1 && ubx < m_gpuGridSize) {
            m_trajBlocks[blockY * m_gpuGridSize + ubx].append(traj);
            if (lby == blockY - 1 && lby >= 0)
                m_trajBlocks[lby * m_gpuGridSize + ubx].append(traj);
            if (uby == blockY + 1 && uby < m_gpuGridSize)
                m_trajBlocks[uby * m_gpuGridSize + ubx].append(traj);
        }

        if (lby == blockY - 1 && lby >= 0)
            m_trajBlocks[lby * m_gpuGridSize + blockX].append(traj);
        if (uby == blockY + 1 && uby < m_gpuGridSize)
            m_trajBlocks[uby * m_gpuGridSize + blockX].append(traj);
    }

    m_kSize = trajData.size();
}


cudaError_t GridGpu::copyTrajBlocks()
{
    // Copy gridding k-trajectory data
    int maxP = 0;
    for (int i = 0; i < m_trajBlocks.size(); i++) {
        if (m_trajBlocks[i].size() > maxP) maxP = m_trajBlocks[i].size();
    }
    m_d_Traj.trajWidth = maxP;

    cudaMallocPitch(&m_d_Traj.trajData, &m_d_Traj.pitchTraj, maxP * sizeof(kTraj), m_trajBlocks.size());
    cudaMemset(m_d_Traj.trajData, 0, m_d_Traj.pitchTraj * m_trajBlocks.size());
    qWarning() << "Partition pitch:" << m_d_Traj.pitchTraj;

    for (int i = 0; i < m_trajBlocks.size(); i++) {
        char *row = (char *)m_d_Traj.trajData + i * m_d_Traj.pitchTraj;
        cudaMemcpy(row, m_trajBlocks[i].data(), m_trajBlocks[i].size() * sizeof(kTraj), cudaMemcpyHostToDevice);
    }

    return cudaGetLastError();
}

cudaError_t GridGpu::mallocGpu()
{
    int gSize = m_gridSize * m_gridSize;

    // Malloc k-space and gridding matrix data
    cudaMalloc(&m_d_kData, m_kSize * sizeof(complexGpu));
    cudaMalloc(&m_d_gData, gSize * sizeof(complexGpu));

    return cudaGetLastError();
}

cudaError_t GridGpu::prepareGPU(QVector<kTraj> &trajData)
{
    createTrajBlocks(trajData);
    copyKernelData();
    copyTrajBlocks();
    mallocGpu();

    m_sharedSize = powf(ceilf((float)m_gridSize / m_gpuGridSize), 2) * sizeof(complexGpu);
    qWarning() << "Shared mem size:" << m_sharedSize;

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        qWarning() << cudaGetErrorString(status);
}

cudaError_t GridGpu::retrieveData(complexVector &gData)
{
    gData.resize(m_gridSize * m_gridSize);
    cudaMemcpy(gData.data(), m_d_gData, gData.size() * sizeof(complexGpu), cudaMemcpyDeviceToHost);
    return cudaGetLastError();
}
