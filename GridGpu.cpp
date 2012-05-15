#include <QDebug>

#include "GridGpu.h"

GridGpu::GridGpu(int gridSize, ConvKernel &kernel)
    : Grid(gridSize, kernel), m_threadsPerBlock(256), m_gpuGridSize(8)
{
}


void GridGpu::gridding(QVector<kData> &kDataSet, complexVector &gDataSet)
{
    prepare(kDataSet);
    gridding(gDataSet);
}


void GridGpu::gridding(complexVector &gDataSet)
{

}


void GridGpu::prepare(QVector<kData> &dataSet)
{
    float kBlockSize = 1.0 / m_gpuGridSize;

    QVector< QVector<kData> > dataPartition(m_gpuGridSize * m_gpuGridSize);
    float kHW = m_kernel.getKernelWidth() / 2 / m_gridSize;

    for (auto &kdat : dataSet) {
        int blockX = (kdat.kx + 0.5) / kBlockSize;
        Q_ASSERT(blockX < m_gpuGridSize);

        int blockY = (kdat.ky + 0.5) / kBlockSize;
        Q_ASSERT(blockY < m_gpuGridSize);

        dataPartition[blockY * m_gpuGridSize + blockX].append(kdat);

        int lbx = (kdat.kx + 0.5 - kHW) / kBlockSize;
        int ubx = (kdat.kx + 0.5 + kHW) / kBlockSize;
        int lby = (kdat.ky + 0.5 - kHW) / kBlockSize;
        int uby = (kdat.ky + 0.5 + kHW) / kBlockSize;

        if (lbx == blockX - 1 && lbx >= 0) {
            dataPartition[blockY * m_gpuGridSize + lbx].append(kdat);
            if (lby == blockY - 1 && lby >= 0)
                dataPartition[lby * m_gpuGridSize + lbx].append(kdat);
            if (uby == blockY + 1 && uby < m_gpuGridSize)
                dataPartition[uby * m_gpuGridSize + lbx].append(kdat);
        }


        if (ubx == blockX + 1 && ubx < m_gpuGridSize) {
            dataPartition[blockY * m_gpuGridSize + ubx].append(kdat);
            if (lby == blockY - 1 && lby >= 0)
                dataPartition[lby * m_gpuGridSize + ubx].append(kdat);
            if (uby == blockY + 1 && uby < m_gpuGridSize)
                dataPartition[uby * m_gpuGridSize + ubx].append(kdat);
        }

        if (lby == blockY - 1 && lby >= 0)
            dataPartition[lby * m_gpuGridSize + blockX].append(kdat);
        if (uby == blockY + 1 && uby < m_gpuGridSize)
            dataPartition[uby * m_gpuGridSize + blockX].append(kdat);
    }

    copyDataGpu(m_kernel.getKernelData(), dataPartition);
}
