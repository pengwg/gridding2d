#ifndef GRIDGPU_H
#define GRIDGPU_H

#include "Grid.h"


class GridGpu : public Grid
{
public:
    GridGpu(int gridSize, ConvKernel &kernel);

    void gridding(QVector<kData> &dataSet, complexVector &gDataSet);
    void gridding(complexVector &gDataSet);

private:
    void prepare(QVector<kData> &dataSet);

    const int m_threadsPerBlock;
    const int m_gpuGridSize;
};

#endif // GRIDGPU_H