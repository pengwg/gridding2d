#ifndef GRIDGPU_H
#define GRIDGPU_H

#include "Grid.h"

typedef struct {
    kTraj*  trajData;
    size_t pitchTraj;
    int trajWidth;
} TrajGpu;


class GridGpu : public Grid
{
public:
    GridGpu(int gridSize, ConvKernel &kernel);
    ~GridGpu();

    void gridding(QVector<kTraj> &trajData, complexVector &kData, complexVector &gData);
    void gridding(complexVector &kData, complexVector &gData);
    void prepare(QVector<kTraj> &trajData);

private:
    const int m_threadsPerBlock;
    const int m_gpuGridSize;
};


#endif // GRIDGPU_H
