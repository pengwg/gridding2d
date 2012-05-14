#ifndef GRID_H
#define GRID_H

#include <complex>
#include <cuda_runtime.h>

#include "ConvKernel.h"

typedef struct __align__(16)
{
    float kx;
    float ky;
    std::complex<float> data;
    float dcf;
} kData;

typedef QVector< std::complex<float> > complexVector;



class Grid
{
public:
    Grid(int gridSize, ConvKernel &kernel);
    ~Grid();

    virtual void gridding(QVector<kData> &dataSet, complexVector &gDataSet) = 0;

protected:
    ConvKernel m_kernel;
    int m_gridSize;
};

#endif // GRID_H
