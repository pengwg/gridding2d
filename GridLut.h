#ifndef GRIDLUT_H
#define GRIDLUT_H

#include <complex>
#include <cuda_runtime.h>

#include "ConvKernel.h"

typedef struct __align__(16)
{
    float kx;
    float ky;
    float dcf;
    int idx;
} Traj;

typedef QVector< std::complex<float> > complexVector;

class GridLut
{
public:
    GridLut(int gridSize, ConvKernel &kernel);
    virtual ~GridLut();

    virtual void gridding(QVector<Traj> &trajPoints, complexVector &trajData, complexVector &gData);

protected:
    ConvKernel m_kernel;
    int m_gridSize;
};

#endif // GRIDLUT_H
