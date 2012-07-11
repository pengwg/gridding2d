#ifndef GRIDLUT_H
#define GRIDLUT_H

#include <complex>

#include "ConvKernel.h"

typedef struct
{
    float kx;
    float ky;
    float dcf;
    int idx;
} TrajPoint;

typedef QVector< std::complex<float> > complexVector;

class GridLut
{
public:
    GridLut(int gridSize, ConvKernel &kernel);
    virtual ~GridLut();

    void gridding(QVector<TrajPoint> &trajPoints, complexVector &trajData, complexVector &gData);

protected:
    ConvKernel m_kernel;
    int m_gridSize;
};

#endif // GRIDLUT_H
