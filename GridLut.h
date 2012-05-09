#ifndef GRIDLUT_H
#define GRIDLUT_H

#include <complex>

#include "ConvKernel.h"

typedef struct
{
    float kx;
    float ky;
    std::complex<float> data;
    float dcf;
} kData;

typedef QVector< std::complex<float> > complexVector;



class GridLut
{
public:
    GridLut(int gridSize);
    void SetConvKernel(ConvKernel &kernel);
    void Gridding(QVector<kData> &dataSet, complexVector &gDataSet);

private:
    ConvKernel m_kernel;
    int m_gridSize;
};

#endif // GRIDLUT_H
