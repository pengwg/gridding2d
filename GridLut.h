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



class GridLut
{
public:
    GridLut();
    void SetConvKernel(ConvKernel &kernel);

private:
    ConvKernel m_kernel;
};

#endif // GRIDLUT_H
