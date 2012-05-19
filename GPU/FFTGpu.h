#ifndef FFTGPU_H
#define FFTGPU_H

#include <cufft.h>


typedef float2 Complex;


class FFTGpu
{
public:
    FFTGpu(int nx, int ny);
    ~FFTGpu();
    cufftResult Execute(cufftComplex *idata);

private:
    cufftHandle m_plan;
};

#endif // FFTGPU_H
