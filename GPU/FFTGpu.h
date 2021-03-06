#ifndef FFTGPU_H
#define FFTGPU_H

#include <cuda_runtime.h>
#include <cufft.h>
#include "FFT2D.h"

class FFTGpu
{
public:
    FFTGpu(int nx, int ny);
    ~FFTGpu();
    cufftResult Execute(cufftComplex *idata);
    cudaError_t retrieveData(complexVector &gData);

private:
    cufftHandle m_plan;
    cufftComplex *m_idata;
    int m_nx;
    int m_ny;
};

#endif // FFTGPU_H
