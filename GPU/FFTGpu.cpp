#include "FFTGpu.h"

FFTGpu::FFTGpu(int nx, int ny)
{
    cufftPlan2d(&m_plan, nx, ny, CUFFT_C2C);
}

FFTGpu::~FFTGpu()
{
    cufftDestroy(m_plan);
}

cufftResult FFTGpu::Execute(cufftComplex *idata)
{
    return cufftExecC2C(m_plan, idata, idata, CUFFT_INVERSE);
}
