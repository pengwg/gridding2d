#include "FFTGpu.h"

FFTGpu::FFTGpu(int nx, int ny)
    : m_nx(nx), m_ny(ny)
{
    cufftPlan2d(&m_plan, nx, ny, CUFFT_C2C);
}

FFTGpu::~FFTGpu()
{
    cufftDestroy(m_plan);
}

cufftResult FFTGpu::Execute(cufftComplex *idata)
{
    m_idata = idata;
    return cufftExecC2C(m_plan, idata, idata, CUFFT_INVERSE);
}

cudaError_t FFTGpu::retrieveData(complexVector &gData)
{
    gData.resize(m_nx * m_ny);
    cudaMemcpy(gData.data(), m_idata, gData.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    return cudaGetLastError();
}
