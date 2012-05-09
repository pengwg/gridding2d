#include <math.h>
#include <gsl/gsl_sf_bessel.h>
#include <QtDebug>
#include "GriddingKernel.h"

GriddingKernel::GriddingKernel(int kWidth,  int overGridFactor, int length)
    : m_kWidth(kWidth), m_ogFactor(overGridFactor), m_length(length),
      m_kernelData(length)
{

}

void GriddingKernel::GetKernelData()
{
    float w = m_kWidth;
    float a = m_ogFactor;

    float beta = M_PI * sqrt(w * w / (a * a) * (a - 0.5) * (a - 0.5) - 0.8);

}
