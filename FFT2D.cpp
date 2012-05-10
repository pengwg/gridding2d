#include "FFT2D.h"

FFT2D::FFT2D(int n0, int n1, bool forward)
{
    m_in = (fftwf_complex *)fftwf_malloc(sizeof(fftw_complex) * n0 * n1);
    int sign = forward ? FFTW_FORWARD : FFTW_BACKWARD;

    m_plan = fftwf_plan_dft_2d(n0, n1, m_in, m_in, sign, FFTW_PATIENT | FFTW_DESTROY_INPUT);
}

FFT2D::~FFT2D()
{
    fftwf_destroy_plan(m_plan);
    fftwf_free(m_in);
}

void FFT2D::Excute(complexVector &data)
{
    fftwf_execute(m_plan);
}
