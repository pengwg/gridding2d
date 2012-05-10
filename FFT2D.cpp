#include "FFT2D.h"

FFT2D::FFT2D(int n0, int n1, bool forward)
    : m_n0(n0), m_n1(n1)
{
    m_in = (fftwf_complex *)fftwf_malloc(sizeof(fftw_complex) * n0 * n1);
    int sign = forward ? FFTW_FORWARD : FFTW_BACKWARD;

    m_plan = fftwf_plan_dft_2d(n0, n1, m_in, m_in, sign, FFTW_ESTIMATE | FFTW_DESTROY_INPUT);
}

FFT2D::~FFT2D()
{
    fftwf_destroy_plan(m_plan);
    fftwf_free(m_in);
}

void FFT2D::excute(complexVector & data)
{
    int i = 0;
    for (auto value : data) {
        m_in[i][0] = value.real();
        m_in[i][1] = value.imag();
        i++;
    }

    fftwf_execute(m_plan);

    i = 0;
    for (auto & value : data) {
        value = std::complex<float> (m_in[i][0], m_in[i][1]);
        i++;
    }
}

void FFT2D::fftShift(complexVector &data)
{
    int n0h = m_n0 / 2;
    int n1h = m_n1 / 2;

    int x1, y1;

    for (int y = 0; y < n0h; y++) {
        y1 = y + n0h;

        for (int x = 0; x < m_n1; x++) {
            if (x < n1h)
                x1 = x + n1h;
            else
                x1 = x - n1h;

            int i = y * m_n1 + x;
            int j = y1 * m_n1 + x1;

            auto tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
}
