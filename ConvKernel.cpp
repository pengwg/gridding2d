#include <math.h>
#include <gsl/gsl_sf_bessel.h>
#include <QtDebug>

#include "ConvKernel.h"

ConvKernel::ConvKernel(float kWidth,  int overGridFactor, int length)
    : m_kWidth(kWidth), m_ogFactor(overGridFactor), m_length(length)
{

}

QVector<float> ConvKernel::GetKernelData()
{
    if (m_kernelData.isEmpty()) {
        m_kernelData.resize(m_length);

        float w = m_kWidth;
        float a = m_ogFactor;
        float beta = M_PI * sqrt(w * w / (a * a) * (a - 0.5) * (a - 0.5) - 0.8);

        float dk = w / 2.0 / (m_length -1);
        float kernel0;

        for (int i = 0; i < m_length; i++) {
            float k = dk * i;
            double x = beta * sqrt(1 - powf(2 * k / w, 2));

            m_kernelData[i] = gsl_sf_bessel_I0(x) / w;

            if (i == 0) kernel0 = m_kernelData[0];
            m_kernelData[i] /= kernel0;

            // qWarning() << "x =" << x << "k[" << i << "] =" << m_kernelData[i];
        }
    }

    return m_kernelData;
}
