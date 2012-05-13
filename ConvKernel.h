#ifndef CONVKERNEL_H
#define CONVKERNEL_H

#include <QVector>

class ConvKernel
{
public:
    ConvKernel(float kWidth, float overGridFactor, int length = 32);
    QVector<float> getKernelData();
    float getKernelWidth();

private:
    float m_kWidth;
    float m_ogFactor;

    QVector<float> m_kernelData;
};

#endif // CONVKERNEL_H
