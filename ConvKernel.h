#ifndef CONVKERNEL_H
#define CONVKERNEL_H

#include <QVector>

class ConvKernel
{
public:
    ConvKernel(float kWidth = 2, float overGridFactor = 2, int length = 32);
    QVector<float> GetKernelData();
    float GetKernelWidth();

private:
    float m_kWidth;
    float m_ogFactor;
    int m_length;

    QVector<float> m_kernelData;
};

#endif // CONVKERNEL_H
