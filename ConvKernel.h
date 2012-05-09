#ifndef CONVKERNEL_H
#define CONVKERNEL_H
#include <QVector>

class ConvKernel
{
public:
    ConvKernel(float kWidth, int overGridFactor, int length = 32);
    QVector<float> GetKernelData();

private:
    float m_kWidth;
    int m_ogFactor;
    const int m_length;

    QVector<float> m_kernelData;
};

#endif // CONVKERNEL_H
