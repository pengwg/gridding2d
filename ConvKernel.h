#ifndef GRIDDINGKERNEL_H
#define GRIDDINGKERNEL_H
#include <QVector>

class GriddingKernel
{
public:
    GriddingKernel(float kWidth, int overGridFactor, int length = 32);
    QVector<float> GetKernelData();

private:
    float m_kWidth;
    int m_ogFactor;
    const int m_length;

    QVector<float> m_kernelData;
};

#endif // GRIDDINGKERNEL_H
