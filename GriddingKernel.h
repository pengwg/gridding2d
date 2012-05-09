#ifndef GRIDDINGKERNEL_H
#define GRIDDINGKERNEL_H
#include <QVector>

class GriddingKernel
{
public:
    GriddingKernel(int kWidth, int overGridFactor, int length = 32);
    void GetKernelData();

private:
    int m_kWidth;
    int m_ogFactor;
    const int m_length;

    QVector<float> m_kernelData;
};

#endif // GRIDDINGKERNEL_H
