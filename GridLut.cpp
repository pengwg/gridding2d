#include "GridLut.h"

GridLut::GridLut(int gridSize)
{
    m_gridSize = gridSize;
}

void GridLut::SetConvKernel(ConvKernel &kernel)
{
    m_kernel = kernel;
}

void GridLut::Gridding(QVector<kData> &kDataSet, complexVector &gDataSet)
{
    float kHW = m_kernel.GetKernelWidth() / 2;
    QVector<float> kernelData = m_kernel.GetKernelData();
    int klength = kernelData.size();

    for (auto kdat : kDataSet) {
        float xCenter = (0.5 + kdat.kx) * m_gridSize; // kx in (-0.5, 0.5)
        int xStart = ceil(xCenter - kHW);
        int xEnd = floor(xCenter + kHW);

        float yCenter = (0.5 + kdat.ky) * m_gridSize; // ky in (-0.5, 0.5)
        int yStart = ceil(yCenter  - kHW);
        int yEnd = floor(yCenter + kHW);

        yStart = fmax(yStart, 0);
        xEnd = fmin(xEnd, m_gridSize - 1);

        yStart = fmax(yStart, 0);
        yEnd = fmin(yEnd, m_gridSize - 1);


        int i = yStart * m_gridSize + xStart;
        int di = m_gridSize - (xEnd - xStart) - 1;

        for (int y = yStart; y <= yEnd; y++) {
            float dy = y - yCenter;

            for (int x = xStart; x <= xEnd; x++) {
                float dx = x - xCenter;
                float dk = sqrt(dy * dy + dx * dx);

                if (dk < kHW) {
                    int ki = round(dk / kHW * (klength - 1));
                    gDataSet[i] += kernelData[ki] * kdat.data * kdat.dcf;
                }
                i++;
            }
            i += di;
        }

    }
}
