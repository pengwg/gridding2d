#include <QDebug>

#include "GridLut.h"


GridLut::GridLut(int gridSize, ConvKernel &kernel)
    : Grid(gridSize, kernel)
{

}

void GridLut::gridding(QVector<kTraj> &trajData, complexVector &kData, complexVector &gData)
{
    gData.resize(m_gridSize * m_gridSize);

    float kHW = m_kernel.getKernelWidth() / 2;
    QVector<float> kernelData = m_kernel.getKernelData();
    int klength = kernelData.size();

    int iData = 0;

    for (auto &traj : trajData) {
        float xCenter = (0.5 + traj.kx) * m_gridSize; // kx in (-0.5, 0.5)
        int xStart = ceil(xCenter - kHW);
        int xEnd = floor(xCenter + kHW);

        float yCenter = (0.5 + traj.ky) * m_gridSize; // ky in (-0.5, 0.5)
        int yStart = ceil(yCenter  - kHW);
        int yEnd = floor(yCenter + kHW);

        xStart = fmax(xStart, 0);
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
                    gData[i] += kernelData[ki] * kData[iData] * traj.dcf;
                }
                i++;
            }
            i += di;
        }
        iData++;
    }
}
