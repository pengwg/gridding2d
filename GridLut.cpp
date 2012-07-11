#include <QDebug>

#include "GridLut.h"


GridLut::GridLut(int gridSize, ConvKernel &kernel)
    : m_gridSize(gridSize), m_kernel(kernel)
{

}

GridLut::~GridLut()
{

}

void GridLut::gridding(QVector<TrajPoint> &trajPoints, complexVector &trajData, complexVector &gData)
{
    gData.resize(m_gridSize * m_gridSize);

    float kHW = m_kernel.getKernelWidth() / 2;
    QVector<float> kernelData = m_kernel.getKernelData();
    int klength = kernelData.size();

    int idx = 0;

    for (auto &traj : trajPoints) {
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

        auto data = traj.dcf * trajData[idx];

        int i = yStart * m_gridSize + xStart;
        int di = m_gridSize - (xEnd - xStart) - 1;

        for (int y = yStart; y <= yEnd; y++) {
            float dy = y - yCenter;
            float dy2 = dy * dy;

            for (int x = xStart; x <= xEnd; x++) {
                float dx = x - xCenter;
                float dk = sqrtf(dy2 + dx * dx);

                if (dk < kHW) {
                    int ki = round(dk / kHW * (klength - 1));
                    gData[i] += kernelData[ki] * data;
                }
                i++;
            }
            i += di;
        }
        idx++;
    }
}
