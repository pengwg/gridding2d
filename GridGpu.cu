#include <QVector>
#include <cuda_runtime.h>

#include "Grid.h"

__constant__ float kernelGpu[256];

cudaError_t copyDataGpu(const QVector<float> &kernelData, const QVector< QVector<kData> > &dataPartition)
{
    Q_ASSERT(kernelData.size() == 256);
    cudaMemcpyToSymbol(kernelGpu, kernelData.data(), kernelData.size() * sizeof(float), cudaMemcpyHostToDevice);

    int maxP = 0;

    for (int i = 0; i < dataPartition.size(); i++) {
        if (dataPartition[i].size() > maxP) maxP = dataPartition.size();
        // qWarning() << "Partition size:" << dataP.size();
    }


    return cudaGetLastError();
}

cudaError_t griddingGpu()
{
    return cudaGetLastError();
}
