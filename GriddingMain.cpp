#include <QString>
#include <QDir>
#include <QFile>
#include <QByteArray>
#include <QDebug>

#include "ConvKernel.h"
#include "GridLut.h"

void LoadData(QVector<kData> & kDataSet, int kSize)
{
    QFile file("liver.trj");
    file.open(QIODevice::ReadOnly);

    QVector<float> buffer(kSize * 3);
    qint64 size = sizeof(float) * kSize * 3;
    Q_ASSERT(file.read((char *)buffer.data(), size) == size);
    file.close();

    float *pdata = buffer.data();
    for (int i = 0; i < kSize; i++) {
        kDataSet[i].kx = pdata[0];
        kDataSet[i].ky = pdata[1];
        kDataSet[i].dcf = pdata[2];
        pdata += 3;
    }

    file.setFileName("liver.0.data");
    file.open(QIODevice::ReadOnly);
    size = sizeof(float) * kSize * 2;
    Q_ASSERT(file.read((char *)buffer.data(), size) == size);
    file.close();

    pdata = buffer.data();
    for (int i = 0; i < kSize; i++) {
        kDataSet[i].data = std::complex<float> (pdata[0], pdata[1]);
        pdata += 2;
    }
}



int main(int argc, char *argv[])
{
    int samples = 2250;
    int arms = 16;
    QDir::setCurrent("../k-export-liver/");

    QVector<kData> kDataSet(samples * arms);
    LoadData(kDataSet, samples * arms);

    int kWidth = 4;
    int overGridFactor = 2;

    ConvKernel kernel(kWidth, overGridFactor);
    GridLut grid;
    grid.SetConvKernel(kernel);

    return 0;
}
