#include <QString>
#include <QDir>
#include <QFile>
#include <QDebug>
#include <QApplication>
#include <QLabel>
#include <QElapsedTimer>

#include <float.h>

#include "ConvKernel.h"
#include "GridLut.h"
#include "GridGpu.h"
#include "FFT2D.h"
#include "FFTGpu.h"

void loadData(QVector<TrajPoint> &TrajPoints, complexVector &TrajData, int TrajSize)
{
    QFile file("liver.trj");
    file.open(QIODevice::ReadOnly);

    QVector<float> buffer(TrajSize * 3);

    qint64 size = sizeof(float) * TrajSize * 3;
    auto count = file.read((char *)buffer.data(), size);
    Q_ASSERT(count == size);

    file.close();

    float *pdata = buffer.data();
    for (int i = 0; i < TrajSize; i++) {
        TrajPoints[i].kx = pdata[0];
        TrajPoints[i].ky = pdata[1];
        TrajPoints[i].dcf = pdata[2];
        TrajPoints[i].idx = i;
        pdata += 3;
    }

    file.setFileName("liver.0.data");
    file.open(QIODevice::ReadOnly);

    size = sizeof(float) * TrajSize * 2;
    count = file.read((char *)buffer.data(), size);
    Q_ASSERT(count == size);

    file.close();

    pdata = buffer.data();
    for (int i = 0; i < TrajSize; i++) {
        TrajData[i] = std::complex<float> (pdata[0], pdata[1]);
        pdata += 2;
    }
}



void displayData(int n0, int n1, const complexVector& data, const QString& title)
{
    QVector<float> dataValue;

    float max = 0;
    float min = FLT_MAX;

    for (auto cValue : data) {
        float value = std::abs(cValue);
        if (value > max) max = value;
        if (value < min) min = value;

        dataValue << value;
    }

    QImage dataImage(n1, n0, QImage::Format_Indexed8);
    for (int i = 0; i < 256; i++) {
        dataImage.setColor(i, qRgb(i, i, i));
    }

    int i = 0;
    for (int y = 0; y < n0; y++) {
        auto imageLine = dataImage.scanLine(y);

        for (int x = 0; x < n1; x++) {
            uint idx;
            if (max == min)
                idx = 127;
            else
                idx = (dataValue[i] - min) / (max - min) * 255;
            imageLine[x] = idx;
            i++;
        }
    }

    QPixmap pixmap = QPixmap::fromImage(dataImage);

    QLabel *imgWnd = new QLabel("Image Window");
    imgWnd->setWindowTitle(title);
    imgWnd->setPixmap(pixmap);
    imgWnd->show();
}



int main(int argc, char *argv[])
{
    int samples = 2250;
    int arms = 16;
    QDir::setCurrent("../k-export-liver/");

    QVector<TrajPoint> TrajPoints(samples * arms);
    complexVector TrajData(samples * arms);

    loadData(TrajPoints, TrajData, samples * arms);

    int kWidth = 4;
    int overGridFactor = 2;
    ConvKernel kernel(kWidth, overGridFactor, 256);

    int gridSize = 256 * overGridFactor;

    complexVector gDataCpu, gDataGpu;
    QElapsedTimer timer;

    GridGpu gridGpu(gridSize, kernel);
    gridGpu.prepareGPU(TrajPoints);

    int rep = 100;
    qWarning() << "\nIteration" << rep << 'x';

    // CPU gridding
    GridLut gridCpu(gridSize, kernel);
    timer.start();
    for (int i = 0; i < rep; i++)
        gridCpu.gridding(TrajPoints, TrajData, gDataCpu);
    qWarning() << "\nCPU gridding time =" << timer.elapsed() << "ms";

    // GPU gridding
    timer.restart();
    for (int i = 0; i < rep; i++)
        gridGpu.transferData(TrajData);

    cudaDeviceSynchronize();
    qWarning() << "\nGPU data transfer time =" << timer.elapsed() << "ms";

    timer.restart();
    for (int i = 0; i < rep; i++)
        gridGpu.gridding();

    cudaDeviceSynchronize();
    qWarning() << "\nGPU gridding time =" << timer.elapsed() << "ms";

    timer.restart();
    for (int i = 0; i < rep; i++)
        gridGpu.retrieveData(gDataGpu);
    qWarning() << "\nGPU data retrive time =" << timer.elapsed() << "ms";

    // CPU FFT
    FFT2D fft(gridSize, gridSize, false);
    timer.restart();
    for (int i = 0; i < rep; i++) {
        fft.fftShift(gDataCpu);
        fft.excute(gDataCpu);
        fft.fftShift(gDataCpu);
    }

    qWarning() << "\nCPU FFT time =" << timer.elapsed() << "ms";

    // GPU FFT
    FFTGpu fftGpu(gridSize, gridSize);
    timer.restart();
    for (int i = 0; i < rep; i++) {
        fftGpu.Execute((cufftComplex *)gridGpu.getDevicePointer());
    }
    cudaDeviceSynchronize();
    qWarning() << "\nGPU FFT time =" << timer.elapsed() << "ms";


    fft.fftShift(gDataGpu);
    fft.excute(gDataGpu);
    fft.fftShift(gDataGpu);

    QApplication app(argc, argv);
    displayData(gridSize, gridSize, gDataGpu, "image");

    return app.exec();
}
