#include <QString>
#include <QDir>
#include <QFile>
#include <QDebug>
#include <QtGui/QApplication>
#include <QLabel>
#include <QElapsedTimer>

#include <float.h>

#include "ConvKernel.h"
#include "GridLut.h"
#include "GridGpu.h"
#include "FFT2D.h"

void loadData(QVector<kTraj> &trajData, complexVector &kData, int kSize)
{
    QFile file("liver.trj");
    file.open(QIODevice::ReadOnly);

    QVector<float> buffer(kSize * 3);

    qint64 size = sizeof(float) * kSize * 3;
    auto count = file.read((char *)buffer.data(), size);
    Q_ASSERT(count == size);

    file.close();

    float *pdata = buffer.data();
    for (int i = 0; i < kSize; i++) {
        trajData[i].kx = pdata[0];
        trajData[i].ky = pdata[1];
        trajData[i].dcf = pdata[2];
        trajData[i].idx = i;
        pdata += 3;
    }

    file.setFileName("liver.0.data");
    file.open(QIODevice::ReadOnly);

    size = sizeof(float) * kSize * 2;
    count = file.read((char *)buffer.data(), size);
    Q_ASSERT(count == size);

    file.close();

    pdata = buffer.data();
    for (int i = 0; i < kSize; i++) {
        kData[i] = std::complex<float> (pdata[0], pdata[1]);
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

    QVector<kTraj> trajData(samples * arms);
    complexVector kData(samples * arms);

    loadData(trajData, kData, samples * arms);

    int kWidth = 4;
    int overGridFactor = 2;
    ConvKernel kernel(kWidth, overGridFactor, 256);

    int gridSize = 234 * overGridFactor;
    complexVector gData(gridSize * gridSize);

    GridLut gridCpu(gridSize, kernel);
    GridGpu gridGpu(gridSize, kernel);
    gridGpu.prepare(trajData);
    FFT2D fft(gridSize, gridSize, false);

    QElapsedTimer timer;
    timer.start();
    for (int i = 0; i < 10; i++)
        gridCpu.gridding(trajData, kData, gData);
    qWarning() << "CPU run time =" << timer.restart() << "ms";

    for (int i = 0; i < 10; i++)
        gridGpu.gridding(kData, gData);
    cudaDeviceSynchronize();
    qWarning() << "GPU run time =" << timer.elapsed() << "ms";


    fft.fftShift(gData);
    fft.excute(gData);
    fft.fftShift(gData);

    QApplication app(argc, argv);
    displayData(gridSize, gridSize, gData, "image");

    return app.exec();
}
