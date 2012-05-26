#ifndef GRIDLUT_H
#define GRIDLUT_H

#include "Grid.h"


class GridLut : public Grid
{
public:
    GridLut(int gridSize, ConvKernel &kernel);

    virtual void gridding(QVector<kTraj> &trajData, complexVector &kData, complexVector &gData);

};

#endif // GRIDLUT_H
