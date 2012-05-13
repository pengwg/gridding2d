#ifndef GRIDLUT_H
#define GRIDLUT_H

#include "Grid.h"


class GridLut : public Grid
{
public:
    GridLut(int gridSize);

    virtual void gridding(QVector<kData> &dataSet, complexVector &gDataSet);

};

#endif // GRIDLUT_H
