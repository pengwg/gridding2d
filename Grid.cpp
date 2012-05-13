#include "Grid.h"

Grid::Grid(int gridSize)
    : m_gridSize(gridSize), m_kernel(nullptr)
{

}

Grid::~Grid()
{
    if (m_kernel) {
        delete m_kernel;
    }
}

void Grid::setConvKernel(ConvKernel &kernel)
{
    if (!m_kernel)
        m_kernel = new ConvKernel(kernel);
    else
        *m_kernel = kernel;
}
