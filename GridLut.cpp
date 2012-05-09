#include "GridLut.h"

GridLut::GridLut()
{

}

void GridLut::SetConvKernel(ConvKernel &kernel)
{
    m_kernel = kernel;
}
