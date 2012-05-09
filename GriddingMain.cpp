#include "ConvKernel.h"

int main(int argc, char *argv[])
{
    int kWidth = 4;
    int overGridFactor = 2;

    GriddingKernel kernel(kWidth, overGridFactor);
    kernel.GetKernelData();

    return 0;
}
