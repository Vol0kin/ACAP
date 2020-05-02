#include "medianKernel.h"
#include <stdio.h>

__global__ void medianKernel()
{
    printf("Esto es un kernel ejecutandose\n");
}


float* medianFilter(float* src, int width, int height, int kernelSize)
{
    medianKernel<<<1,1>>>();
    return 0;
}
