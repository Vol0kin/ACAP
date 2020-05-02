#include "medianKernel.h"
#include <stdio.h>
#include <cuda.h>

__global__ void medianKernel()
{
    extern __shared__ float localWindow[];

    


    //printf("Block (%d, %d) thread (%d, %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}


float* medianFilter(float* hSrc, int width, int height, int kernelSize, int windowSize)
{
    // Size of image with replicated borders
    int borderSize = kernelSize / 2;
    int srcWidth = width + 2*borderSize;
    int srcHeight = height + 2*borderSize;

    // Allocate local memory for filtered image
    float* hDest = new float[width * height];

    // Allocate memory for images in devide
    float* dSrc;
    float* dDest;

    cudaMalloc(&dSrc, srcWidth * srcHeight * sizeof(float));
    cudaMalloc(&dDest, width * height  * sizeof(float));

    // Copy image to device
    cudaMemcpy(dSrc, hSrc, srcWidth * srcHeight * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 gridSize(width / windowSize, height / windowSize, 1);
    dim3 blockSize(windowSize, windowSize, 1);

    // Compute size of shared memory (in Bytes)
    int expandedWindowSize = windowSize + borderSize * 2;
    int sharedMemory = expandedWindowSize * expandedWindowSize * sizeof(float);

    medianKernel<<<gridSize, blockSize, sharedMemory>>>();

    // TODO: Copy result from device

    // Free memory
    cudaFree(dSrc);
    cudaFree(dDest);

    return 0;
}
