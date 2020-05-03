#include "medianKernel.h"
#include <thrust/sort.h>
#include <chrono>

__global__ void medianKernel(float* dSrc, float* dDest, int srcWidth, int destWidth, int kernelSize,  int expandedWindowSize)
{
    
    extern __shared__ float localWindow[];

    int xStart = blockIdx.x * blockDim.x;
    int yStart = blockIdx.y * blockDim.y;

    int xIdx = threadIdx.x;
    int yIdx = threadIdx.y;

    
    for (int j = threadIdx.y; j < expandedWindowSize; j += blockDim.y)
    {
        for (int i = threadIdx.x; i < expandedWindowSize; i += blockDim.x)
        {
            localWindow[j*expandedWindowSize + i] = dSrc[(yStart + j) * srcWidth + xStart + i];

        }
    }

    __syncthreads();

    float* kernel = new float[kernelSize * kernelSize];


    for (int j = 0; j < kernelSize; j++)
    {
        for (int i = 0; i < kernelSize; i++)
        {
            kernel[j*kernelSize + i] = localWindow[(yIdx + j) * expandedWindowSize + xIdx + i];

        }
    }

    thrust::sort(thrust::seq, kernel, kernel + kernelSize * kernelSize);
    float medianVal = kernel[(kernelSize * kernelSize) / 2];
    dDest[(yStart + yIdx) * destWidth + xStart + xIdx] = medianVal;

    delete[] kernel;
}


float* medianFilter(float* hSrc, int width, int height, int kernelSize, int windowSize,
                    double& execTime)
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

    // Define grid and block sizes
    dim3 gridSize(width / windowSize, height / windowSize, 1);
    dim3 blockSize(windowSize, windowSize, 1);

    // Compute size of shared memory (in Bytes)
    int expandedWindowSize = windowSize + borderSize * 2;
    int sharedMemory = expandedWindowSize * expandedWindowSize * sizeof(float);

    auto t1 = std::chrono::high_resolution_clock::now();

    // Copy image to device
    cudaMemcpy(dSrc, hSrc, srcWidth * srcHeight * sizeof(float), cudaMemcpyHostToDevice);


    // Apply median filter by calling the kernel
    medianKernel<<<gridSize, blockSize, sharedMemory>>>(dSrc, dDest, srcWidth,
                                                        width, kernelSize,
                                                        expandedWindowSize);

    // Copy result from device
    cudaMemcpy(hDest, dDest, width * height * sizeof(float), cudaMemcpyDeviceToHost);


    auto t2 = std::chrono::high_resolution_clock::now();

    execTime = std::chrono::duration<double>(t2 - t1).count();

    // Free device memory
    cudaFree(dSrc);
    cudaFree(dDest);

    return hDest;
}
