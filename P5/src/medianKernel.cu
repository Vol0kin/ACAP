#include "medianKernel.h"
#include <thrust/sort.h>
#include <chrono>

__global__ void medianKernel(float* dSrc, float* dDest, int srcWidth, int destWidth, int kernelSize,
                             int windowSize, int expandedWindowSize, int pixelsPerThread)
{
    // Window which contains all the pixels that will be used in this block
    extern __shared__ float localWindow[];

    int xStartBlock = blockIdx.x * windowSize;
    int yStartBlock = blockIdx.y * windowSize;

    int xIdx = threadIdx.x;
    int yIdx = threadIdx.y;

    int xStartWindow = xIdx * pixelsPerThread;
    int yStartWindow = yIdx * pixelsPerThread;

    // Load local window from global memory and store it in local memory
    for (int j = yIdx; j < expandedWindowSize; j += blockDim.y)
    {
        for (int i = xIdx; i < expandedWindowSize; i += blockDim.x)
        {
            localWindow[j*expandedWindowSize + i] = dSrc[(yStartBlock + j) * srcWidth + xStartBlock + i];

        }
    }

    // Wait for all threads in the block to finish loading the data
    __syncthreads();

    // Allocate memory for kernel
    int kernelSquareSize = kernelSize * kernelSize;
    float* kernel = new float[kernelSquareSize];

    // Process local region inside local window
    for (int j = 0; j < pixelsPerThread; j++)
    {
        for (int i = 0; i < pixelsPerThread; i++)
        {
            // Get kernel's values
            for (int y = 0; y < kernelSize; y++)
            {
                for (int x = 0; x < kernelSize; x++)
                {
                    kernel[y*kernelSize + x] = localWindow[(yStartWindow + j + y) * expandedWindowSize + xStartWindow + i + x];
                }
            }

            // Sort values and get median
            thrust::sort(thrust::seq, kernel, kernel + kernelSquareSize);
            float median = kernel[kernelSquareSize / 2];
            dDest[(yStartBlock + yStartWindow + j) * destWidth + xStartBlock + xStartWindow + i] = median;
        }
    }

    // Free memory
    delete[] kernel;
}


float* medianFilter(float* hSrc, int width, int height, int kernelSize, int windowSize,
                    int pixelsPerThread, double& execTime)
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
    dim3 gridSize((width - 1) / windowSize + 1, (height - 1) / windowSize + 1, 1);
    dim3 blockSize(windowSize / pixelsPerThread, windowSize / pixelsPerThread, 1);

    // Compute size of shared memory (in Bytes)
    int expandedWindowSize = windowSize + borderSize * 2;
    int sharedMemory = expandedWindowSize * expandedWindowSize * sizeof(float);

    auto t1 = std::chrono::high_resolution_clock::now();

    // Copy image to device
    cudaMemcpy(dSrc, hSrc, srcWidth * srcHeight * sizeof(float), cudaMemcpyHostToDevice);


    // Apply median filter by calling the kernel
    medianKernel<<<gridSize, blockSize, sharedMemory>>>(dSrc, dDest, srcWidth,
                                                        width, kernelSize,
                                                        windowSize,
                                                        expandedWindowSize,
                                                        pixelsPerThread);

    // Copy result from device
    cudaMemcpy(hDest, dDest, width * height * sizeof(float), cudaMemcpyDeviceToHost);


    auto t2 = std::chrono::high_resolution_clock::now();
    execTime = std::chrono::duration<double>(t2 - t1).count();

    // Free device memory
    cudaFree(dSrc);
    cudaFree(dDest);

    return hDest;
}
