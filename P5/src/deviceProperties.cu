#include <cuda.h>
#include <stdio.h>

int main()
{
    int n_devices;

    cudaGetDeviceCount(&n_devices);

    int i;

    for (i = 0; i < n_devices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device number: %d\n", i);
        printf(" Device name: %s\n", prop.name);
        printf(" Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf(" Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf(" Peak Memory Bandwidth (GB/s): %f\n",
                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf(" Max. Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf(" Max. Threads Per Dimension: x: %d, y: %d, z: %d\n",
                prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf(" Number of multiprocessors: %d\n", prop.multiProcessorCount);
        //printf(" Max. Number of Threads: %d\n", prop.maxThreadsPerBlock * prop.multiProcessorCount);
    }

    return 0;
}
