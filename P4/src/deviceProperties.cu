#include <cuda.h>
#include <stdio.h>

int getSPcores(cudaDeviceProp devProp)
{  
    int cores = 0;

    switch (devProp.major)
    {
        case 2: // Fermi
            if (devProp.minor == 1) cores = 48;
            else cores = 32;
            break;
        case 3: // Kepler
            cores = 192;
            break;
        case 5: // Maxwell
            cores = 128;
            break;
        case 6: // Pascal
            if ((devProp.minor == 1) || (devProp.minor == 2)) cores = 128;
            else if (devProp.minor == 0) cores = 64;
            else printf("Unknown device type\n");
            break;
        case 7: // Volta and Turing
            if ((devProp.minor == 0) || (devProp.minor == 5)) cores = 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n"); 
            break;
    }

    return cores;
}

int main()
{
    int n_devices;

    cudaGetDeviceCount(&n_devices);

    int i;

    for (i = 0; i < n_devices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        int numSPs = getSPcores(prop);

        printf("Device number: %d\n", i);
        printf(" Device name: %s\n", prop.name);
        printf(" Number of SMs: %d\n", prop.multiProcessorCount);
        printf(" Number of SPs per SM: %d\n", numSPs);
        printf(" Total Number of SPs: %d\n", numSPs * prop.multiProcessorCount);
        printf(" Total Available Global Memory Size (MB): %zu\n", prop.totalGlobalMem / (1<<20));
        printf(" Shared Memory Per SM (KB): %zu\n", prop.sharedMemPerMultiprocessor / (1<<10));
        printf(" Shared Memory Per Block (KB): %zu\n", prop.sharedMemPerBlock / (1<<10));
        printf(" Max. Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf(" Max. Threads Per Dimension: x: %d, y: %d, z: %d\n",
                prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    }

    return 0;
}
