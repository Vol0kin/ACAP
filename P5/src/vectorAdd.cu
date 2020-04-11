#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>

/*
 * Function that allocates an array of N floats
 */
void allocate_array(float **array, int N)
{
    *array = (float*)malloc(N * sizeof(float));

    if (*array == NULL)
    {
        printf("Could not allocate memory for array. Exiting...");
        exit(EXIT_FAILURE);
    }
}

/*
 * Function that reads a file and writes its content repeated n_rep times
 * in an array of size N * n_rep
 */
void read_file(char* file, float** array, int* N, int n_rep)
{
    FILE *fp;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    // Get file descriptor and check if the file exists
    fp = fopen(file, "r");

    if (fp == NULL)
    {
        printf("Error. File %s not found!", file);
        exit(EXIT_FAILURE);
    }

    // Get array size and multiply it by 2
    // Array elements are duplicated
    read = getline(&line, &len, fp);
    int n_elements = atoi(line);
    *N = n_elements * n_rep;

    // Allocate memory
    allocate_array(array, *N);

    // Read file and load its values
    float value;
    int i = 0;
    int j;

    while((read = getline(&line, &len, fp)) != -1)
    {
        value = atof(line);

        (*array)[i] = value;

        for (j = 1; j < n_rep; j++)
        {
            (*array)[i + j * n_elements] = value;
        }

        i++;
    }
}

/*
 * CUDA function that adds two arrays using a mathematical formula
 */
__global__ void addVectorsKernel(float *d_A, float *d_B, float *d_C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        d_C[i] = pow(pow(log(5*d_A[i]*100*d_B[i] + 7*d_A[i]) / 0.33, 3), 7);
    }
}

int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        printf("ERROR. Expected 3 arguments\n");
        printf("Usage: %s [input data 1] [input data 2] [n_rep]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    double t1, t_time;
    float *h_A, *h_B, *h_C;
    int N;
    int n_rep = atoi(argv[3]);

    // Read files and allocate arrays
    read_file(argv[1], &h_A, &N, n_rep);
    read_file(argv[2], &h_B, &N, n_rep);
    allocate_array(&h_C, N);

    // Allocate CUDA arrays and copy data
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);


    // Set number of threads and blocks

    // Use 1024 threads per block since the device supports it
    int DIM_BLOCK = 1 << 10;
    int DIM_GRID = ((N - 1) / DIM_BLOCK) + 1;

    
    // Add vectors and retrieve information from device
    t1 = omp_get_wtime();

    addVectorsKernel<<<DIM_GRID, DIM_BLOCK>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    t_time = omp_get_wtime() - t1;

    // Show some values
    printf("h_C[0] = %f\nh_C[1] = %f\nh_C[%d] = %f\nh_C[%d] = %f\n",
            h_C[0], h_C[1], N-2, h_C[N-2], N-1, h_C[N-1]);
    printf("N. elements, Total time\n");
    printf("%d, %f\n", N, t_time);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
