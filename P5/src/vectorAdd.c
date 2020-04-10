#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void allocate_array(float **array, int N)
{
    *array = (float*)malloc(N * sizeof(float));

    if (*array == NULL)
    {
        printf("Could not allocate memory for array. Exiting...");
        exit(EXIT_FAILURE);
    }
}

void read_file(char* file, float** array, int* N)
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
    *N = n_elements * 2;

    // Allocate memory
    allocate_array(array, *N);

    // Read file and load its values
    float value;
    int i = 0;

    while((read = getline(&line, &len, fp)) != -1)
    {
        value = atof(line);

        (*array)[i] = value;
        (*array)[i + n_elements] = value;

        i++;
    }
}

void add_vectors(float *A, float *B, float *C, int N)
{
    int i;

    for (i = 0; i < N; i++)
    {
        C[i] = pow(pow(log(5*A[i]*100*B[i] + 7*A[i]) / 0.33, 3), 7);
    }
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        printf("ERROR. Expected two arguments\n");
        printf("Usage: %s [input data 1] [input data 2]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    double t1, t_time;
    float *A, *B, *C;
    int N;

    // Read files and allocate arrays
    read_file(argv[1], &A, &N);
    read_file(argv[2], &B, &N);
    allocate_array(&C, N);

    // Add vectors
    t1 = omp_get_wtime();

    add_vectors(A, B, C, N);

    t_time = omp_get_wtime() - t1;

    // Show some values
    printf("C[0] = %f\nC[1] = %f\nC[%d] = %f\nC[%d] = %f\n", C[0], C[1], N-2, C[N-2], N-1, C[N-1]);
    printf("Total time: %f\n", t_time);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
