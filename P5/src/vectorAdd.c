#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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
 * Function that adds two arrays using a mathematical formula
 */
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
    if (argc != 4)
    {
        printf("ERROR. Expected 3 arguments\n");
        printf("Usage: %s [input data 1] [input data 2] [n_rep]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    double t1, t_time;
    float *A, *B, *C;
    int N;
    int n_rep = atoi(argv[3]);

    // Read files and allocate arrays
    read_file(argv[1], &A, &N, n_rep);
    read_file(argv[2], &B, &N, n_rep);
    allocate_array(&C, N);

    // Add vectors
    t1 = omp_get_wtime();

    add_vectors(A, B, C, N);

    t_time = omp_get_wtime() - t1;

    // Show some values
    printf("C[0] = %f\nC[1] = %f\nC[%d] = %f\nC[%d] = %f\n", C[0], C[1], N-2, C[N-2], N-1, C[N-1]);
    printf("N. elements, Total time\n");
    printf("%d, %f\n", N, t_time);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
