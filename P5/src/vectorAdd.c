#include <stdio.h>
#include <stdlib.h>

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

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        printf("ERROR. Expected two arguments\n");
        printf("Usage: %s [input data 1] [input data 2]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    // Read files and allocate arrays
    float *A, *B, *C;
    int N;

    read_file(argv[1], &A, &N);
    read_file(argv[2], &B, &N);
    allocate_array(&C, N);


    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
