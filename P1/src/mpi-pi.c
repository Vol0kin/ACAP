#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"

#define PI 3.141592653589793238462643

int main(int argc, char * argv[]) {
    double width, sum, pi, sum, x;
    int intervals, i, myid, numprocs;
    
    intervals = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, & numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, & myid);
    
    // Broadcast intervals value so all processes have it
    MPI_Bcast(&intervals, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Compute width and initialize sum
    width = 1.0 / intervals;    
    sum = 0.0;
    
    // Do computation
    for (i = myid; i < intervals; i += numprocs)
    {
        x = (i + 0.5) * width;
        sum += 4.0 / (1.0 + x * x);
    }
    
    sum *= width;
    
    // Reduce all values into PI
    MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    
    printf("Number of intervals: %d\n", intervals);
    printf("PI is %0.24f\n", PI);
    printf("Estimation of PI is %0.24f\n", sum);
    printf("Error: %0.24f\n", fabs(PI - sum));
    
    return 0;
}
