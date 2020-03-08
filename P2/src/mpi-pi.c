#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"

#define PI 3.141592653589793238462643

int main(int argc, char * argv[]) {
    double width, pi, sum, x;
    int intervals, i, myid, numprocs;
    double t, t_ini, t_comp, t_reduce;
    
    intervals = atoi(argv[1]);
    width = 1.0 / intervals;

    t = omp_get_wtime();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    if (myid == 0)
    {
        t_ini = omp_get_wtime() - t;
        t = omp_get_wtime();
    }    
    
    // Initialize sum
    sum = 0.0;
    
    // Do computation
    for (i = myid; i < intervals; i += numprocs)
    {
        x = (i + 0.5) * width;
        sum += 4.0 / (1.0 + x * x);
    }
    
    sum *= width;

    if (myid == 0)
    {
        t_comp = omp_get_wtime() - t;
        t = omp_get_wtime();
    } 
    
    // Reduce all values into pi
    MPI_Reduce(&sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        t_reduce = omp_get_wtime() - t;
    } 

    MPI_Finalize();

    if (myid == 0)
    {
        printf("Number of intervals: %d\n", intervals);
        printf("PI is %0.24f\n", PI);
        printf("Estimation of PI is %0.24f\n", pi);
        printf("Error: %0.24f\n", fabs(PI - pi));
        printf("Initialization time: %f seconds\n", t_ini);
        printf("Computation time: %f seconds\n", t_comp);
        printf("Reduce time: %f seconds\n", t_reduce);
    }
    
    return 0;
}
