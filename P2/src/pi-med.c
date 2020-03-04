#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

main(int argc, char **argv)
{
  register double width, sum;
  register int intervals, i;
  const double PI = 3.141592653589793238462643;

  /* get the number of intervals */
  intervals = atoi(argv[1]);
  width = 1.0 / intervals;

  /* do the computation */
  sum = 0;
  
  double start = omp_get_wtime();
  
  for (i=0; i<intervals; ++i) {
    // Compute the integral in the central point
    register double x = (i + 0.5) * width;
    
    sum += 4.0 / (1.0 + x * x);
  }
  sum *= width;
  
  double time = omp_get_wtime() - start;
  
  printf("Number of intervals: %d\n", intervals);
  printf("PI is %0.24f\n", PI);
  printf("Estimation of PI is %0.24f\n", sum);
  printf("Error: %0.24f\n", fabs(PI - sum));
  printf("Time spent: %f\n", time);

  return(0);
}

