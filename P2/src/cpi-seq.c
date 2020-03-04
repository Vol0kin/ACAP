#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

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
  
  clock_t start = clock();
  
  for (i=0; i<intervals; ++i) {
    /* 
      This computes the integral in the central point
      This is actually the best way to approximate PI since the approximation
      error is the lowest one out of all the different methods
    */
    register double x = (i + 0.5) * width;
    
    // This computes the integral in the left end
    //register double x = i * width;
    
    // This computes the integral in the right end
    //register double x = (i + 1.0) * width;
    
    sum += 4.0 / (1.0 + x * x);
  }
  sum *= width;
  
  clock_t end = clock();
  
  float total_time = (float) (end - start) / CLOCKS_PER_SEC;
  
  printf("PI is %0.24f\n", PI);
  printf("Estimation of PI is %0.24f\n", sum);
  printf("Error: %0.24f\n", fabs(PI - sum));
  printf("Time spent: %f\n", total_time);

  return(0);
}

