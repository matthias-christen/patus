#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define IDX(i,j) ((i)+x_max*(j))

#if defined(_OPENMP)
#	include <omp.h>
#endif


/**
 * Get current time in seconds.
 */
double seconds ()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return ((double) tv.tv_sec) + 1e-6 * tv.tv_usec;
}

/**
 * Do the calculation.
 */
int main(int argc, char** argv)
{
    int x_max, y_max;
    int i, j, k, t;
    int x, y, z;
    double t1, t2, nFlops;

    float* __restrict__ u_0_0 = NULL;
    float* __restrict__ u_0_1 = NULL;
    float f, s0, s1, s2, s4, s5, s8;

	const int T_MAX = 5;

	if (argc != 3)
	{
		printf ("Wrong number of parameters.\n", argv[0]);
		exit (-1);
	}

	x_max = atoi (argv[1]);
	y_max = atoi (argv[2]);


    /* allocate memory */
    u_0_0 = (float*) malloc (x_max * y_max * sizeof (float));
    u_0_1 = (float*) malloc (x_max * y_max * sizeof (float));

    f = 0.1f / (float) x_max;
    s0 = 0.2f / (float) (x_max + 1);
    s1 = 0.3f / (float) (x_max + 2);
    s2 = 0.4f / (float) (x_max + 3);
    s4 = 0.5f / (float) (x_max + 4);
    s5 = 0.6f / (float) (x_max + 5);
    s8 = 0.7f / (float) (x_max + 6);

    /* initialize the first timesteps */
	#pragma omp parallel for private (k,j,i)
	for (j = 0; j < y_max; j++)
	{
		for (i = 0; i < x_max; i++)
		{
        	u_0_0[IDX(i,j)] = 1. + i*0.1 + j*0.01;
         	u_0_1[IDX(i,j)] = 2. + i*0.1 + j*0.01;
		}
	}


    /* do the calculation */
	t1 = seconds();
	for (t = 0; t < T_MAX; t++)
	{
		#pragma omp parallel for private(y,x)
		for (y = 2; y < y_max - 2; y++)
	 	{
			for (x = 2; x < x_max - 2; x++)
			{
                u_0_1[IDX(x, y)] = f * (
                	s0 * u_0_0[IDX(x, y)] +
                	s1 * (u_0_0[IDX(x - 1, y)] + u_0_0[IDX(x + 1, y)] + u_0_0[IDX(x, y - 1)] + u_0_0[IDX(x, y + 1)]) +
                	s2 * (u_0_0[IDX(x - 1, y - 1)] + u_0_0[IDX(x + 1, y - 1)] + u_0_0[IDX(x - 1, y + 1)] + u_0_0[IDX(x + 1, y + 1)]) +
                	s4 * (u_0_0[IDX(x - 2, y)] + u_0_0[IDX(x + 2, y)] + u_0_0[IDX(x, y - 2)] + u_0_0[IDX(x, y + 2)]) +
                	s5 * (
                		u_0_0[IDX(x - 2, y - 1)] + u_0_0[IDX(x - 1, y - 2)] + u_0_0[IDX(x + 1, y - 2)] + u_0_0[IDX(x + 2, y - 1)] +
                		u_0_0[IDX(x - 2, y + 1)] + u_0_0[IDX(x - 1, y + 2)] + u_0_0[IDX(x + 1, y + 2)] + u_0_0[IDX(x + 2, y + 1)]
                	) +
                	s8 * (u_0_0[IDX(x - 2, y - 2)] + u_0_0[IDX(x + 2, y - 2)] + u_0_0[IDX(x - 2, y + 2)] + u_0_0[IDX(x + 2, y + 2)])
                );
			}
		}

    	float* tmp = u_0_0;
    	u_0_0 = u_0_1;
    	u_0_1 = tmp;
	}
	t2 = seconds ();

    /* print statistics */
    nFlops = (double) (x_max-4) * (double) (y_max-4) * T_MAX * 31.0;
    printf ("FLOPs in stencil code:      %e\n", nFlops);
	printf ("Time spent in stencil code: %f\n", t2 - t1);
	printf ("Performance in GFlop/s:     %f\n", nFlops / (1e9 * (t2 -t1)));

    /* clean up */
	free (u_0_0);
	free (u_0_1);

	return EXIT_SUCCESS;
}


