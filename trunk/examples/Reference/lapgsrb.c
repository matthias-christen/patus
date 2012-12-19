#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define IDX(i,j,k) ((i)+x_max*((j)+y_max*(k)))

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
    int x_max, y_max, z_max;
    int i, j, k, t;
    int x, y, z;
    double t1, t2, nFlops;

    float* __restrict__ u_0_0 = NULL;
    float* __restrict__ u_0_1 = NULL;
    float alpha, beta, c0, c1, c2, c3;

	const int T_MAX = 5;

	if (argc != 4)
	{
		printf ("Wrong number of parameters.\n", argv[0]);
		exit (-1);
	}

	x_max = atoi (argv[1]);
	y_max = atoi (argv[2]);
	z_max = atoi (argv[3]);


    /* allocate memory */
    u_0_0 = (float*) malloc (x_max * y_max * z_max * sizeof (float));
    u_0_1 = (float*) malloc (x_max * y_max * z_max * sizeof (float));

    alpha = 1.f / (float) x_max;
    beta = 2.f / (float) y_max;

    c0 = alpha + 6 * beta*beta;
	c1 = alpha * beta;
	c3 = beta*beta;
	c2 = 2 * c3;

    /* initialize the first timesteps */
	#pragma omp parallel for private (k,j,i)
    for (k = 0; k < z_max; k++)
    {
		for (j = 0; j < y_max; j++)
		{
			for (i = 0; i < x_max; i++)
			{
	        	u_0_0[IDX(i,j,k)] = 1. + i*0.1 + j*0.01 + k*0.001;
	         	u_0_1[IDX(i,j,k)] = 2. + i*0.1 + j*0.01 + k*0.001;
			}
		}
	}


    /* do the calculation */
	t1 = seconds();
	for (t = 0; t < T_MAX; t++)
	{
		#pragma omp parallel for private(z,y,x)
		for (z = 2; z < z_max - 2; z++)
      	{
    		for (y = 2; y < y_max - 2; y++)
    	 	{
    			for (x = 2; x < x_max - 2; x++)
    			{
                    u_0_1[IDX(x, y, z)] = c0 * u_0_0[IDX(x, y, z)] +
                        c1 * (
                            u_0_0[IDX(x+1, y, z)] + u_0_0[IDX(x-1, y, z)] +
                            u_0_0[IDX(x, y+1, z)] + u_0_0[IDX(x, y-1, z)] +
                            u_0_0[IDX(x, y, z+1)] + u_0_0[IDX(x, y, z-1)]
                        ) +
                        c2 * (
            				u_0_0[IDX(x+1, y+1, z)] + u_0_0[IDX(x+1, y, z+1)] + u_0_0[IDX(x+1, y, z-1)] + u_0_0[IDX(x+1, y-1, z)] +
            				u_0_0[IDX(x, y+1, z+1)] + u_0_0[IDX(x, y+1, z-1)] + u_0_0[IDX(x, y-1, z+1)] + u_0_0[IDX(x, y-1, z-1)] +
            				u_0_0[IDX(x-1, y+1, z)] + u_0_0[IDX(x-1, y, z+1)] + u_0_0[IDX(x-1, y, z-1)] + u_0_0[IDX(x-1, y-1, z)]
                        ) +
                        c3 * (
            				u_0_0[IDX(x+2, y, z)] + u_0_0[IDX(x-2, y, z)] +
                            u_0_0[IDX(x, y+2, z)] + u_0_0[IDX(x, y-2, z)] +
                            u_0_0[IDX(x, y, z+2)] + u_0_0[IDX(x, y, z-2)]
                        );
    			}
    		}
    	}

    	float* tmp = u_0_0;
    	u_0_0 = u_0_1;
    	u_0_1 = tmp;
	}
	t2 = seconds ();

    /* print statistics */
    nFlops = (double) (x_max-4) * (double) (y_max-4) * (double) (z_max-4) * T_MAX * 32.0;
    printf ("FLOPs in stencil code:      %e\n", nFlops);
	printf ("Time spent in stencil code: %f\n", t2 - t1);
	printf ("Performance in GFlop/s:     %f\n", nFlops / (1e9 * (t2 -t1)));

    /* clean up */
	free (u_0_0);
	free (u_0_1);

	return EXIT_SUCCESS;
}


