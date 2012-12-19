#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

//#define WRITE_OUTPUT
#define NONUMA
#define IDX(i,j,k) ((i)+x_max*((j)+y_max*(k)))

#ifndef M_PI
#	define M_PI 3.14159265358979323846
#endif

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
 * Write a cross section of the solution in u to a file.
 */
void write (float* u, int timestep, int x_max, int y_max, int z_max)
{
	int i, j;
	char szFilename[255];
	sprintf (szFilename, "%04d.txt", timestep);
	printf ("Writing file %s...\n", szFilename);
	FILE* file = fopen (szFilename, "w");

	const int k = z_max / 3;
	for (j = 0; j < y_max; j++)
	{
		for (i = 0; i < x_max; i++)
			fprintf (file, "%f ", u[IDX(i,j,k)]);
		fprintf (file, "\n");
	}

	fclose (file);
}

int malloc_error (const char* err)
{
	fprintf (stderr, "Failed to allocate the field %s.\n", err);
	return EXIT_FAILURE;
}

/**
 * Do the calculation.
 */
int main(int argc, char** argv)
{
    int i, j, k, t;
    double t1, t2, nFlops;

    float* __restrict__ u_0_m1 = NULL;
    float* __restrict__ u_0_0 = NULL;
    float* __restrict__ u_0_1 = NULL;

	if (argc != 4)
	{
		printf ("Wrong number of parameters. Syntax:\n%s <x_max> <y_max> <z_max>\n", argv[0]);
		exit (-1);
	}
	
	int x_max = atoi (argv[1]) + 4;
	int y_max = atoi (argv[2]) + 4;
	int z_max = atoi (argv[3]) + 4;

	const int T_MAX = 5;

	const float MIN = -1.f;
	const float MAX = 1.f;
	const float DX = (MAX - MIN) / (x_max - 3);
	const float DT = DX / 2.0f;

	const float DT_DX_SQUARE = DT * DT / (DX * DX);

 
    /* allocate memory */
    u_0_m1 = (float*) malloc (x_max * y_max * z_max * sizeof (float));
    if (u_0_m1 == NULL)
        return malloc_error ("u_0_m1");
    u_0_0 = (float*) malloc (x_max * y_max * z_max * sizeof (float));
    if (u_0_0 == NULL)
    {
        free (u_0_m1);
        return malloc_error ("u_0_0");
    }
    u_0_1 = (float*) malloc (x_max * y_max * z_max * sizeof (float));
    if (u_0_1 == NULL)
    {
        free (u_0_m1);
        free (u_0_0);
        return malloc_error ("u_0_1");
    }

    /* initialize the first timesteps */
#ifdef NONUMA
	memset (u_0_m1, 0, x_max * y_max * z_max * sizeof (float));
	memset (u_0_0, 0, x_max * y_max * z_max * sizeof (float));
	memset (u_0_1, 0, x_max * y_max * z_max * sizeof (float));
#endif

	#pragma omp parallel for private (k,j,i)
    for (k = 2; k < z_max - 2; k++)
    {
		for (j = 2; j < y_max - 2; j++)
		{
			for (i = 2; i < x_max - 2; i++)
			{
				float x = (i - 1) * DX + MIN;
				float y = (j - 1) * DX + MIN;
				float z = (k - 1) * DX + MIN;

#ifndef NONUMA
				if (k == 2)
				{
					u_0_m1[IDX(i, j, 0)] = 0;
					u_0_m1[IDX(i, j, 1)] = 0;
					u_0_0[IDX(i, j, 0)] = 0;
					u_0_0[IDX(i, j, 1)] = 0;
				}
				if (k == z_max - 3)
				{
					u_0_m1[IDX(i, j, z_max - 2)] = 0;
					u_0_m1[IDX(i, j, z_max - 1)] = 0;
					u_0_0[IDX(i, j, z_max - 2)] = 0;
					u_0_0[IDX(i, j, z_max - 1)] = 0;
				}
				if (j == 2)
				{
					u_0_m1[IDX(i, 0, k)] = 0;
					u_0_m1[IDX(i, 1, k)] = 0;
					u_0_0[IDX(i, 0, k)] = 0;
					u_0_0[IDX(i, 1, k)] = 0;
				}
				if (j == y_max - 3)
				{
					u_0_m1[IDX(i, y_max - 2, k)] = 0;
					u_0_m1[IDX(i, y_max - 1, k)] = 0;
					u_0_0[IDX(i, y_max - 2, k)] = 0;
					u_0_0[IDX(i, y_max - 1, k)] = 0;
				}
				if (i == 2)
				{
					u_0_m1[IDX(0, j, k)] = 0;
					u_0_m1[IDX(1, j, k)] = 0;
					u_0_0[IDX(0, j, k)] = 0;
					u_0_0[IDX(1, j, k)] = 0;
				}
				if (i == x_max - 3)
				{
					u_0_m1[IDX(x_max - 2, j, k)] = 0;
					u_0_m1[IDX(x_max - 1, j, k)] = 0;
					u_0_0[IDX(x_max - 2, j, k)] = 0;
					u_0_0[IDX(x_max - 1, j, k)] = 0;
				}
#endif

	        	u_0_0[IDX(i,j,k)] = (float) (sin (2 * M_PI * x) * sin(2 * M_PI * y) * sin(2 * M_PI * z));
	         	u_0_m1[IDX(i,j,k)] = u_0_0[IDX(i,j,k)];
			}
		}
	}
	
#ifdef WRITE_OUTPUT
    write (u_0_0, 0, x_max, y_max, z_max);
#endif	

	const float c1 = 2.0f - DT_DX_SQUARE * 7.5f;
	const float c2 = DT_DX_SQUARE * 4.0f / 3.0f;
	const float c3 = DT_DX_SQUARE * (-1.0f/ 12.0f);

    /* do the calculation */ 
	t1 = seconds();
	for (t = 0; t < T_MAX; t++)
	{
		#pragma omp parallel for private(k,j,i)
		for (k = 2; k < z_max - 2; k++)
      	{
    		for (j = 2; j < y_max - 2; j++)
    	 	{
    			for (i = 2; i < x_max - 2; i++)
    			{
    				u_0_1[IDX(i,j,k)] =  c1 * u_0_0[IDX(i,j,k)] - u_0_m1[IDX(i,j,k)] +
    					+ c2 * (
    						u_0_0[IDX(i+1,j,k)] + u_0_0[IDX(i-1,j,k)]+
	                       	u_0_0[IDX(i,j+1,k)] + u_0_0[IDX(i,j-1,k)]+
    	                  	u_0_0[IDX(i,j,k+1)] + u_0_0[IDX(i,j,k-1)])
    	                + c3 * (
    						u_0_0[IDX(i+2,j,k)] + u_0_0[IDX(i-2,j,k)]+
	                       	u_0_0[IDX(i,j+2,k)] + u_0_0[IDX(i,j-2,k)]+
    	                  	u_0_0[IDX(i,j,k+2)] + u_0_0[IDX(i,j,k-2)]);
    			}
    		}
    	}

#ifdef WRITE_OUTPUT
        write (u_0_1, t + 1, x_max, y_max, z_max);
#endif

    	float* tmp = u_0_m1;
    	u_0_m1 = u_0_0;
    	u_0_0 = u_0_1;
    	u_0_1 = tmp;
	}
	t2 = seconds ();

    /* print statistics */    
    nFlops = (double) (x_max-4) * (double) (y_max-4) * (double) (z_max-4) * T_MAX * 16.0;
    printf ("FLOPs in stencil code:      %e\n", nFlops);    
	printf ("Time spent in stencil code: %f\n", t2 - t1);
	printf ("Performance in GFlop/s:     %f\n", nFlops / (1e9 * (t2 -t1)));
   
    /* clean up */
	free (u_0_m1);
	free (u_0_0);
	free (u_0_1);
	
	return EXIT_SUCCESS;
}


