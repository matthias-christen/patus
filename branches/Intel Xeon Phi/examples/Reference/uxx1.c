#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#define IDX(i,j,k) ((i+1)+nxt*((j+1)+nyt*(k+1)))

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
 * Do the calculation.
 */
int main(int argc, char** argv)
{
    int i, j, k, t;
    double t1, t2, nFlops;

    float* __restrict__ u1;
    float* __restrict__ v1;
    float* __restrict__ w1;
    float* __restrict__ xx;
    float* __restrict__ xy;
    float* __restrict__ xz;
    float* __restrict__ d1;

	if (argc != 10)
	{
		printf ("Wrong number of parameters.\n", argv[0]);
		exit (-1);
	}
	
	int nxb = atoi (argv[1]);
	int nyb = atoi (argv[2]);
	int nzb = atoi (argv[3]);

	int nxe = atoi (argv[4]);
	int nye = atoi (argv[5]);
	int nze = atoi (argv[6]);

	int nxt = atoi (argv[7]);
	int nyt = atoi (argv[8]);
	int nzt = atoi (argv[9]);

    float dth = 1. / nxt;
    float c1 = 9./8.;
    float c2 = -1./24.;


	const int T_MAX = 5;

 
    /* allocate memory */
    u1 = (float*) malloc ((nxt+4) * (nyt+4) * (nzt+4) * sizeof (float));
    v1 = (float*) malloc ((nxt+4) * (nyt+4) * (nzt+4) * sizeof (float));
    w1 = (float*) malloc ((nxt+4) * (nyt+4) * (nzt+4) * sizeof (float));
    xx = (float*) malloc ((nxt+4) * (nyt+4) * (nzt+4) * sizeof (float));
    xy = (float*) malloc ((nxt+4) * (nyt+4) * (nzt+4) * sizeof (float));
    xz = (float*) malloc ((nxt+4) * (nyt+4) * (nzt+4) * sizeof (float));
    d1 = (float*) malloc ((nxt+4) * (nyt+4) * (nzt+4) * sizeof (float));

    /* initialize the first timesteps */
	#pragma omp parallel for private (k,j,i)
	for (k = nzb; k <= nze; k++)
  	{
		for (j = nyb; j <= nye; j++)
	 	{
			for (i = nxb; i <= nxe; i++)
			{
                d1[IDX(i,j,k)] = 1. + i*0.1 + j*0.01 + k*0.001;
                u1[IDX(i,j,k)] = 2. + i*0.1 + j*0.01 + k*0.001;
                xx[IDX(i,j,k)] = 3. + i*0.1 + j*0.01 + k*0.001;
                xy[IDX(i,j,k)] = 4. + i*0.1 + j*0.01 + k*0.001;
                xz[IDX(i,j,k)] = 5. + i*0.1 + j*0.01 + k*0.001;
			}
		}
	}
	

    /* do the calculation */ 
	t1 = seconds();
	for (t = 0; t < T_MAX; t++)
	{
		#pragma omp parallel for private(k,j,i)
		for (k = nzb; k <= nze; k++)
      	{
    		for (j = nyb; j <= nye; j++)
    	 	{
    			for (i = nxb; i <= nxe; i++)
    			{
                    float d = 0.25*(d1[IDX(i,j,k)]+d1[IDX(i,j-1,k)]+d1[IDX(i,j,k-1)]+d1[IDX(i,j-1,k-1)]);
                    u1[IDX(i,j,k)]=u1[IDX(i,j,k)]+(dth/d)*(
                        c1*(xx[IDX(i,j,k)]-xx[IDX(i-1,j,k)])+
                        c2*(xx[IDX(i+1,j,k)]-xx[IDX(i-2,j,k)])+
                        c1*(xy[IDX(i,j,k)]-xy[IDX(i,j-1,k)])+
                        c2*(xy[IDX(i,j+1,k)]-xy[IDX(i,j-2,k)])+
                        c1*(xz[IDX(i,j,k)]-xz[IDX(i,j,k-1)])+
                        c2*(xz[IDX(i,j,k+1)]-xz[IDX(i,j,k-2)]));
    			}
    		}
    	}
	}
	t2 = seconds ();

    /* print statistics */    
    nFlops = (double) (nxe-nxb+1) * (double) (nye-nyb+1) * (double) (nze-nzb+1) * T_MAX * 20.0;
    printf ("FLOPs in stencil code:      %e\n", nFlops);
	printf ("Time spent in stencil code: %f\n", t2 - t1);
	printf ("Performance in GFlop/s:     %f\n", nFlops / (1e9 * (t2 -t1)));
   
    /* clean up */
	free (u1);
	free (v1);
	free (w1);
	free (xx);
	free (xy);
	free (xz);
	free (d1);

	return EXIT_SUCCESS;
}


