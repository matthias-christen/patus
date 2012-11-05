#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "cycle.h"


ticks g_tickStart;
double g_fTimeStart;


double gettime ()
{
	struct timeval tp;
	struct timezone tzp;
	int i;

	i = gettimeofday (&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}



void tic ()
{
	g_fTimeStart = gettime ();
	g_tickStart = getticks ();
}

void toc (long nFlopsPerStencil, long nStencilComputationsCount, long nBytesTransferred)
{
	ticks tickEnd = getticks ();
	double fTime = gettime () - g_fTimeStart;

	long nTotalFlops = nFlopsPerStencil * nStencilComputationsCount;
	printf ("Flops / stencil call:  %ld\n", nFlopsPerStencil);
	printf ("Stencil computations:  %ld\n", nStencilComputationsCount);
	printf ("Bytes transferred:     %ld\n", nBytesTransferred);
	printf ("Total Flops:           %ld\n", nTotalFlops);
	printf ("Seconds elapsed:       %f\n", fTime);
	printf ("Performance:           %f GFlop/s\n", nTotalFlops / fTime / 1.e9);
	printf ("Bandwidth utilization: %f GB/s\n", nBytesTransferred / fTime / 1.e9);

	printf ("%f\n", elapsed (tickEnd, g_tickStart));
}
