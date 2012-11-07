#ifndef __WAVE_H__
#define __WAVE_H__

#include <string.h>
#include <stdint.h>
#include <sys/time.h>

#define min(a,b) ((a)<(b) ? (a) : (b))

#define IDX(i,j,k) ((i)+size*((j)+size*(k)))


/**
 * Allocate data.
 *
 * ptr:          A pointer to the array which will be allocated
 * ptr_aligned:  A pointer which will contain the aligned address
 * x_max, y_max: The size of the grid
 */
void allocate(float** ptr, float** ptr_aligned, int x_max)
{
	*ptr = (float*) malloc (x_max * x_max * x_max * sizeof (float) + 31);
	*ptr_aligned = (float*) (((uintptr_t) *ptr + 31) & (~((uintptr_t) 31)));
}

/**
 * Get current time in seconds.
 */
double seconds()
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return ((double) tv.tv_sec) + 1e-6 * tv.tv_usec;
}

/**
 * Write a cross section of the solution in u to a file.
 */
void write(float* u, int timestep, int size)
{
	int i, j;
	char szFilename[255];
	sprintf (szFilename, "%04d.txt", timestep);
	printf ("Writing file %s...\n", szFilename);
	FILE* file = fopen (szFilename, "w");

	const int k = size / 3;
	for (j = 0; j < size; j++)
	{
		for (i = 0; i < size; i++)
			fprintf (file, "%f ", u[IDX(i,j,k)]);
		fprintf (file, "\n");
	}

	fclose (file);
}

void initialize(float* u_m1, float* u_0, float* u_1, float min_x, float dx, int size);

#endif
