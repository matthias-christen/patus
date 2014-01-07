#ifndef __WHISPERING_GALLERY_H__
#define __WHISPERING_GALLERY_H__

#include <stdint.h>
#include <string.h>
#include <sys/time.h>

/**
 * Allocate data.
 *
 * ptr:          A pointer to the array which will be allocated
 * ptr_aligned:  A pointer which will contain the aligned address
 * x_max, y_max: The size of the grid
 */
void allocate(float** ptr, float** ptr_aligned, int x_max, int y_max)
{
	*ptr = (float*) malloc (x_max * y_max * sizeof (float) + 31);
	*ptr_aligned = (float*) (((uintptr_t) *ptr + 31) & (~((uintptr_t) 31)));
}

/**
 * Returns the current time in microseconds.
 */
double gettime()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return (double) tp.tv_sec + (double) tp.tv_usec * 1e-6;
}

/**
 * Write the time averaged solution to a text file.
 */
void write(float* u, int x_max, int y_max)
{
	int i, j;
	FILE* file = fopen("output.txt", "w");
	
	for (j = 0; j < y_max; j++)	{
		for (i = 0; i < x_max; i++)
			fprintf (file, "%E ", (u[i + x_max * j]));
		fprintf (file, "\n");
	}
	
	fclose (file);
	printf ("Data written to file output.txt.\n");
}


float gaussianSource(float t);

void write(float* u, int x_max, int y_max);

float calculateCa(float sigma, float er);

float calculateCb(float sigma, float er);

float calculateDa(float sigma, float mur);

float calculateDb(float sigma, float mur);


#endif
