/**
 * This program solves the classical wave equation using a fourth-order
 * finite difference scheme in space and a second-order scheme in time.
 *
 * Basic implementation in C, without Patus.
 *
 * Author: Matthias Christen
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utils.h"


// Length of one side of the cube of the simulation
// IMPORTANT: Needs to be divisible by 8
#define GRID_SIZE 128

// Default number of time steps
#define T_MAX 1000

// Write an output file every WRITE_EVERY timesteps
#define WRITE_EVERY 10


int main(int argc, char** argv)
{
    int t;
    double t1, t2, num_flops;
	
	int num_timesteps = 0;
	if (argc > 1)
		num_timesteps = atoi (argv[1]);
	if (num_timesteps == 0)
		num_timesteps = T_MAX;
	
	const int size = GRID_SIZE + 4;

	// compute some simulation-specific parameters
	const float min_x = -1.0f;
	const float max_x = 1.0f;
	const float dx = (max_x - min_x) / (GRID_SIZE + 1);
	const float dt = dx / 20.0f;

	const float dt_dx_square = dt * dt / (dx * dx);

 
    // Allocate memory and initialize.
    float *_U_0, *U_0;
    float *_U_m1, *U_m1;
    float *_U_1, *U_1;
    
    allocate(&_U_0, &U_0, size);
    allocate(&_U_m1, &U_m1, size);
    allocate(&_U_1, &U_1, size);

	initialize(U_m1, U_0, U_1, min_x, dx, size);
    write(U_0, 0, size);

	const float c0 = 2.0f - 7.5f * dt_dx_square;
	const float c1 = 4.0f / 3.0f * dt_dx_square;
	const float c2 = -1.0f / 12.0f * dt_dx_square;

    // do the calculation
	t1 = seconds();
		
	for (t = 0; t < num_timesteps; t++)
	{
		int i, j, k;
		
		#pragma omp parallel for private (i, j, k)
		for (k = 2; k < size - 2; k++)
      	{
    		for (j = 2; j < size - 2; j++)
    	 	{
    			for (i = 2; i < size - 2; i++)
    			{
    				U_1[IDX(i,j,k)] =  c0 * U_0[IDX(i,j,k)] - U_m1[IDX(i,j,k)] +
						c1 * (
    						U_0[IDX(i+1, j, k)] + U_0[IDX(i-1, j, k)] +
	                       	U_0[IDX(i, j+1, k)] + U_0[IDX(i, j-1, k)] +
    	                  	U_0[IDX(i, j, k+1)] + U_0[IDX(i, j, k-1)]
						) +
						c2 * (
    						U_0[IDX(i+2, j, k)] + U_0[IDX(i-2, j, k)] +
	                       	U_0[IDX(i, j+2, k)] + U_0[IDX(i, j-2, k)] +
    	                  	U_0[IDX(i, j, k+2)] + U_0[IDX(i, j, k-2)]
    	                );
    			}
    		}
    	}

		// write output
		if (t % WRITE_EVERY == 0)
		    write(U_1, t + 1, size);

		// swap the pointers to the grids
		float* tmp = U_m1;
		U_m1 = U_0;
		U_0 = U_1;
		U_1 = tmp;
	}

	t2 = seconds();

    // print statistics (16 FLOPs per stencil evaluation)
    num_flops = (double) (size - 4) * (double) (size - 4) * (double) (size - 4)
    	* num_timesteps * 16.0;

    printf("Flops in stencil code:      %e\n", num_flops);
	printf("Time spent in stencil code: %f\n", t2 - t1);
	printf("Performance in GFlop/s:     %f\n", num_flops / (1e9 * (t2 - t1)));

    // clean up
	free(_U_m1);
	free(_U_0);
	free(_U_1);

	return 0;
}

/**
 * Initialize the grids.
 */
void initialize(float* u_m1, float* u_0, float* u_1, float min_x, float dx, int size)
{
	int i, j, k;
	
	// do the actual initialization with the correct values
    for (k = 0; k < size; k++)
    {
		for (j = 0; j < size; j++)
		{
			for (i = 0; i < size; i++)
			{
				if (i == 0 || i == 1 || i == size - 2 || i == size - 1 ||
					j == 0 || j == 1 || j == size - 2 || j == size - 1 ||
					k == 0 || k == 1 || k == size - 2 || k == size - 1)
				{
					u_0[IDX(i, j, k)] = 0.0f;
					u_m1[IDX(i, j, k)] = 0.0f;
				}
				else
				{
					float x = (i - 1) * dx + min_x;
					float y = (j - 1) * dx + min_x;
					float z = (k - 1) * dx + min_x;

			    	u_0[IDX(i,j,k)] = sinf(2 * M_PI * x) * sinf(2 * M_PI * y) * sinf(2 * M_PI * z);
			     	u_m1[IDX(i,j,k)] = u_0[IDX(i,j,k)];
				}
			}
		}
	}	
}

