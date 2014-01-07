/**
 * This program solves the classical wave equation using a fourth-order
 * finite difference scheme in space and a second-order scheme in time.
 *
 * This example program shows how to use a stencil specification
 * embedded into the application source file.
 *
 * Author: Matthias Christen
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "utils.h"

// Include header files which will be created after running Patus
// (cf. Makefile)
#include "wave_patus.h"


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
    // In the embedded stencil specification (see below), we use a single grid
    // "U". If the stencil specification uses more than one timestep per grid,
    // Patus expects that there are multiple grid arrays, which are named
    // "U_<timestep>", where <timestep> is the temporal index occurring in the
    // stencil specification. Minus signs are replaced by the letter "m".
    // In the future, we'll allow binding variables to Patus grids.
    float *_U_0, *U_0;
    float *_U_m1, *U_m1;
    float *_U_1, *U_1;
    
    allocate(&_U_0, &U_0, size);
    allocate(&_U_m1, &U_m1, size);
    allocate(&_U_1, &U_1, size);

	initialize(U_m1, U_0, U_1, min_x, dx, size);
    write(U_0, 0, size);


    // do the calculation
	t1 = seconds();

	#pragma omp parallel private (t)
	for (t = 0; t < num_timesteps; t++)
	{
		// Embed a Patus stencil specification.
		// When Patus is run, this stencil specification will be replaced
		// by a call to the generated function.

		#pragma patus begin-stencil-specification
		stencil wave (
			float grid U(0 .. size-1, 0 .. size-1, 0 .. size-1), 
			float param dt_dx_square)
		{
			domainsize = (2 .. size-3, 2 .. size-3, 2 .. size-3);

			operation
			{
				float c1 = 2 - 15/2 * dt_dx_square;
				float c2 = 4/3 * dt_dx_square;
				float c3 = -1/12 * dt_dx_square;
				
				U[x, y, z; t+1] = c1 * U[x, y, z; t] - U[x, y, z; t-1] +
					c2 * (U[x+1, y, z; t] + U[x-1, y, z; t] + U[x, y+1, z; t] +
						U[x, y-1, z; t] + U[x, y, z+1; t] + U[x, y, z-1; t]) +
					c3 * (U[x+2, y, z; t] + U[x-2, y, z; t] + U[x, y+2, z; t] +
						U[x, y-2, z; t] + U[x, y, z+2; t] + U[x, y, z-2; t]);					
			}
		}
		#pragma patus end-stencil-specification
		#pragma omp barrier

		#pragma omp single
		{
			// write output
			if (t % WRITE_EVERY == 0)
			    write(U_1, t + 1, size);

			// swap the pointers to the grids
			float* tmp = U_m1;
			U_m1 = U_0;
			U_0 = U_1;
			U_1 = tmp;
		}
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
	
	// do the NUMA-aware initialization
	initialize_wave(u_m1, u_0, u_1, dx, size);

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

