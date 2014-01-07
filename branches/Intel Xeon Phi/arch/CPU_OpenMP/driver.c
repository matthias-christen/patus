#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>


#pragma patus forward_decls

int main (int argc, char** argv)
{
	int i;

	// prepare grids
	#pragma patus declare_grids
	#pragma patus allocate_grids

	// initialize
	#pragma omp parallel
	{
		#pragma patus initialize_grids
	}
	
	// write output
	if (has_arg ("-o", argc, argv))
	{
		#pragma patus write_grids("%s.0.data", input)
	}

	long nFlopsPerStencil = PATUS_FLOPS_PER_STENCIL;
	long nGridPointsCount = 5 * PATUS_GRID_POINTS_COUNT;
	long nBytesTransferred = 5 * PATUS_BYTES_TRANSFERRED;

	// warm up
	#pragma omp parallel
	{
		#pragma patus compute_stencil
	}
	
	// run the benchmark
	tic ();
	#pragma omp parallel private(i)
	for (i = 0; i < 5; i++)
	{
		#pragma patus compute_stencil
		#pragma omp barrier
	}
	toc (nFlopsPerStencil, nGridPointsCount, nBytesTransferred);
	
	// write output
	if (has_arg ("-o", argc, argv))
	{
		#pragma omp parallel
		{
			#pragma patus initialize_grids
			#pragma omp barrier
			#pragma patus compute_stencil
		}
		#pragma patus write_grids("%s.1.data", output)
	}
	
	// validate
	if (PATUS_DO_VALIDATION)
	{
		#pragma omp parallel
		{
			#pragma patus initialize_grids
			#pragma omp barrier
			#pragma patus compute_stencil
		}
		#pragma patus validate_computation
		if (PATUS_VALIDATES)
			puts ("Validation OK.");
		else
		{
			#pragma patus deallocate_grids
			puts ("Validation failed.");
			return -1;
		}
	}

	// free memory
	#pragma patus deallocate_grids

	return 0;
}
