#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <cutil.h>

typedef uint64_t gpu_ptr_t;

#pragma patus forward_decls

int main (int argc, char** argv)
{
	int i;
	cudaError_t res;

	// prepare grids
	#pragma patus declare_grids
	#pragma patus allocate_grids
	
	#pragma patus declare_GPU_grids
	#pragma patus allocate_GPU_grids
	#pragma patus copy_grids_to_GPU

	#pragma patus initialize_grids
	cudaThreadSynchronize ();
	res = cudaGetLastError ();
	if (res != cudaSuccess)
	{
		printf ("CUDA Error [Initialization]: %s.\n", cudaGetErrorString (res));
		#pragma patus deallocate_grids
		cudaThreadExit ();
		return -1;
	}

	long nFlopsPerStencil = PATUS_FLOPS_PER_STENCIL;
	long nGridPointsCount = 5 * PATUS_GRID_POINTS_COUNT;
	long nBytesTransferred = 5 * PATUS_BYTES_TRANSFERRED;

	// warm up
	#pragma patus compute_stencil
	cudaThreadSynchronize ();
	res = cudaGetLastError ();
	if (res != cudaSuccess)
	{
		printf ("CUDA Error [Stencil]: %s.\n", cudaGetErrorString (res));
		#pragma patus deallocate_grids
		cudaThreadExit ();
		return -1;
	}

	// run the benchmark
	tic ();
	for (i = 0; i < 5; i++)
	{
		#pragma patus compute_stencil
		cudaThreadSynchronize ();
	}
	toc (nFlopsPerStencil, nGridPointsCount, nBytesTransferred);

	// validate
	if (PATUS_DO_VALIDATION)
	{
		#pragma patus initialize_grids
		#pragma patus copy_input_grids_from_GPU_to_reference_grids
		#pragma patus compute_stencil
		#pragma patus copy_output_grids_from_GPU
		#pragma patus validate_computation
		
		if (PATUS_VALIDATES)
			puts ("Validation OK.");
		else
		{
			#pragma patus deallocate_grids
			puts ("Validation failed.");
			cudaThreadExit ();
			return -1;
		}
	}	

	// free memory
	#pragma patus deallocate_grids

	cudaThreadExit ();
	return 0;
}
