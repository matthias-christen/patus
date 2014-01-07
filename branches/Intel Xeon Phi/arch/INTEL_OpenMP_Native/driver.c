/**
 * DISCLAIMER:
 * The generated code was not tested on actual Xeon Phi (MIC) devices and may be erroneous.
 * For now, only native (no-offload) executables are built, which are to be run directly on the co-processor.
 */
 
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
	//#pragma offload target(mic) in(<grid>:length(<grid_size>) alloc_if(1) free_if(0))
	#pragma patus offload_mic_allocate
	#pragma omp parallel
	{
		#pragma patus initialize_grids
	}

	long nFlopsPerStencil = PATUS_FLOPS_PER_STENCIL;
	long nGridPointsCount = 5 * PATUS_GRID_POINTS_COUNT;
	long nBytesTransferred = 5 * PATUS_BYTES_TRANSFERRED;

	// warm up
	//#pragma offload target(mic) nocopy(<grid>:length(<grid_size>) alloc_if(0) free_if(0))
	#pragma patus offload_mic
	#pragma omp parallel
	{
		#pragma patus compute_stencil
	}
	
	// run the benchmark
	tic ();
	//#pragma offload target(mic) nocopy(<grid>:length(<grid_size>) alloc_if(0) free_if(0))
	#pragma patus offload_mic
	#pragma omp parallel private(i)
	for (i = 0; i < 5; i++)
	{
		#pragma patus compute_stencil
		#pragma omp barrier
	}
	toc (nFlopsPerStencil, nGridPointsCount, nBytesTransferred);
	
	// validate
	if (PATUS_DO_VALIDATION)
	{
		//#pragma offload target(mic) out(<grid>:length(<grid_size>) alloc_if(0) free_if(0))
		#pragma patus offload_mic_copyback
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
			#pragma patus deallocate_mic_grids
			#pragma patus deallocate_grids
			puts ("Validation failed.");
			return -1;
		}
	}	

	// free memory
	//#pragma offload target(mic) out(<grid>:length(<grid_size>) alloc_if(0) free_if(1))
	#pragma patus deallocate_mic_grids
	{}
	
	#pragma patus deallocate_grids

	return 0;
}
