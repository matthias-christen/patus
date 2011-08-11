#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <cutil.h>

#include "patusrt.h"

typedef uint64_t gpu_ptr_t;

// forward_decls -->
__global__ void initialize_laplacian(float *  u_0_0, float *  u_0_1, float alpha, float beta, int x_max, int y_max, int z_max, int cb_x, int cb_y, int cb_z, int chunk);
__global__ void laplacian(float *  *  u_0_1_out, float *  u_0_0, float *  u_0_1, float alpha, float beta, int x_max, int y_max, int z_max, int cb_x, int cb_y, int cb_z, int chunk, int _unroll_p3);

// <--


int main (int argc, char** argv)
{
	int i;
	cudaError_t res;
	
	// prepare grids
	// declare_grids -->
	float *  u_0_1_out;
	float *  u_0_1_out_ref;
	float *  u_0_0;
	float *  u_0_0_ref;
	float *  u_0_1;
	float *  u_0_1_ref;
	if ((argc!=12))
	{
		printf("Wrong number of parameters. Syntax:\n%s <thds_x> <thds_y> <thds_z> <x_max> <y_max> <z_max> <cb_x> <cb_y> <cb_z> <chunk> <_unroll_p3>\n", argv[0]);
		exit(-1);
	}
	int thds_x = atoi(argv[1]);
	int thds_y = atoi(argv[2]);
	int thds_z = atoi(argv[3]);
	int x_max = atoi(argv[4]);
	int y_max = atoi(argv[5]);
	int z_max = atoi(argv[6]);
	int cb_x = atoi(argv[7]);
	int cb_y = atoi(argv[8]);
	int cb_z = atoi(argv[9]);
	int chunk = atoi(argv[10]);
	int _unroll_p3 = atoi(argv[11]);
	// <--
	
	// allocate_grids -->
	u_0_0=((float * )malloc(((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float))));
	u_0_0_ref=((float * )malloc(((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float))));
	u_0_1=((float * )malloc(((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float))));
	u_0_1_ref=((float * )malloc(((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float))));
	// <--
	
	
	// declare_GPU_grids -->
	gpu_ptr_t *  u_0_1_out_gpu;
	float *  u_0_0_gpu;
	float *  u_0_1_gpu;
	dim3 thds(thds_x, thds_y, thds_z);
	dim3 blks(((int)(((x_max+(cb_x*thds_x))-1)/(cb_x*thds_x))), (((int)(((y_max+(cb_y*thds_y))-1)/(cb_y*thds_y)))*((int)(((z_max+(cb_z*thds_z))-1)/(cb_z*thds_z)))), 1);
	// <--
	
	// allocate_GPU_grids -->
	cudaMalloc(((void *  * )( & u_0_1_out_gpu)), sizeof (gpu_ptr_t));
	cudaMalloc(((void *  * )( & u_0_0_gpu)), ((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float)));
	cudaMalloc(((void *  * )( & u_0_1_gpu)), ((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float)));
	// <--
	
	// copy_grids_to_GPU -->
	cudaMemcpy(((void * )u_0_0_gpu), ((void * )u_0_0), ((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float)), cudaMemcpyHostToDevice);
	cudaMemcpy(((void * )u_0_1_gpu), ((void * )u_0_1), ((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float)), cudaMemcpyHostToDevice);
	// <--
	
	
	// initialize_grids -->
	initialize_laplacian<<<blks, thds>>>(u_0_0_gpu, u_0_1_gpu, 0.1, 0.2, x_max, y_max, z_max, cb_x, cb_y, cb_z, chunk);
	// <--
	
	cudaThreadSynchronize ();
	res = cudaGetLastError ();
	if (res != cudaSuccess)
	{
		printf ("CUDA Error [Initialization]: %s.\n", cudaGetErrorString (res));
		// deallocate_grids -->
		cudaFree(((void * )u_0_0_gpu));
		cudaFree(((void * )u_0_1_gpu));
		free(u_0_0);
		free(u_0_0_ref);
		free(u_0_1);
		free(u_0_1_ref);
		// <--
		
		cudaThreadExit ();
		return -1;
	}
	
	long nFlopsPerStencil = 8;
	long nGridPointsCount = 5 * ((x_max*y_max)*z_max);
	long nBytesTransferred = 5 * (1*(((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float))+((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float))));
	
	// warm up
	// compute_stencil -->
	laplacian<<<blks, thds>>>(((float *  * )u_0_1_out_gpu), u_0_0_gpu, u_0_1_gpu, 0.1, 0.2, x_max, y_max, z_max, cb_x, cb_y, cb_z, chunk, _unroll_p3);
	// <--
	
	cudaThreadSynchronize ();
	res = cudaGetLastError ();
	if (res != cudaSuccess)
	{
		printf ("CUDA Error [Stencil]: %s.\n", cudaGetErrorString (res));
		// deallocate_grids -->
		cudaFree(((void * )u_0_0_gpu));
		cudaFree(((void * )u_0_1_gpu));
		free(u_0_0);
		free(u_0_0_ref);
		free(u_0_1);
		free(u_0_1_ref);
		// <--
		
		cudaThreadExit ();
		return -1;
	}
	
	// run the benchmark
	tic ();
	for (i = 0; i < 5; i++)
	{
		// compute_stencil -->
		laplacian<<<blks, thds>>>(((float *  * )u_0_1_out_gpu), u_0_0_gpu, u_0_1_gpu, 0.1, 0.2, x_max, y_max, z_max, cb_x, cb_y, cb_z, chunk, _unroll_p3);
		// <--
		
		cudaThreadSynchronize ();
	}
	toc (nFlopsPerStencil, nGridPointsCount, nBytesTransferred);
	
	// validate
	if (1)
	{
		// initialize_grids -->
		initialize_laplacian<<<blks, thds>>>(u_0_0_gpu, u_0_1_gpu, 0.1, 0.2, x_max, y_max, z_max, cb_x, cb_y, cb_z, chunk);
		// <--
		
		// copy_input_grids_from_GPU_to_reference_grids -->
		cudaMemcpy(((void * )u_0_0_ref), ((void * )u_0_0_gpu), ((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float)), cudaMemcpyDeviceToHost);
		cudaMemcpy(((void * )u_0_1_ref), ((void * )u_0_1_gpu), ((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float)), cudaMemcpyDeviceToHost);
		// <--
		
		// compute_stencil -->
		laplacian<<<blks, thds>>>(((float *  * )u_0_1_out_gpu), u_0_0_gpu, u_0_1_gpu, 0.1, 0.2, x_max, y_max, z_max, cb_x, cb_y, cb_z, chunk, _unroll_p3);
		// <--
		
		// copy_output_grids_from_GPU -->
		gpu_ptr_t u_0_1_out_gpu_ptr;
		u_0_1_out=u_0_1;
		cudaMemcpy(((void * )( & u_0_1_out_gpu_ptr)), ((void * )u_0_1_out_gpu), sizeof (gpu_ptr_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(((void * )u_0_1_out), ((void * )u_0_1_out_gpu_ptr), ((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float)), cudaMemcpyDeviceToHost);
		// <--
		
		// validate_computation -->
		int bHasErrors = 0;
		int _idx10;
		int _idx11;
		int _idx12;
		int _idx13;
		int _idx7;
		int _idx8;
		int _idx9;
		int pt_ref_idx_x;
		int pt_ref_idx_y;
		int pt_ref_idx_z;
		int t_ref;
		float *  tmp_swap_0;
		{
			/*
			for t_ref = 1..1 by 1 parallel 1 <level 0> schedule 1 { ... }
			*/
			for (t_ref=1; t_ref<=1; t_ref+=1)
			{
				/*
				for POINT pt_ref[t=t][0] of size [1, 1, 1] in u0[t=t][0] + [ min=[0, 0, 0], max=[0, 0, 0] ] parallel 1 <level 0> schedule default { ... }
				*/
				{
					/* Index bounds calculations for iterators in pt_ref[t=t][0] */
					for (pt_ref_idx_z=1; pt_ref_idx_z<(z_max+1); pt_ref_idx_z+=1)
					{
						for (pt_ref_idx_y=1; pt_ref_idx_y<(y_max+1); pt_ref_idx_y+=1)
						{
							for (pt_ref_idx_x=1; pt_ref_idx_x<(x_max+1); pt_ref_idx_x+=1)
							{
								/*
								pt_ref[t=t][0]=stencil(pt_ref[t=t][0])
								*/
								/* _idx7 = (((x_max+2)*(((y_max+2)*pt_ref_idx_z)+pt_ref_idx_y))+pt_ref_idx_x) */
								_idx7=(((x_max+2)*(((y_max+2)*pt_ref_idx_z)+pt_ref_idx_y))+pt_ref_idx_x);
								/* _idx8 = (((x_max+2)*(((y_max+2)*pt_ref_idx_z)+pt_ref_idx_y))+(pt_ref_idx_x+1)) */
								_idx8=(_idx7+1);
								/* _idx9 = (((x_max+2)*(((y_max+2)*pt_ref_idx_z)+pt_ref_idx_y))+(pt_ref_idx_x-1)) */
								_idx9=(_idx7-1);
								/* _idx10 = (((x_max+2)*(((y_max+2)*pt_ref_idx_z)+(pt_ref_idx_y+1)))+pt_ref_idx_x) */
								_idx10=((_idx9+x_max)+3);
								/* _idx11 = (((x_max+2)*(((y_max+2)*pt_ref_idx_z)+(pt_ref_idx_y-1)))+pt_ref_idx_x) */
								_idx11=((_idx9-x_max)-1);
								/* _idx12 = (((x_max+2)*(((y_max+2)*(pt_ref_idx_z+1))+pt_ref_idx_y))+pt_ref_idx_x) */
								_idx12=(((_idx7+((x_max+2)*y_max))+(2*x_max))+4);
								/* _idx13 = (((x_max+2)*(((y_max+2)*(pt_ref_idx_z-1))+pt_ref_idx_y))+pt_ref_idx_x) */
								_idx13=(((_idx7+((( - x_max)-2)*y_max))-(2*x_max))-4);
								u_0_1_ref[_idx7]=((0.1*u_0_0_ref[_idx7])+(0.2*((u_0_0_ref[_idx8]+(u_0_0_ref[_idx9]+u_0_0_ref[_idx10]))+(u_0_0_ref[_idx11]+(u_0_0_ref[_idx12]+u_0_0_ref[_idx13])))));
							}
						}
					}
				}
				u_0_1_out_ref=u_0_1_ref;
				tmp_swap_0=u_0_0_ref;
				u_0_0_ref=u_0_1_ref;
				u_0_1_ref=tmp_swap_0;
			}
		}
		/*
		for POINT pt_ref[t=t][0] of size [1, 1, 1] in u0[t=t][0] + [ min=[0, 0, 0], max=[0, 0, 0] ] parallel 1 <level 0> schedule default { ... }
		*/
		{
			/* Index bounds calculations for iterators in pt_ref[t=t][0] */
			for (pt_ref_idx_z=1; pt_ref_idx_z<(z_max+1); pt_ref_idx_z+=1)
			{
				for (pt_ref_idx_y=1; pt_ref_idx_y<(y_max+1); pt_ref_idx_y+=1)
				{
					for (pt_ref_idx_x=1; pt_ref_idx_x<(x_max+1); pt_ref_idx_x+=1)
					{
						/* _idx7 = (((x_max+2)*(((y_max+2)*pt_ref_idx_z)+pt_ref_idx_y))+pt_ref_idx_x) */
						_idx7=(((x_max+2)*(((y_max+2)*pt_ref_idx_z)+pt_ref_idx_y))+pt_ref_idx_x);
						if ((fabs(((u_0_1_out[_idx7]-u_0_1_out_ref[_idx7])/u_0_1_out_ref[_idx7]))>1.0E-5))
						{
							bHasErrors=1;
							break;
						}
					}
				}
			}
		}
		// <--
		
		
		if (( ! bHasErrors))
		puts ("Validation OK.");
		else
		{
			// deallocate_grids -->
			cudaFree(((void * )u_0_0_gpu));
			cudaFree(((void * )u_0_1_gpu));
			free(u_0_0);
			free(u_0_0_ref);
			free(u_0_1);
			free(u_0_1_ref);
			// <--
			
			puts ("Validation failed.");
			cudaThreadExit ();
			return -1;
		}
	}
	
	// free memory
	// deallocate_grids -->
	cudaFree(((void * )u_0_0_gpu));
	cudaFree(((void * )u_0_1_gpu));
	free(u_0_0);
	free(u_0_0_ref);
	free(u_0_1);
	free(u_0_1_ref);
	// <--
	
	
	cudaThreadExit ();
	return 0;
}
