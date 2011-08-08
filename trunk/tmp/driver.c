#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>


#include <omp.h>
#include "patusrt.h"

// forward_decls -->
void initialize_laplacian(float *  u_0_0, float *  u_0_1, float alpha, float beta, int x_max, int y_max, int z_max, int cb_x, int cb_y, int cb_z, int chunk);
void laplacian(float *  *  u_0_1_out, float *  u_0_0, float *  u_0_1, float alpha, float beta, int x_max, int y_max, int z_max, int cb_x, int cb_y, int cb_z, int chunk, int _unroll_p3);

// <--


int main (int argc, char** argv)
{
	int i;
	
	// prepare grids
	// declare_grids -->
	float *  u_0_1_out;
	float *  u_0_1_out_ref;
	float *  u_0_0;
	float *  u_0_0_ref;
	float *  u_0_1;
	float *  u_0_1_ref;
	if ((argc!=9))
	{
		printf("Wrong number of parameters. Syntax:\n%s <x_max> <y_max> <z_max> <cb_x> <cb_y> <cb_z> <chunk> <_unroll_p3>\n", argv[0]);
		exit(-1);
	}
	int x_max = atoi(argv[1]);
	int y_max = atoi(argv[2]);
	int z_max = atoi(argv[3]);
	int cb_x = atoi(argv[4]);
	int cb_y = atoi(argv[5]);
	int cb_z = atoi(argv[6]);
	int chunk = atoi(argv[7]);
	int _unroll_p3 = atoi(argv[8]);
	// <--
	
	// allocate_grids -->
	u_0_0=((float * )malloc(((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float))));
	u_0_0_ref=((float * )malloc(((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float))));
	u_0_1=((float * )malloc(((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float))));
	u_0_1_ref=((float * )malloc(((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float))));
	// <--
	
	
	// initialize
#pragma omp parallel
	{
		// initialize_grids -->
		initialize_laplacian(u_0_0, u_0_1, 0.1, 0.2, x_max, y_max, z_max, cb_x, cb_y, cb_z, chunk);
		initialize_laplacian(u_0_0_ref, u_0_1_ref, 0.1, 0.2, x_max, y_max, z_max, cb_x, cb_y, cb_z, chunk);
		// <--
		
	}
	
	long nFlopsPerStencil = 8;
	long nGridPointsCount = 5 * ((x_max*y_max)*z_max);
	long nBytesTransferred = 5 * (1*(((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float))+((((x_max+2)*(y_max+2))*(z_max+2))*sizeof (float))));
	
	// warm up
#pragma omp parallel
	{
		// compute_stencil -->
		laplacian(( & u_0_1_out), u_0_0, u_0_1, 0.1, 0.2, x_max, y_max, z_max, cb_x, cb_y, cb_z, chunk, _unroll_p3);
		// <--
		
	}
	
	// run the benchmark
	tic ();
#pragma omp parallel private(i)
	for (i = 0; i < 5; i++)
	{
		// compute_stencil -->
		laplacian(( & u_0_1_out), u_0_0, u_0_1, 0.1, 0.2, x_max, y_max, z_max, cb_x, cb_y, cb_z, chunk, _unroll_p3);
		// <--
		
#pragma omp barrier
	}
	toc (nFlopsPerStencil, nGridPointsCount, nBytesTransferred);
	
	// validate
	if (1)
	{
#pragma omp parallel
		{
			// initialize_grids -->
			initialize_laplacian(u_0_0, u_0_1, 0.1, 0.2, x_max, y_max, z_max, cb_x, cb_y, cb_z, chunk);
			initialize_laplacian(u_0_0_ref, u_0_1_ref, 0.1, 0.2, x_max, y_max, z_max, cb_x, cb_y, cb_z, chunk);
			// <--
			
#pragma omp barrier
			// compute_stencil -->
			laplacian(( & u_0_1_out), u_0_0, u_0_1, 0.1, 0.2, x_max, y_max, z_max, cb_x, cb_y, cb_z, chunk, _unroll_p3);
			// <--
			
		}
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
			free(u_0_0);
			free(u_0_0_ref);
			free(u_0_1);
			free(u_0_1_ref);
			// <--
			
			puts ("Validation failed.");
			return -1;
		}
	}
	
	// free memory
	// deallocate_grids -->
	free(u_0_0);
	free(u_0_0_ref);
	free(u_0_1);
	free(u_0_1_ref);
	// <--
	
	
	return 0;
}
