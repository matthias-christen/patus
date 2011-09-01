#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>


#include <omp.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include "patusrt.h"

// forward_decls -->
void initialize_game_of_life(float *  u_0_0, float *  u_0_1, int width, int height, int cb_x, int cb_y, int chunk);
void game_of_life(float *  *  u_0_1_out, float *  u_0_0, float *  u_0_1, int width, int height, int cb_x, int cb_y, int chunk, int _unroll_p3);

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
	if ((argc!=7))
	{
		printf("Wrong number of parameters. Syntax:\n%s <width> <height> <cb_x> <cb_y> <chunk> <_unroll_p3>\n", argv[0]);
		exit(-1);
	}
	int width = atoi(argv[1]);
	int height = atoi(argv[2]);
	int cb_x = atoi(argv[3]);
	int cb_y = atoi(argv[4]);
	int chunk = atoi(argv[5]);
	int _unroll_p3 = atoi(argv[6]);
	// <--
	
	// allocate_grids -->
	if ((((width+2)%4)!=0))
	{
		printf("Non-native SIMD type mode requires that (width+2) is divisible by 4 [(width+2) = %d].\n", (width+2));
		return -1;
	}
	u_0_0=((float * )malloc(((((width+2)*(height+2))*sizeof (float))+15)));
	u_0_0_ref=((float * )malloc(((((width+2)*(height+2))*sizeof (float))+15)));
	if ((((width+2)%4)!=0))
	{
		printf("Non-native SIMD type mode requires that (width+2) is divisible by 4 [(width+2) = %d].\n", (width+2));
		return -1;
	}
	u_0_1=((float * )malloc(((((width+2)*(height+2))*sizeof (float))+15)));
	u_0_1_ref=((float * )malloc(((((width+2)*(height+2))*sizeof (float))+15)));
	// <--
	
	
	// initialize
#pragma omp parallel
	{
		// initialize_grids -->
		initialize_game_of_life(((float * )((((uintptr_t)u_0_0)+15)&( ~ ((uintptr_t)15)))), ((float * )((((uintptr_t)u_0_1)+15)&( ~ ((uintptr_t)15)))), width, height, cb_x, cb_y, chunk);
		initialize_game_of_life(u_0_0_ref, u_0_1_ref, width, height, cb_x, cb_y, chunk);
		// <--
		
	}
	
	long nFlopsPerStencil = 14;
	long nGridPointsCount = 5 * (height*width);
	long nBytesTransferred = 5 * (1*((((width+2)*(height+2))*sizeof (float))+(((width+2)*(height+2))*sizeof (float))));
	
	// warm up
#pragma omp parallel
	{
		// compute_stencil -->
		game_of_life(( & u_0_1_out), ((float * )((((uintptr_t)u_0_0)+15)&( ~ ((uintptr_t)15)))), ((float * )((((uintptr_t)u_0_1)+15)&( ~ ((uintptr_t)15)))), width, height, cb_x, cb_y, chunk, _unroll_p3);
		// <--
		
	}
	
	// run the benchmark
	tic ();
#pragma omp parallel private(i)
	for (i = 0; i < 5; i++)
	{
		// compute_stencil -->
		game_of_life(( & u_0_1_out), ((float * )((((uintptr_t)u_0_0)+15)&( ~ ((uintptr_t)15)))), ((float * )((((uintptr_t)u_0_1)+15)&( ~ ((uintptr_t)15)))), width, height, cb_x, cb_y, chunk, _unroll_p3);
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
			initialize_game_of_life(((float * )((((uintptr_t)u_0_0)+15)&( ~ ((uintptr_t)15)))), ((float * )((((uintptr_t)u_0_1)+15)&( ~ ((uintptr_t)15)))), width, height, cb_x, cb_y, chunk);
			initialize_game_of_life(u_0_0_ref, u_0_1_ref, width, height, cb_x, cb_y, chunk);
			// <--
			
#pragma omp barrier
			// compute_stencil -->
			game_of_life(( & u_0_1_out), ((float * )((((uintptr_t)u_0_0)+15)&( ~ ((uintptr_t)15)))), ((float * )((((uintptr_t)u_0_1)+15)&( ~ ((uintptr_t)15)))), width, height, cb_x, cb_y, chunk, _unroll_p3);
			// <--
			
		}
		// validate_computation -->
		int bHasErrors = 0;
		float L;
		int _idx10;
		int _idx11;
		int _idx12;
		int _idx13;
		int _idx14;
		int _idx15;
		int _idx16;
		int _idx17;
		int _idx9;
		int pt_ref_idx_x;
		int pt_ref_idx_y;
		int t_ref;
		float *  tmp_swap_0;
		{
			/*
			for t_ref = 1..1 by 1 parallel 1 <level 0> schedule 1 { ... }
			*/
			for (t_ref=1; t_ref<=1; t_ref+=1)
			{
				/*
				for POINT pt_ref[t=t][0] of size [1, 1] in u0[t=t][0] + [ min=[0, 0], max=[0, 0] ] parallel 1 <level 0> schedule default { ... }
				*/
				{
					/* Index bounds calculations for iterators in pt_ref[t=t][0] */
					for (pt_ref_idx_y=1; pt_ref_idx_y<(height+1); pt_ref_idx_y+=1)
					{
						for (pt_ref_idx_x=1; pt_ref_idx_x<(width+1); pt_ref_idx_x+=1)
						{
							/*
							pt_ref[t=t][0]=stencil(pt_ref[t=t][0])
							*/
							/* _idx9 = (((width+2)*(pt_ref_idx_y-1))+(pt_ref_idx_x-1)) */
							_idx9=(((width+2)*(pt_ref_idx_y-1))+(pt_ref_idx_x-1));
							/* _idx10 = (((width+2)*(pt_ref_idx_y-1))+pt_ref_idx_x) */
							_idx10=(_idx9+1);
							/* _idx11 = (((width+2)*(pt_ref_idx_y-1))+(pt_ref_idx_x+1)) */
							_idx11=(_idx10+1);
							/* _idx12 = (((width+2)*pt_ref_idx_y)+(pt_ref_idx_x-1)) */
							_idx12=(_idx11+width);
							/* _idx13 = (((width+2)*pt_ref_idx_y)+(pt_ref_idx_x+1)) */
							_idx13=((_idx11+width)+2);
							/* _idx14 = (((width+2)*(pt_ref_idx_y+1))+(pt_ref_idx_x-1)) */
							_idx14=(_idx13+width);
							/* _idx15 = (((width+2)*(pt_ref_idx_y+1))+pt_ref_idx_x) */
							_idx15=(_idx14+1);
							/* _idx16 = (((width+2)*(pt_ref_idx_y+1))+(pt_ref_idx_x+1)) */
							_idx16=(_idx14+2);
							L=(((u_0_0_ref[_idx9]+u_0_0_ref[_idx10])+(u_0_0_ref[_idx11]+u_0_0_ref[_idx12]))+((u_0_0_ref[_idx13]+u_0_0_ref[_idx14])+(u_0_0_ref[_idx15]+u_0_0_ref[_idx16])));
							/* _idx17 = (((width+2)*pt_ref_idx_y)+pt_ref_idx_x) */
							_idx17=((_idx11+width)+1);
							u_0_1_ref[_idx17]=(1.0f/((((u_0_0_ref[_idx17]+L)-3.0f)*((L-3.0f)*1.0E20f))+1.0f));
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
		for POINT pt_ref[t=t][0] of size [1, 1] in u0[t=t][0] + [ min=[0, 0], max=[0, 0] ] parallel 1 <level 0> schedule default { ... }
		*/
		{
			/* Index bounds calculations for iterators in pt_ref[t=t][0] */
			for (pt_ref_idx_y=1; pt_ref_idx_y<(height+1); pt_ref_idx_y+=1)
			{
				for (pt_ref_idx_x=1; pt_ref_idx_x<(width+1); pt_ref_idx_x+=1)
				{
					/* _idx9 = (((width+2)*(pt_ref_idx_y-1))+(pt_ref_idx_x-1)) */
					_idx9=(((width+2)*(pt_ref_idx_y-1))+(pt_ref_idx_x-1));
					/* _idx10 = (((width+2)*(pt_ref_idx_y-1))+pt_ref_idx_x) */
					_idx10=(_idx9+1);
					/* _idx11 = (((width+2)*(pt_ref_idx_y-1))+(pt_ref_idx_x+1)) */
					_idx11=(_idx10+1);
					/* _idx17 = (((width+2)*pt_ref_idx_y)+pt_ref_idx_x) */
					_idx17=((_idx11+width)+1);
					if ((fabs(((u_0_1_out[_idx17]-u_0_1_out_ref[_idx17])/u_0_1_out_ref[_idx17]))>1.0E-5))
					{
						bHasErrors=1;
						break;
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
