#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#define X_MAX 8
#define Y_MAX 8
#define Z_MAX 8


#define __device__
#define __global__

#define min(a,b) ((a)<(b)?(a):(b))

typedef struct
{
    int x;
    int y;
    int z;
} dim3;


dim3 blockDim = { 4, 4, 4 };

dim3 gridDim = { 1, 1, 1 };
dim3 threadIdx;    
dim3 blockIdx;


/******************************************************************************/

__device__ void laplacian___unroll_p3_10101(float *  *  u_0_1_out, float *  u_0_0, float *  u_0_1, float alpha, float beta, int x_max, int y_max, int z_max, int cb_x, int cb_y, int cb_z, int chunk)
{
	int _idx0;
	int _idx1;
	int _idx2;
	int _idx3;
	int _idx4;
	int _idx5;
	int _idx6;
	int idx_1_1;
	int idx_1_2;
	int p3_idx_x;
	int p3_idx_y;
	int p3_idx_z;
	int stride_1;
	int tmp;
	int v2_idx_x;
	int v2_idx_x_max;
	int v2_idx_y;
	int v2_idx_y_max;
	int v2_idx_z;
	int v2_idx_z_max;
	/*
	Initializations
	*/
	stride_1=((int)(((blockDim.y+((int)(((y_max+cb_y)-1)/cb_y)))-1)/blockDim.y));
	tmp=blockIdx.y;
	idx_1_2=(tmp/stride_1);
	tmp-=(idx_1_2*stride_1);
	idx_1_1=tmp;
	/*
	Implementation
	*/
	v2_idx_z=(1+(((blockDim.z*idx_1_2)+threadIdx.z)*cb_z));
	v2_idx_z_max=(v2_idx_z+cb_z);
	v2_idx_y=(1+(((blockDim.y*idx_1_1)+threadIdx.y)*cb_y));
	v2_idx_y_max=(v2_idx_y+cb_y);
	
	printf ("(?..?), (%d..%d), (%d..%d)\n", v2_idx_y, v2_idx_y_max-1, v2_idx_z, v2_idx_z_max-1);
	
	/*
	for v2_idx_x = (1+(((blockDim.x*blockIdx.x)+threadIdx.x)*(cb_x*chunk)))..((1+(((blockDim.x*blockIdx.x)+threadIdx.x)*(cb_x*chunk)))+(chunk-1)) by 1 parallel 1 <level 1> schedule 1 { ... }
	*/
	for (v2_idx_x=(1+(((blockDim.x*blockIdx.x)+threadIdx.x)*(cb_x*chunk))); v2_idx_x<=((1+(((blockDim.x*blockIdx.x)+threadIdx.x)*(cb_x*chunk)))+(cb_x*chunk-1)); v2_idx_x+=cb_x)
	{
		v2_idx_x_max=(v2_idx_x+(cb_x));
		if ((((v2_idx_x<=x_max)&&(v2_idx_y<=y_max))&&(v2_idx_z<=z_max)))
		{
			/* Index bounds calculations for iterators in v2[t=t][0] */
			/*
			for POINT p3[t=t][0] of size [1, 1, 1] in v2[t=t][0] + [ min=[0, 0, 0], max=[0, 0, 0] ] parallel 1 <level 1> schedule default { ... }
			*/
			printf ("(%d..%d), (%d..%d), (%d..%d)\n", v2_idx_x, v2_idx_x_max-1, v2_idx_y, v2_idx_y_max-1, v2_idx_z, v2_idx_z_max-1);
			{
				/* Index bounds calculations for iterators in p3[t=t][0] */
				for (p3_idx_z=v2_idx_z; p3_idx_z<v2_idx_z_max; p3_idx_z+=1)
				{
					for (p3_idx_y=v2_idx_y; p3_idx_y<v2_idx_y_max; p3_idx_y+=1)
					{
						for (p3_idx_x=v2_idx_x; p3_idx_x<v2_idx_x_max; p3_idx_x+=1)
						{
							/* Index bounds calculations for iterators in p3[t=t][0] */
							/*
							p3[t=(t+1)][0]=stencil(p3[t=t][0])
							*/
							/* _idx0 = (((x_max+2)*(((y_max+2)*p3_idx_z)+p3_idx_y))+p3_idx_x) */
							_idx0=(((x_max+2)*(((y_max+2)*p3_idx_z)+p3_idx_y))+p3_idx_x);
							/* _idx1 = (((x_max+2)*(((y_max+2)*p3_idx_z)+p3_idx_y))+(p3_idx_x+1)) */
							_idx1=(_idx0+1);
							/* _idx2 = (((x_max+2)*(((y_max+2)*p3_idx_z)+p3_idx_y))+(p3_idx_x-1)) */
							_idx2=(_idx0-1);
							/* _idx3 = (((x_max+2)*(((y_max+2)*p3_idx_z)+(p3_idx_y+1)))+p3_idx_x) */
							_idx3=((_idx2+x_max)+3);
							/* _idx4 = (((x_max+2)*(((y_max+2)*p3_idx_z)+(p3_idx_y-1)))+p3_idx_x) */
							_idx4=((_idx2-x_max)-1);
							/* _idx5 = (((x_max+2)*(((y_max+2)*(p3_idx_z+1))+p3_idx_y))+p3_idx_x) */
							_idx5=(((_idx3+((x_max+2)*y_max))+x_max)+2);
							/* _idx6 = (((x_max+2)*(((y_max+2)*(p3_idx_z-1))+p3_idx_y))+p3_idx_x) */
							_idx6=(((_idx3+((( - x_max)-2)*y_max))-(3*x_max))-6);
							//u_0_1[_idx0]=((alpha*u_0_0[_idx0])+(beta*((u_0_0[_idx1]+(u_0_0[_idx2]+u_0_0[_idx3]))+(u_0_0[_idx4]+(u_0_0[_idx5]+u_0_0[_idx6])))));
							printf ("STENCIL: %d %d %d %d %d %d %d\n", _idx0, _idx1, _idx2, _idx3, _idx4, _idx5, _idx6);
						}
					}
				}
			}
		}
	}
//	( * u_0_1_out)=u_0_1;
}

#if 0
__device__ void laplacian___unroll_p3_10101(float *  *  u_0_1_out, float *  u_0_0, float *  u_0_1, float alpha, float beta, int x_max, int y_max, int z_max, int cb_x, int cb_y, int cb_z, int chunk)
{
	int _idx0;
	int _idx1;
	int _idx2;
	int _idx3;
	int _idx4;
	int _idx5;
	int _idx6;
	int end0;
	int end1;
	int end2;
	int numthds0;
	int numthds1;
	int p3_idx_x;
	int p3_idx_y;
	int p3_idx_z;
	int start0;
	int start1;
	int start2;
	int stepouter0;
	int stepouter1;
	int stepouter2;
	int v2_blkidx_x;
	int v2_blkidx_x_idxouter;
	int v2_blkidx_y;
	int v2_blkidx_z;
	int v2_idx_x;
	int v2_idx_x_max;
	int v2_idx_y;
	int v2_idx_y_max;
	int v2_idx_z;
	int v2_idx_z_max;
	/*
	Initializations
	*/
	/*
	Implementation
	*/
	start0=((threadIdx.x+(blockDim.x*blockIdx.x))*cb_x);
	end0=(x_max-1);
	numthds0=(blockDim.x*gridDim.x);
	stepouter0=((cb_x*chunk)*numthds0);
	start1=((threadIdx.y+(blockDim.y*blockIdx.y))*cb_y);
	end1=(y_max-1);
	numthds1=(blockDim.y*gridDim.y);
	stepouter1=(cb_y*numthds1);
	start2=(threadIdx.z*cb_z);
	end2=(z_max-1);
	stepouter2=(cb_z*blockDim.z);
	
	printf ("start0=%d, end0=%d, numthds0=%d, stepouter0=%d, start1=%d, end1=%d, numthds1=%d, stepouter1=%d, end2=%d, stepouter2=%d\n", start0, end0, numthds0, stepouter0, start1, end1, numthds1, stepouter1, end2, stepouter2);

	/*
	for v2_blkidx_z = start2..end2 by stepouter2 parallel 1 <level 1> schedule 1 { ... }
	*/
	for (v2_blkidx_z=start2; v2_blkidx_z<=end2; v2_blkidx_z+=stepouter2)
	{
		v2_idx_z=v2_blkidx_z;
		v2_idx_z=(v2_idx_z+1);
		v2_idx_z_max=min((v2_idx_z+cb_z), (z_max+1));
		/*
		for v2_blkidx_y = start1..end1 by stepouter1 parallel 1 <level 1> schedule 1 { ... }
		*/
		for (v2_blkidx_y=start1; v2_blkidx_y<=end1; v2_blkidx_y+=stepouter1)
		{
			v2_idx_y=v2_blkidx_y;
			v2_idx_y=(v2_idx_y+1);
			v2_idx_y_max=min((v2_idx_y+cb_y), (y_max+1));
			/*
			for v2_blkidx_x_idxouter = (start0*chunk)..end0 by stepouter0 parallel 1 <level 1> schedule 1 { ... }
			*/
			for (v2_blkidx_x_idxouter=(start0*chunk); v2_blkidx_x_idxouter<=end0; v2_blkidx_x_idxouter+=stepouter0)
			{
				/*
				for v2_blkidx_x = v2_blkidx_x_idxouter..min(end0, (v2_blkidx_x_idxouter+(chunk-1))) by cb_x parallel 1 <level 1> schedule 1 { ... }
				*/
				for (v2_blkidx_x=v2_blkidx_x_idxouter; v2_blkidx_x<=min(end0, (v2_blkidx_x_idxouter+(chunk-1))); v2_blkidx_x+=cb_x)
				{
					v2_idx_x=v2_blkidx_x;
					v2_idx_x=(v2_idx_x+1);
					v2_idx_x_max=min((v2_idx_x+cb_x), (x_max+1));
					/* Index bounds calculations for iterators in v2[t=t][0] */
					/*
					for POINT p3[t=t][0] of size [1, 1, 1] in v2[t=t][0] + [ min=[0, 0, 0], max=[0, 0, 0] ] parallel 1 <level 1> schedule default { ... }
					*/
					{
    					printf ("(%d..%d), (%d..%d), (%d..%d)\n", v2_idx_x, v2_idx_x_max-1, v2_idx_y, v2_idx_y_max-1, v2_idx_z, v2_idx_z_max-1);

                        /* Index bounds calculations for iterators in p3[t=t][0] */
						for (p3_idx_z=v2_idx_z; p3_idx_z<v2_idx_z_max; p3_idx_z+=1)
						{
							for (p3_idx_y=v2_idx_y; p3_idx_y<v2_idx_y_max; p3_idx_y+=1)
							{
								for (p3_idx_x=v2_idx_x; p3_idx_x<v2_idx_x_max; p3_idx_x+=1)
								{
									/* Index bounds calculations for iterators in p3[t=t][0] */
									/*
									p3[t=(t+1)][0]=stencil(p3[t=t][0])
									*/
									/* _idx0 = (((x_max+2)*(((y_max+2)*p3_idx_z)+p3_idx_y))+p3_idx_x) */
									_idx0=(((x_max+2)*(((y_max+2)*p3_idx_z)+p3_idx_y))+p3_idx_x);
									/* _idx1 = (((x_max+2)*(((y_max+2)*p3_idx_z)+p3_idx_y))+(p3_idx_x+1)) */
									_idx1=(_idx0+1);
									/* _idx2 = (((x_max+2)*(((y_max+2)*p3_idx_z)+p3_idx_y))+(p3_idx_x-1)) */
									_idx2=(_idx0-1);
									/* _idx3 = (((x_max+2)*(((y_max+2)*p3_idx_z)+(p3_idx_y+1)))+p3_idx_x) */
									_idx3=((_idx2+x_max)+3);
									/* _idx4 = (((x_max+2)*(((y_max+2)*p3_idx_z)+(p3_idx_y-1)))+p3_idx_x) */
									_idx4=((_idx2-x_max)-1);
									/* _idx5 = (((x_max+2)*(((y_max+2)*(p3_idx_z+1))+p3_idx_y))+p3_idx_x) */
									_idx5=(((_idx3+((x_max+2)*y_max))+x_max)+2);
									/* _idx6 = (((x_max+2)*(((y_max+2)*(p3_idx_z-1))+p3_idx_y))+p3_idx_x) */
									_idx6=(((_idx3+((( - x_max)-2)*y_max))-(3*x_max))-6);
//									u_0_1[_idx0]=((alpha*u_0_0[_idx0])+(beta*((u_0_0[_idx1]+(u_0_0[_idx2]+u_0_0[_idx3]))+(u_0_0[_idx4]+(u_0_0[_idx5]+u_0_0[_idx6])))));
									printf ("STENCIL: %d %d %d %d %d %d %d\n", _idx0, _idx1, _idx2, _idx3, _idx4, _idx5, _idx6);
								}
							}
						}
					}
				}
			}
		}
	}
//	( * u_0_1_out)=u_0_1;
}
#endif


/******************************************************************************/


int main (int argc, char** argv)
{
    int cb_x = 2;
    int cb_y = 2;
    int cb_z = 2;
    int chunk_x = 2;
    
    
    gridDim.x = (int) ceil ((double) X_MAX / (double) (cb_x * chunk_x * blockDim.x));
    gridDim.y = (int) (ceil ((double) Y_MAX / (double) (cb_y * blockDim.y)) * ceil ((double) Z_MAX / (double) (cb_z * blockDim.z)));
    
    printf ("gridDim=(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
    printf ("blockDim=(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    
    
    for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++)
    {
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++)
        {
            for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++)
            {
                printf ("!!! blk (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
                for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++)
                {
                    for (threadIdx.y = 0; threadIdx.y < blockDim.y; threadIdx.y++)
                    {
                        for (threadIdx.x = 0; threadIdx.x < blockDim.x; threadIdx.x++)
                        {
                            printf ("/// blk (%d, %d, %d), thd (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
                            laplacian___unroll_p3_10101(NULL, NULL, NULL, 0.1f, 0.2f, X_MAX, Y_MAX, Z_MAX, cb_x, cb_y, cb_z, chunk_x);
                        }
                        printf ("\n");
                    }
                }
                printf ("\n");
            }
        }
    }
    
    return 0;
}
