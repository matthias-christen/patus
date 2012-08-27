/**
 * Patus Runtime Library
 */

#ifndef __PATUSRT_H__
#define __PATUSRT_H__

#ifdef __cplusplus
extern "C" {
#endif


/*************************************************************************/
/* Timer Functions                                                       */
/*************************************************************************/

/**
 * Starts the timer.
 */
void tic ();

/**
 * Stops the timer and prints a time value to stdout.
 */
void toc (long nFlopsPerStencil, long nStencilComputationsCount, long nBytesTransferred);


/*************************************************************************/
/* Utility Functions                                                     */
/*************************************************************************/

/**
 * Determines whether the command line argument szArg was passed to the program.
 */
int has_arg (char* szArg, int argc, char** argv);

void write_data_f (char* szFilename, int nDim, float* pData, ...);
void write_data_d (char* szFilename, int nDim, double* pData, ...);


#ifndef __cplusplus
/*
inline int min (int a, int b)
{
	if (a < b)
		return a;
	return b;
}*/
#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
inline int max (int a, int b)
{
	if (a < b)
		return b;
	return a;
}*/
#define max(a, b) (((a) < (b)) ? (b) : (a))
#endif


#ifdef __cplusplus
}
#endif

#endif