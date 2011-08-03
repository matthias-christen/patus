/**
 * Patus Runtime Library
 */

#ifndef __PATUSRT_H__
#define __PATUSRT_H__

//#ifdef __cplusplus
//extern "C" {
//#endif


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


#ifndef __cplusplus
inline int min (int a, int b)
{
	if (a < b)
		return a;
	return b;
}

inline int max (int a, int b)
{
	if (a < b)
		return b;
	return a;
}
#endif


//#ifdef __cplusplus
//}
//#endif

#endif
