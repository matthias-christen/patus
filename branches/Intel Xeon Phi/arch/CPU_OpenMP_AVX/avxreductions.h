#ifndef __AVX_REDUCTIONS_H__
#define __AVX_REDUCTIONS_H__

#include "immintrin.h"

/**
 * Single Precision
 */
 
inline float vec_reduce_sum_ps (__m256 v)
{
	__m256 ones = { 1, 1, 1, 1, 1, 1, 1, 1 };
	__m256 r = _mm256_dp_ps (v, ones, 0xf1);
	r = _mm256_add_ps (r, _mm256_permute2f128_ps (r, r, 0x81));	
	return *((float*) &r);
}

inline float vec_reduce_product_ps (__m256 v)
{
	__m256 r = _mm256_mul_ps (v, _mm256_permute2f128_ps (v, v, 0x81));
	r = _mm256_mul_ps (r, _mm256_shuffle_ps (r, r, 0x0e));
	r = _mm256_mul_ps (r, _mm256_shuffle_ps (r, r, 0x01));
	return *((float*) &r);
}

inline float vec_reduce_min_ps (__m256 v)
{
	__m256 r = _mm256_min_ps (v, _mm256_permute2f128_ps (v, v, 0x81));
	r = _mm256_min_ps (r, _mm256_shuffle_ps (r, r, 0x0e));
	r = _mm256_min_ps (r, _mm256_shuffle_ps (r, r, 0x01));
	return *((float*) &r);
}

inline float vec_reduce_max_ps (__m256 v)
{
	__m256 r = _mm256_max_ps (v, _mm256_permute2f128_ps (v, v, 0x81));
	r = _mm256_max_ps (r, _mm256_shuffle_ps (r, r, 0x0e));
	r = _mm256_max_ps (r, _mm256_shuffle_ps (r, r, 0x01));
	return *((float*) &r);
}


/**
 * Double Precision
 */

inline double vec_reduce_sum_pd (__m256d v)
{
	__m256d q = _mm256_add_pd (v, _mm256_permute2f128_pd (v, v, 0x81));
	q = _mm256_add_pd (q, _mm256_shuffle_pd (q, q, 0x01));	
	return *((double*) &q);
}

inline float vec_reduce_product_ps (__m256d v)
{
	__m256d q = _mm256_mul_pd (v, _mm256_permute2f128_pd (v, v, 0x81));
	q = _mm256_mul_pd (q, _mm256_shuffle_pd (q, q, 0x01));	
	return *((double*) &q);
}

inline float vec_reduce_min_ps (__m256d v)
{
	__m256d q = _mm256_min_pd (v, _mm256_permute2f128_pd (v, v, 0x81));
	q = _mm256_min_pd (q, _mm256_shuffle_pd (q, q, 0x01));	
	return *((double*) &q);
}

inline float vec_reduce_max_ps (__m256d v)
{
	__m256d q = _mm256_max_pd (v, _mm256_permute2f128_pd (v, v, 0x81));
	q = _mm256_max_pd (q, _mm256_shuffle_pd (q, q, 0x01));	
	return *((double*) &q);
}

#endif
