#ifndef __AVX_MATH_LIB_H__
#define __AVX_MATH_LIB_H__

#include "immintrin.h"


#ifdef _MSC_VER /* Visual C++ */
#   define AVX_ALIGN32_BEG __declspec(align(32))
#   define AVX_ALIGN32_END 
#else           /* gcc or icc */
#   define AVX_ALIGN32_BEG
#   define AVX_ALIGN32_END __attribute__((aligned(32)))
#endif

#define AVX_DECLARE_INT_CONST(Name, Val) \
    static const AVX_ALIGN32_BEG int Name##_PI32_[8] AVX_ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val };
#define AVX_DECLARE_LONG_CONST(Name, Val1, Val2) \
    static const AVX_ALIGN32_BEG int Name##_PI32_[8] AVX_ALIGN32_END = { Val1, Val2, Val1, Val2, Val1, Val2, Val1, Val2 };

AVX_DECLARE_INT_CONST(AVX_SIGN_MASK_SINGLE, 0x80000000)
#define AVX_SIGN_MASK_SINGLE *(__m256i*) AVX_SIGN_MASK_SINGLE_PI32_
AVX_DECLARE_LONG_CONST(AVX_SIGN_MASK_DOUBLE, 0x00000000, 0x80000000)
#define AVX_SIGN_MASK_DOUBLE *(__m256i*) AVX_SIGN_MASK_DOUBLE_PI32_

#define _mm256_load1_ps(x) _mm256_set_ps(x, x, x, x, x, x, x, x)
#define _mm256_load1_pd(x) _mm256_set_pd(x, x, x, x)

#define _mm256_neg_ps(x) _mm256_xor_ps (x, (__m256) AVX_SIGN_MASK_SINGLE)
#define _mm256_neg_pd(x) _mm256_xor_pd (x, (__m256d) AVX_SIGN_MASK_DOUBLE)

#define clamp0(x) ((x) > 0) ? 0 : (x)
#define _mm256_clamp0_ps(x) _mm256_and_ps (x, _mm256_cmp_ps (x, _mm256_xor_ps (x, x), 1))
#define _mm256_clamp0_pd(x) _mm256_and_pd (x, _mm256_cmp_pd (x, _mm256_xor_pd (x, x), 1))

#define clamp0if(value, threshold) ((threshold) > 0) ? 0 : (value)
#define _mm256_clamp0if_ps(value, threshold) _mm256_and_ps (value, _mm256_cmp_ps (threshold, _mm256_xor_ps (value, value), 1))
#define _mm256_clamp0if_pd(value, threshold) _mm256_and_pd (value, _mm256_cmp_pd (threshold, _mm256_xor_pd (value, value), 1))

#endif
