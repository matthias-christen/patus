#ifndef __MIC_MATH_LIB_H__
#define __MIC_MATH_LIB_H__

#include <zmmintrin.h>


#ifdef _MSC_VER /* Visual C++ */
#   define MIC_ALIGN64_BEG __declspec(align(64))
#   define MIC_ALIGN64_END 
#else           /* gcc or icc */
#   define MIC_ALIGN64_BEG
#   define MIC_ALIGN64_END __attribute__((aligned(64)))
#endif

#define MIC_DECLARE_INT_CONST(Name, Val) \
    static const MIC_ALIGN64_BEG int Name##_PI32_[16] MIC_ALIGN64_END = { Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val };
#define MIC_DECLARE_LONG_CONST(Name, Val1, Val2) \
    static const MIC_ALIGN64_BEG int Name##_PI32_[16] MIC_ALIGN64_END = { Val1, Val2, Val1, Val2, Val1, Val2, Val1, Val2, Val1, Val2, Val1, Val2, Val1, Val2, Val1, Val2 };

MIC_DECLARE_INT_CONST(MIC_SIGN_MASK_SINGLE, 0x80000000)
#define MIC_SIGN_MASK_SINGLE *(__m512i*) MIC_SIGN_MASK_SINGLE_PI32_
MIC_DECLARE_LONG_CONST(MIC_SIGN_MASK_DOUBLE, 0x00000000, 0x80000000)
#define MIC_SIGN_MASK_DOUBLE *(__m512i*) MIC_SIGN_MASK_DOUBLE_PI32_

#define _mm512_neg_ps(x) _mm512_xor_epi32 ((__m512i) x, MIC_SIGN_MASK_SINGLE)
#define _mm512_neg_pd(x) _mm512_xor_epi32 ((__m512i) x, MIC_SIGN_MASK_DOUBLE)

#endif
