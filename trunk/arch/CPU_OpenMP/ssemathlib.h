#ifndef __SSE_MATH_LIB_H__
#define __SSE_MATH_LIB_H__

#include "xmmintrin.h"
#include "emmintrin.h"


#ifdef _MSC_VER /* Visual C++ */
#   define ALIGN16_BEG __declspec(align(16))
#   define ALIGN16_END 
#else           /* gcc or icc */
#   define ALIGN16_BEG
#   define ALIGN16_END __attribute__((aligned(16)))
#endif

#define DECLARE_CONST(Name, Val) \
    static const __m128 Name##_PS = { Val, Val, Val, Val }; \
    static const __m128d Name##_PD = { Val, Val };
#define DECLARE_INT_CONST(Name, Val) \
    static const ALIGN16_BEG int Name##_PI32_[4] ALIGN16_END = { Val, Val, Val, Val };
#define DECLARE_LONG_CONST(Name, Val1, Val2) \
    static const ALIGN16_BEG int Name##_PI32_[4] ALIGN16_END = { Val1, Val2, Val1, Val2 };
#define DECLARE_V4_CONST(Name, Val1, Val2, Val3, Val4) \
    static const ALIGN16_BEG int Name##_PI32_[4] ALIGN16_END = { Val1, Val2, Val3, Val4 };


#define MANTISSA_BITS_SINGLE 23
#define MANTISSA_BITS_DOUBLE 52

DECLARE_INT_CONST(SIGN_MASK_SINGLE, 0x80000000)
#define SIGN_MASK_SINGLE *(__m128i*) SIGN_MASK_SINGLE_PI32_
DECLARE_LONG_CONST(SIGN_MASK_DOUBLE, 0x00000000, 0x80000000)
#define SIGN_MASK_DOUBLE *(__m128i*) SIGN_MASK_DOUBLE_PI32_

DECLARE_INT_CONST(INV_SIGN_MASK_SINGLE, 0x7fffffff)
#define INV_SIGN_MASK_SINGLE *(__m128i*) INV_SIGN_MASK_SINGLE_PI32_
DECLARE_LONG_CONST(INV_SIGN_MASK_DOUBLE, 0xffffffff, 0x7fffffff)
#define INV_SIGN_MASK_DOUBLE *(__m128i*) INV_SIGN_MASK_DOUBLE_PI32_

DECLARE_INT_CONST(MINNORMPOS_SINGLE, 1 << MANTISSA_BITS_SINGLE)
#define MINNORMPOS_SINGLE *(__m128i*) MINNORMPOS_SINGLE_PI32_
DECLARE_LONG_CONST(MINNORMPOS_DOUBLE, 0x00000000, 1 << (MANTISSA_BITS_DOUBLE - 32))
#define MINNORMPOS_DOUBLE *(__m128i*) MINNORMPOS_DOUBLE_PI32_

DECLARE_INT_CONST(MANTISSAMASK_SINGLE, (0xffffffff << MANTISSA_BITS_SINGLE) & 0x7fffffff)
#define MANTISSAMASK_SINGLE *(__m128i*) MANTISSAMASK_SINGLE_PI32_
DECLARE_LONG_CONST(MANTISSAMASK_DOUBLE, 0x00000000, (0xffffffff << (MANTISSA_BITS_DOUBLE - 32)) & 0x7fffffff)
#define MANTISSAMASK_DOUBLE *(__m128i*) MANTISSAMASK_DOUBLE_PI32_

DECLARE_INT_CONST(INVMANTISSAMASK_SINGLE, ~((0xffffffff << MANTISSA_BITS_SINGLE) & 0x7fffffff))
#define INVMANTISSAMASK_SINGLE *(__m128i*) INVMANTISSAMASK_SINGLE_PI32_
DECLARE_LONG_CONST(INVMANTISSAMASK_DOUBLE, 0xffffffff, ~((0xffffffff << (MANTISSA_BITS_DOUBLE - 32)) & 0x7fffffff))
#define INVMANTISSAMASK_DOUBLE *(__m128i*) INVMANTISSAMASK_DOUBLE_PI32_

// exponent bias in single precision (127)
DECLARE_INT_CONST(EXPBIAS_SINGLE, 0x0000007f)
#define EXPBIAS_SINGLE *(__m128i*) EXPBIAS_SINGLE_PI32_

// exponent bias in double precision (1023)
DECLARE_LONG_CONST(EXPBIAS_DOUBLE, 0x000003ff, 0x00000000)
#define EXPBIAS_DOUBLE *(__m128i*) EXPBIAS_DOUBLE_PI32_

DECLARE_INT_CONST(ONE, 0x00000001)
#define ONE_PI32 *(__m128i*) ONE_PI32_
DECLARE_INT_CONST(INVONE, ~0x00000001)
#define INVONE_PI32 *(__m128i*) INVONE_PI32_
DECLARE_INT_CONST(TWO, 0x00000002)
#define TWO_PI32 *(__m128i*) TWO_PI32_
DECLARE_INT_CONST(FOUR, 0x00000004)
#define FOUR_PI32 *(__m128i*) FOUR_PI32_

DECLARE_CONST(ONE, 1.0)
DECLARE_CONST(NEGONE, -1.0)
DECLARE_CONST(TWO, 2.0)
DECLARE_CONST(HALF, 0.5)
DECLARE_CONST(PI, 3.1415926535897932384626433832795)
DECLARE_CONST(NEG_PI, -3.1415926535897932384626433832795)
DECLARE_CONST(PI_OVER_TWO, 1.57079632679489661923132169)
DECLARE_CONST(NEG_PI_OVER_TWO, -1.57079632679489661923132169)
DECLARE_CONST(PI_OVER_FOUR, 0.7853981633974483096156608)
DECLARE_CONST(THREE_PI_OVER_EIGHT, 2.41421356237309504880)
DECLARE_CONST(FOUR_OVER_PI, 1.27323954473516)
DECLARE_CONST(LOG2EF, 1.44269504088896341)
DECLARE_CONST(SQRTHALF, 0.707106781186547524)

DECLARE_INT_CONST(NAN, 0xffffffff)
#define NAN_PS *(__m128*) NAN_PI32_
#define NAN_PD *(__m128d*) NAN_PI32_
DECLARE_INT_CONST(NEG_ZERO_PS, 0x80000000)
#define NEG_ZERO_PS *(__m128*) NEG_ZERO_PS_PI32_
DECLARE_LONG_CONST(NEG_ZERO_PD, 0x00000000, 0x80000000)
#define NEG_ZERO_PD *(__m128d*) NEG_ZERO_PD_PI32_

DECLARE_CONST(MIN_INT, -2147483648)
DECLARE_CONST(MAX_INT,  2147483647)

DECLARE_CONST(EXPHI_SINGLE, 88.3762626647949)
DECLARE_CONST(EXPLO_SINGLE, -88.3762626647949)
DECLARE_CONST(EXPHI_DOUBLE, 709.78288357821549920801706215)
DECLARE_CONST(EXPLO_DOUBLE, -709.78288357821549920801706215)

DECLARE_CONST(EXP_C1, 0.693359375)
DECLARE_CONST(EXP_C2, -2.12194440e-4)
DECLARE_CONST(EXP_P0, 1.9875691500E-4)
DECLARE_CONST(EXP_P1, 1.3981999507E-3)
DECLARE_CONST(EXP_P2, 8.3334519073E-3)
DECLARE_CONST(EXP_P3, 4.1665795894E-2)
DECLARE_CONST(EXP_P4, 1.6666665459E-1)
DECLARE_CONST(EXP_P5, 5.0000001201E-1)

DECLARE_CONST(LOG_P0, 7.0376836292E-2)
DECLARE_CONST(LOG_P1, -1.1514610310E-1)
DECLARE_CONST(LOG_P2, 1.1676998740E-1)
DECLARE_CONST(LOG_P3, -1.2420140846E-1)
DECLARE_CONST(LOG_P4, 1.4249322787E-1)
DECLARE_CONST(LOG_P5, -1.6668057665E-1)
DECLARE_CONST(LOG_P6, 2.0000714765E-1)
DECLARE_CONST(LOG_P7, -2.4999993993E-1)
DECLARE_CONST(LOG_P8, 3.3333331174E-1)
DECLARE_CONST(LOG_Q1, -2.12194440e-4)
DECLARE_CONST(LOG_Q2, 0.693359375)

DECLARE_CONST(DP1, -0.78515625)
DECLARE_CONST(DP2, -2.4187564849853515625e-4)
DECLARE_CONST(DP3, -3.77489497744594108e-8)

DECLARE_CONST(SINCOF_P0, -1.9515295891E-4)
DECLARE_CONST(SINCOF_P1, 8.3321608736E-3)
DECLARE_CONST(SINCOF_P2, -1.6666654611E-1)

DECLARE_CONST(COSCOF_P0, 2.443315711809948E-005)
DECLARE_CONST(COSCOF_P1, -1.388731625493765E-003)
DECLARE_CONST(COSCOF_P2, 4.166664568298827E-002)

DECLARE_CONST(ASIN_BRANCH1, 0.625)
DECLARE_CONST(ASIN_BRANCH2, 1.0e-8)
DECLARE_CONST(ASIN_MOREBITS, 6.123233995736765886130E-17)

DECLARE_CONST(ASIN_P0, 4.253011369004428248960E-3)
DECLARE_CONST(ASIN_P1, -6.019598008014123785661E-1)
DECLARE_CONST(ASIN_P2, 5.444622390564711410273E0)
DECLARE_CONST(ASIN_P3, -1.626247967210700244449E1)
DECLARE_CONST(ASIN_P4, 1.956261983317594739197E1)
DECLARE_CONST(ASIN_P5, -8.198089802484824371615E0)

DECLARE_CONST(ASIN_Q1, -1.474091372988853791896E1)
DECLARE_CONST(ASIN_Q2, 7.049610280856842141659E1)
DECLARE_CONST(ASIN_Q3, -1.471791292232726029859E2)
DECLARE_CONST(ASIN_Q4, 1.395105614657485689735E2)
DECLARE_CONST(ASIN_Q5, -4.918853881490881290097E1)

DECLARE_CONST(ASIN_R0, .967721961301243206100E-3)
DECLARE_CONST(ASIN_R1, -5.634242780008963776856E-1)
DECLARE_CONST(ASIN_R2, 6.968710824104713396794E0)
DECLARE_CONST(ASIN_R3, -2.556901049652824852289E1)
DECLARE_CONST(ASIN_R4, 2.853665548261061424989E1)

DECLARE_CONST(ASIN_S1, -2.194779531642920639778E1)
DECLARE_CONST(ASIN_S2, 1.470656354026814941758E2)
DECLARE_CONST(ASIN_S3, -3.838770957603691357202E2)
DECLARE_CONST(ASIN_S4, 3.424398657913078477438E2)

DECLARE_CONST(ATAN_BRANCH, 0.66)

DECLARE_CONST(ATAN_P0, -8.750608600031904122785E-1)
DECLARE_CONST(ATAN_P1, -1.615753718733365076637E1)
DECLARE_CONST(ATAN_P2, -7.500855792314704667340E1)
DECLARE_CONST(ATAN_P3, -1.228866684490136173410E2)
DECLARE_CONST(ATAN_P4, -6.485021904942025371773E1)

DECLARE_CONST(ATAN_Q1, 2.485846490142306297962E1)
DECLARE_CONST(ATAN_Q2, 1.650270098316988542046E2)
DECLARE_CONST(ATAN_Q3, 4.328810604912902668951E2)
DECLARE_CONST(ATAN_Q4, 4.853903996359136964868E2)
DECLARE_CONST(ATAN_Q5, 1.945506571482613964425E2)


/**
 * Returns a word of a if the corresponding word in mask is 0x00000000 and a word of b otherwise (mask == 0xffffffff).
 * Example: To select the larger values of a, b: sel_ps (a, b, _mm_cmpgt (b, a))
 */
/*
extern inline __m128 sel_ps (const __m128 a, const __m128 b, const __m128 mask)
{
    // (((b ^ a) & mask) ^ a)
    return _mm_xor_ps (a, _mm_and_ps (mask, _mm_xor_ps (b, a)));
}*/
#define sel_ps(a, b, mask) _mm_xor_ps ((a), _mm_and_ps ((mask), _mm_xor_ps ((b), (a))))

/*
extern inline __m128d sel_pd (const __m128d a, const __m128d b, const __m128d mask)
{
    return _mm_xor_pd (a, _mm_and_pd (mask, _mm_xor_pd (b, a)));
}*/
#define sel_pd(a, b, mask) _mm_xor_pd ((a), _mm_and_pd ((mask), _mm_xor_pd ((b), (a))))

/*
extern inline __m128 neg_ps (__m128 x)
{
    // to perform negating, store 1 at the most significant bit and 0s at the rest bits. Then perform XOR.
    // http://www.songho.ca/misc/sse/sse.html
    
    return _mm_xor_ps (x, (__m128) SIGN_MASK_SINGLE);
}*/
#define neg_ps(x) _mm_xor_ps (x, (__m128) SIGN_MASK_SINGLE)

/*
extern inline __m128d neg_pd (__m128d x)
{
    // to perform negating, store 1 at the most significant bit and 0s at the rest bits. Then perform XOR.
    // http://www.songho.ca/misc/sse/sse.html
    
    return _mm_xor_pd (x, (__m128d) SIGN_MASK_DOUBLE);
}*/
#define neg_pd(x) _mm_xor_pd (x, (__m128d) SIGN_MASK_DOUBLE)

/*
extern inline __m128 abs_ps (__m128 x)
{
    // to perform absolute value operation, store 0 at the most significant bit (sign bit) and 1s
    // at the rest bits in source register. Then perform AND operation: number & 7FFFFFFFh.
    // http://www.songho.ca/misc/sse/sse.html
    
    return _mm_and_ps (x, (__m128) INV_SIGN_MASK_SINGLE);
}*/
#define abs_ps(x) _mm_and_ps (x, (__m128) INV_SIGN_MASK_SINGLE)

/*
extern inline __m128d abs_pd (__m128d x)
{
    // to perform absolute value operation, store 0 at the most significant bit (sign bit) and 1s
    // at the rest bits in source register. Then perform AND operation: number & 7FFFFFFFh.
    // http://www.songho.ca/misc/sse/sse.html
    
    return _mm_and_pd (x, (__m128d) INV_SIGN_MASK_DOUBLE);
}*/
#define abs_pd(x) _mm_and_pd (x, (__m128d) INV_SIGN_MASK_DOUBLE)

/*
extern inline __m128i cvttpd_epi64 (__m128d x)
{
    __m128 y = (__m128) _mm_cvttpd_epi32 (x);
    return (__m128i) _mm_shuffle_ps (y, y, _MM_SHUFFLE (3, 1, 3, 0));
}*/
#define cvttpd_epi64(x) (__m128i) _mm_shuffle_ps ((__m128) _mm_cvttpd_epi32 (x), (__m128) _mm_cvttpd_epi32 (x), _MM_SHUFFLE (3, 1, 3, 0))

/*
extern inline __m128d cvtepi64_pd (__m128i x)
{
    return _mm_cvtepi32_pd ((__m128i) _mm_shuffle_ps ((__m128) x, (__m128) x, _MM_SHUFFLE (3, 3, 2, 0)));
}*/
#define cvtepi64_pd(x) _mm_cvtepi32_pd ((__m128i) _mm_shuffle_ps ((__m128) (x), (__m128) (x), _MM_SHUFFLE (3, 3, 2, 0)))

__m128 floor_ps (__m128 x);
__m128d floor_pd (__m128d x);
__m128 ceil_ps (__m128 x);
__m128d ceil_pd (__m128d x);
__m128 exp_ps (__m128 x);
__m128d exp_pd (__m128d x);
__m128 log_ps (__m128 x);
__m128d log_pd (__m128d x);
__m128 log10_ps (__m128 x);
__m128d log10_pd (__m128d x);

/*
inline __m128 pow_ps (__m128 basis, __m128 exponent)
{
    return exp_ps (_mm_mul_ps (exponent, log_ps (basis)));
}*/
#define pow_ps(basis, exponent) exp_ps (_mm_mul_ps ((exponent), log_ps (basis)))

/*
inline __m128d pow_pd (__m128d basis, __m128d exponent)
{
    return exp_pd (_mm_mul_pd (exponent, log_pd (basis)));
}*/
#define pow_pd(basis, exponent) exp_pd (_mm_mul_pd ((exponent), log_pd (basis)))

__m128 sin_ps (__m128 x);
__m128d sin_pd (__m128d x);
__m128 cos_ps (__m128 x);
__m128d cos_pd (__m128d x);
void sincos_ps (__m128 x, __m128* s, __m128* c);
void sincos_pd (__m128d x, __m128d* s, __m128d* c);
__m128 tan_ps (__m128 x);
__m128d tan_pd (__m128d x);
__m128 asin_ps (__m128 x);
__m128d asin_pd (__m128d x);
__m128 acos_ps (__m128 x);
__m128d acos_pd (__m128d x);
__m128 atan_ps (__m128 x);
__m128d atan_pd (__m128d x);
__m128 atan2_ps (__m128 y, __m128 x);
__m128d atan2_pd (__m128d y, __m128d x);


DECLARE_CONST(ZERO, 0.0)

#define clamp0(x) ((x) > 0) ? 0 : (x)
#define clamp0_ps(x) _mm_and_ps (x, _mm_cmp_ps (x, ZERO_PS, 1))
#define clamp0_pd(x) _mm_and_pd (x, _mm_cmp_pd (x, ZERO_PD, 1))

#define clamp0if(value, threshold) ((threshold) > 0) ? 0 : (value)
#define clamp0if_ps(value, threshold) _mm_and_ps (value, _mm_cmp_ps (threshold, ZERO_PS, 1))
#define clamp0if_pd(value, threshold) _mm_and_pd (value, _mm_cmp_pd (threshold, ZERO_PD, 1))

#endif
