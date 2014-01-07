#include "ssemathlib.h"


__m128 floor_ps (__m128 x)
{
    // subtract 1 if the sign bit is set
    __m128i subtract = _mm_srli_epi32 ((__m128i) x, 31);
    
    // convert to int and back
    // cvttpd truncates, i.e. rounds towards zero:
    // 1.9 -> 1, 1.1 -> 1
    __m128 tmp = _mm_cvtepi32_ps (_mm_sub_epi32 (_mm_cvttps_epi32 (x), subtract));

    // return the conversion, but don't change values which are out of int range
    return sel_ps (
        x,
        sel_ps (x, tmp, _mm_cmpgt_ps (x, MIN_INT_PS)),
        _mm_cmplt_ps (x, MAX_INT_PS)
    );
}

__m128d floor_pd (__m128d x)
{
    // subtract 1 if the sign bit is set
    __m128i subtract = _mm_srli_epi64 ((__m128i) x, 63);
        
    // move the results into r0, r1
    subtract = (__m128i) _mm_shuffle_ps ((__m128) subtract, (__m128) subtract, _MM_SHUFFLE (1, 1, 2, 0));
    
    // convert to int and back
    // cvttpd truncates, i.e. rounds towards zero:
    // 1.9 -> 1, 1.1 -> 1
    __m128d tmp = _mm_cvtepi32_pd (_mm_sub_epi32 (_mm_cvttpd_epi32 (x), subtract));

    // return the conversion, but don't change values which are out of int range
    return sel_pd (
        x,
        sel_pd (x, tmp, _mm_cmpgt_pd (x, MIN_INT_PD)),
        _mm_cmplt_pd (x, MAX_INT_PD)
    );
}

__m128 ceil_ps (__m128 x)
{
    // http://www.masm32.com/board/index.php?topic=9514.0
    //
    // reference assembly code:
    //
    // .DATA
    //      sngMinusOneHalf REAL4 -0.5
    // .CODE
    //      movss xmm0, FP4(1.2)
    //      addss xmm0, xmm0
    //      movss xmm1, sngMinusOneHalf
    //      subss xmm1, xmm0
    //      cvtss2si eax, xmm1
    //      sar eax, 1
    //      neg eax
    //      cvtsi2ss xmm0, eax
    
    __m128 mhalf = { -0.5f, -0.5f, -0.5f, -0.5f };

    // cvtpd rounds towards the nearest int, i.e. 1.9 -> 2, 1.1 -> 1
    __m128i rounded = _mm_cvtps_epi32 (_mm_sub_ps (mhalf, _mm_add_ps (x, x)));
    
    // arithmetic shift to the right
    rounded = _mm_srai_epi32 (rounded, 1);

    // return the conversion, but don't change values which are out of int range
    return sel_ps (
        x,
        sel_ps (x, neg_ps (_mm_cvtepi32_ps (rounded)), _mm_cmpgt_ps (x, MIN_INT_PS)),
        _mm_cmplt_ps (x, MAX_INT_PS)
    );
}

__m128d ceil_pd (__m128d x)
{
    __m128d mhalf = { -0.5, -0.5 };

    // cvtpd rounds towards the nearest int, i.e. 1.9 -> 2, 1.1 -> 1
    __m128i rounded = _mm_cvtpd_epi32 (_mm_sub_pd (mhalf, _mm_add_pd (x, x)));
    
    // arithmetic shift to the right
    rounded = _mm_srai_epi32 (rounded, 1);

    // return the conversion, but don't change values which are out of int range
    return sel_pd (
        x,
        sel_pd (x, neg_pd (_mm_cvtepi32_pd (rounded)), _mm_cmpgt_pd (x, MIN_INT_PD)),
        _mm_cmplt_pd (x, MAX_INT_PD)
    );
}

__m128 exp_ps (__m128 x)
{
    __m128 tmp = _mm_setzero_ps (), fx;
    __m128i emm0;
    
    x = _mm_min_ps (x, EXPHI_SINGLE_PS);
    x = _mm_max_ps (x, EXPLO_SINGLE_PS);

    // express exp(x) as exp(g + n*log(2))
    fx = _mm_mul_ps (x, LOG2EF_PS);
    fx = _mm_add_ps (fx, HALF_PS);

    emm0 = _mm_cvttps_epi32 (fx);
    tmp = _mm_cvtepi32_ps (emm0);

    // if greater, substract 1
    __m128 mask = _mm_cmpgt_ps (tmp, fx);
    mask = _mm_and_ps (mask, ONE_PS);
    fx = _mm_sub_ps (tmp, mask);

    tmp = _mm_mul_ps (fx, EXP_C1_PS);
    __m128 z = _mm_mul_ps (fx, EXP_C2_PS);
    x = _mm_sub_ps (x, tmp);
    x = _mm_sub_ps (x, z);

    z = _mm_mul_ps (x,x);
  
    __m128 y = EXP_P0_PS;
    y = _mm_mul_ps (y, x);
    y = _mm_add_ps (y, EXP_P1_PS);
    y = _mm_mul_ps (y, x);
    y = _mm_add_ps (y, EXP_P2_PS);
    y = _mm_mul_ps (y, x);
    y = _mm_add_ps (y, EXP_P3_PS);
    y = _mm_mul_ps (y, x);
    y = _mm_add_ps (y, EXP_P4_PS);
    y = _mm_mul_ps (y, x);
    y = _mm_add_ps (y, EXP_P5_PS);
    y = _mm_mul_ps (y, z);
    y = _mm_add_ps (y, x);
    y = _mm_add_ps (y, ONE_PS);

    // build 2^n
    emm0 = _mm_cvttps_epi32 (fx);
    emm0 = _mm_add_epi32 (emm0, EXPBIAS_SINGLE);
    emm0 = _mm_slli_epi32 (emm0, MANTISSA_BITS_SINGLE);
    __m128 pow2n = _mm_castsi128_ps (emm0);

    return _mm_mul_ps (y, pow2n);
}

__m128d exp_pd (__m128d x)
{
    __m128d tmp = _mm_setzero_pd (), fx;
    __m128i emm0;
    
    x = _mm_min_pd (x, EXPHI_DOUBLE_PD);
    x = _mm_max_pd (x, EXPLO_DOUBLE_PD);

    // express exp(x) as exp(g + n*log(2))
    fx = _mm_mul_pd (x, LOG2EF_PD);
    fx = _mm_add_pd (fx, HALF_PD);

    emm0 = _mm_cvttpd_epi32 (fx);
    tmp = _mm_cvtepi32_pd (emm0);

    // if greater, substract 1
    __m128d mask = _mm_cmpgt_pd (tmp, fx);
    mask = _mm_and_pd (mask, ONE_PD);
    fx = _mm_sub_pd (tmp, mask);

    tmp = _mm_mul_pd (fx, EXP_C1_PD);
    __m128d z = _mm_mul_pd (fx, EXP_C2_PD);
    x = _mm_sub_pd (x, tmp);
    x = _mm_sub_pd (x, z);

    z = _mm_mul_pd (x,x);
  
    __m128d y = EXP_P0_PD;
    y = _mm_mul_pd (y, x);
    y = _mm_add_pd (y, EXP_P1_PD);
    y = _mm_mul_pd (y, x);
    y = _mm_add_pd (y, EXP_P2_PD);
    y = _mm_mul_pd (y, x);
    y = _mm_add_pd (y, EXP_P3_PD);
    y = _mm_mul_pd (y, x);
    y = _mm_add_pd (y, EXP_P4_PD);
    y = _mm_mul_pd (y, x);
    y = _mm_add_pd (y, EXP_P5_PD);
    y = _mm_mul_pd (y, z);
    y = _mm_add_pd (y, x);
    y = _mm_add_pd (y, ONE_PD);

    // build 2^n
    emm0 = cvttpd_epi64 (fx);
    emm0 = _mm_add_epi32 (emm0, EXPBIAS_DOUBLE);    // use 32 because cvttpd_epi64 always sets the upper words to 0
    emm0 = _mm_slli_epi64 (emm0, MANTISSA_BITS_DOUBLE);
    __m128d pow2n = _mm_castsi128_pd (emm0);

    return _mm_mul_pd (y, pow2n);
}

__m128 log_ps (__m128 x)
{
    __m128i emm0;
    __m128 invalid_mask = _mm_cmple_ps (x, _mm_setzero_ps ());

    // cut off denormalized stuff
    x = _mm_max_ps (x, (__m128) MINNORMPOS_SINGLE);

    // extract exponent
    emm0 = _mm_castps_si128 (x);
    emm0 = _mm_srli_epi32 (emm0, MANTISSA_BITS_SINGLE);
  
    // keep only the fractional part
    x = _mm_and_ps (x, (__m128) INVMANTISSAMASK_SINGLE);
    x = _mm_or_ps (x, HALF_PS);
  
    // e = floor (log2 (x))
    emm0 = _mm_sub_epi32 (emm0, EXPBIAS_SINGLE);
    __m128 e = _mm_cvtepi32_ps (emm0);
  
    e = _mm_add_ps (e, ONE_PS);
    
    __m128 mask = _mm_cmplt_ps (x, SQRTHALF_PS);
    __m128 tmp = _mm_and_ps (x, mask);
    x = _mm_sub_ps (x, ONE_PS);
    e = _mm_sub_ps (e, _mm_and_ps (ONE_PS, mask));
    x = _mm_add_ps (x, tmp);

    __m128 z = _mm_mul_ps (x, x);

    __m128 y = LOG_P0_PS;
    y = _mm_mul_ps (y, x);
    y = _mm_add_ps (y, LOG_P1_PS);
    y = _mm_mul_ps (y, x);
    y = _mm_add_ps (y, LOG_P2_PS);
    y = _mm_mul_ps (y, x);
    y = _mm_add_ps (y, LOG_P3_PS);
    y = _mm_mul_ps (y, x);
    y = _mm_add_ps (y, LOG_P4_PS);
    y = _mm_mul_ps (y, x);
    y = _mm_add_ps (y, LOG_P5_PS);
    y = _mm_mul_ps (y, x);
    y = _mm_add_ps (y, LOG_P6_PS);
    y = _mm_mul_ps (y, x);
    y = _mm_add_ps (y, LOG_P7_PS);
    y = _mm_mul_ps (y, x);
    y = _mm_add_ps (y, LOG_P8_PS);
    y = _mm_mul_ps (y, x);

    y = _mm_mul_ps (y, z);

    tmp = _mm_mul_ps (e, LOG_Q1_PS);
    y = _mm_add_ps (y, tmp);

    tmp = _mm_mul_ps (z, HALF_PS);
    y = _mm_sub_ps (y, tmp);

    tmp = _mm_mul_ps (e, LOG_Q2_PS);
    x = _mm_add_ps (x, y);
    x = _mm_add_ps (x, tmp);
    
    // set negative args to be NaN
    x = _mm_or_ps (x, invalid_mask);

    return x;
}

__m128d log_pd (__m128d x)
{
    __m128i emm0;
    __m128d invalid_mask = _mm_cmple_pd (x, _mm_setzero_pd ());

    // cut off denormalized stuff
    x = _mm_max_pd (x, (__m128d) MINNORMPOS_DOUBLE);

    emm0 = _mm_castpd_si128 (x);
    emm0 = _mm_srli_epi64 (emm0, MANTISSA_BITS_DOUBLE);
  
    // keep only the fractional part
    x = _mm_and_pd (x, (__m128d) INVMANTISSAMASK_DOUBLE);
    x = _mm_or_pd (x, HALF_PD);
  
    emm0 = _mm_sub_epi64 (emm0, EXPBIAS_DOUBLE);
    __m128d e = cvtepi64_pd (emm0);
  
    e = _mm_add_pd (e, ONE_PD);
    
    __m128d mask = _mm_cmplt_pd (x, SQRTHALF_PD);
    __m128d tmp = _mm_and_pd (x, mask);
    x = _mm_sub_pd (x, ONE_PD);
    e = _mm_sub_pd (e, _mm_and_pd (ONE_PD, mask));
    x = _mm_add_pd (x, tmp);

    __m128d z = _mm_mul_pd (x, x);

    __m128d y = LOG_P0_PD;
    y = _mm_mul_pd (y, x);
    y = _mm_add_pd (y, LOG_P1_PD);
    y = _mm_mul_pd (y, x);
    y = _mm_add_pd (y, LOG_P2_PD);
    y = _mm_mul_pd (y, x);
    y = _mm_add_pd (y, LOG_P3_PD);
    y = _mm_mul_pd (y, x);
    y = _mm_add_pd (y, LOG_P4_PD);
    y = _mm_mul_pd (y, x);
    y = _mm_add_pd (y, LOG_P5_PD);
    y = _mm_mul_pd (y, x);
    y = _mm_add_pd (y, LOG_P6_PD);
    y = _mm_mul_pd (y, x);
    y = _mm_add_pd (y, LOG_P7_PD);
    y = _mm_mul_pd (y, x);
    y = _mm_add_pd (y, LOG_P8_PD);
    y = _mm_mul_pd (y, x);

    y = _mm_mul_pd (y, z);

    tmp = _mm_mul_pd (e, LOG_Q1_PD);
    y = _mm_add_pd (y, tmp);

    tmp = _mm_mul_pd (z, HALF_PD);
    y = _mm_sub_pd (y, tmp);

    tmp = _mm_mul_pd (e, LOG_Q2_PD);
    x = _mm_add_pd (x, y);
    x = _mm_add_pd (x, tmp);
    
    // set negative args to be NaN
    x = _mm_or_pd (x, invalid_mask);

    return x;
}

__m128 log10_ps (__m128 x)
{
    __m128 basis = { 2.30258509f, 2.30258509f, 2.30258509f, 2.30258509f };
    return _mm_div_ps (log_ps (x), basis);
}

__m128d log10_pd (__m128d x)
{
    __m128d basis = { 2.3025850929940456840179914546843642076, 2.3025850929940456840179914546843642076 };
    return _mm_div_pd (log_pd (x), basis);
}

__m128 sin_ps (__m128 x)
{
    __m128 xmm1, xmm2 = _mm_setzero_ps (), xmm3, sign_bit, y;
    __m128i emm0, emm2;

    sign_bit = x;
    // take the absolute value
    x = abs_ps (x);
    // extract the sign bit (upper one)
    sign_bit = _mm_and_ps (sign_bit, (__m128) SIGN_MASK_SINGLE);
  
    // scale by 4/Pi
    y = _mm_mul_ps (x, FOUR_OVER_PI_PS);

    // store the integer part of y in mm0
    emm2 = _mm_cvttps_epi32 (y);
    // j=(j+1) & (~1) (see the cephes sources)
    emm2 = _mm_add_epi32 (emm2, ONE_PI32);
    emm2 = _mm_and_si128 (emm2, INVONE_PI32);
    y = _mm_cvtepi32_ps (emm2);
    
    // get the swap sign flag
    emm0 = _mm_and_si128 (emm2, FOUR_PI32);
    emm0 = _mm_slli_epi32 (emm0, 29);
    
    // get the polynomial selection mask:
    // there is one polynomial for 0 <= x <= Pi/4 and another one for Pi/4<x<=Pi/2
    // both branches will be computed
    emm2 = _mm_and_si128 (emm2, TWO_PI32);
    emm2 = _mm_cmpeq_epi32 (emm2, _mm_setzero_si128 ());
  
    __m128 swap_sign_bit = _mm_castsi128_ps (emm0);
    __m128 poly_mask = _mm_castsi128_ps (emm2);
    sign_bit = _mm_xor_ps (sign_bit, swap_sign_bit);
  
    // the magic pass: "Extended precision modular arithmetic"
    // x = ((x - y * DP1) - y * DP2) - y * DP3
    xmm1 = _mm_mul_ps (y, DP1_PS);
    xmm2 = _mm_mul_ps (y, DP2_PS);
    xmm3 = _mm_mul_ps (y, DP3_PS);
    x = _mm_add_ps (x, xmm1);
    x = _mm_add_ps (x, xmm2);
    x = _mm_add_ps (x, xmm3);

    // evaluate the first polynomial (0 <= x <= Pi/4)
    y = COSCOF_P0_PS;
    __m128 z = _mm_mul_ps (x, x);

    y = _mm_mul_ps (y, z);
    y = _mm_add_ps (y, COSCOF_P1_PS);
    y = _mm_mul_ps (y, z);
    y = _mm_add_ps (y, COSCOF_P2_PS);
    y = _mm_mul_ps (y, z);
    y = _mm_mul_ps (y, z);
    __m128 tmp = _mm_mul_ps (z, HALF_PS);
    y = _mm_sub_ps (y, tmp);
    y = _mm_add_ps (y, ONE_PS);
  
    // evaluate the second polynomial (Pi/4 <= x <= 0)
    __m128 y2 = SINCOF_P0_PS;
    y2 = _mm_mul_ps (y2, z);
    y2 = _mm_add_ps (y2, SINCOF_P1_PS);
    y2 = _mm_mul_ps (y2, z);
    y2 = _mm_add_ps (y2, SINCOF_P2_PS);
    y2 = _mm_mul_ps (y2, z);
    y2 = _mm_mul_ps (y2, x);
    y2 = _mm_add_ps (y2, x);

    // select the correct result from the two polynomials
    y = sel_ps (y, y2, poly_mask);

    // update the sign
    y = _mm_xor_ps (y, sign_bit);

    return y;
}

__m128d sin_pd (__m128d x)
{
    __m128d xmm1, xmm2 = _mm_setzero_pd (), xmm3, sign_bit, y;
    __m128i emm0, emm2;

    sign_bit = x;
    // take the absolute value
    x = abs_pd (x);
    // extract the sign bit (upper one)
    sign_bit = _mm_and_pd (sign_bit, (__m128d) SIGN_MASK_DOUBLE);
  
    // scale by 4/Pi
    y = _mm_mul_pd (x, FOUR_OVER_PI_PD);

    // store the integer part of y in emm2
    emm2 = _mm_cvttpd_epi32 (y);
    // j=(j+1) & (~1) (see the cephes sources)
    emm2 = _mm_add_epi32 (emm2, ONE_PI32);
    emm2 = _mm_and_si128 (emm2, INVONE_PI32);
    y = _mm_cvtepi32_pd (emm2);
    
    // get the swap sign flag
    emm0 = _mm_and_si128 (emm2, FOUR_PI32);
    emm0 = _mm_slli_epi32 (emm0, 29);
    // emm0 contains data (the sign flag 0x00000000 or 0x80000000) in the first two 32-bit words;
    // create two 64-bit words (0x0000000000000000 or 0x8000000000000000)
    // (the last two words contain 0x00000000)
    emm0 = (__m128i) _mm_shuffle_ps ((__m128) emm0, (__m128) emm0, _MM_SHUFFLE (1, 2, 0, 2));
    
    // get the polynomial selection mask:
    // there is one polynomial for 0 <= x <= Pi/4 and another one for Pi/4<x<=Pi/2
    // both branches will be computed
    emm2 = _mm_and_si128 (emm2, TWO_PI32);
    emm2 = _mm_cmpeq_epi32 (emm2, _mm_setzero_si128 ());
    // emm2 contains data (0x00000000 or 0xffffffff) in the first two 32-bit words; create two 64-bit words (0x0000000000000000 or 0xffffffffffffffff)
    emm2 = (__m128i) _mm_shuffle_ps ((__m128) emm2, (__m128) emm2, _MM_SHUFFLE (1, 1, 0, 0));
  
    __m128d swap_sign_bit = _mm_castsi128_pd (emm0);
    __m128d poly_mask = _mm_castsi128_pd (emm2);
    sign_bit = _mm_xor_pd (sign_bit, swap_sign_bit);
  
    // the magic pass: "Extended precision modular arithmetic"
    // x = ((x - y * DP1) - y * DP2) - y * DP3
    xmm1 = _mm_mul_pd (y, DP1_PD);
    xmm2 = _mm_mul_pd (y, DP2_PD);
    xmm3 = _mm_mul_pd (y, DP3_PD);
    x = _mm_add_pd (x, xmm1);
    x = _mm_add_pd (x, xmm2);
    x = _mm_add_pd (x, xmm3);

    // evaluate the first polynomial (0 <= x <= Pi/4)
    y = COSCOF_P0_PD;
    __m128d z = _mm_mul_pd (x, x);

    y = _mm_mul_pd (y, z);
    y = _mm_add_pd (y, COSCOF_P1_PD);
    y = _mm_mul_pd (y, z);
    y = _mm_add_pd (y, COSCOF_P2_PD);
    y = _mm_mul_pd (y, z);
    y = _mm_mul_pd (y, z);
    __m128d tmp = _mm_mul_pd (z, HALF_PD);
    y = _mm_sub_pd (y, tmp);
    y = _mm_add_pd (y, ONE_PD);
  
    // evaluate the second polynomial (Pi/4 <= x <= 0)
    __m128d y2 = SINCOF_P0_PD;
    y2 = _mm_mul_pd (y2, z);
    y2 = _mm_add_pd (y2, SINCOF_P1_PD);
    y2 = _mm_mul_pd (y2, z);
    y2 = _mm_add_pd (y2, SINCOF_P2_PD);
    y2 = _mm_mul_pd (y2, z);
    y2 = _mm_mul_pd (y2, x);
    y2 = _mm_add_pd (y2, x);

    // select the correct result from the two polynomials
    y = sel_pd (y, y2, poly_mask);

    // update the sign
    y = _mm_xor_pd (y, sign_bit);

    return y;
}

__m128 cos_ps (__m128 x)
{
    __m128 xmm1, xmm2 = _mm_setzero_ps (), xmm3, y;
    __m128i emm0, emm2;

    // take the absolute value
    x = abs_ps (x);
  
    // scale by 4/Pi
    y = _mm_mul_ps (x, FOUR_OVER_PI_PS);

    // store the integer part of y in mm0
    emm2 = _mm_cvttps_epi32 (y);
    // j=(j+1) & (~1) (see the cephes sources)
    emm2 = _mm_add_epi32 (emm2, ONE_PI32);
    emm2 = _mm_and_si128 (emm2, INVONE_PI32);
    y = _mm_cvtepi32_ps (emm2);
    
    // get the swap sign flag
    emm2 = _mm_sub_epi32 (emm2, TWO_PI32);
    emm0 = _mm_andnot_si128 (emm2, FOUR_PI32);
    emm0 = _mm_slli_epi32 (emm0, 29);
    
    // get the polynomial selection mask:
    // there is one polynomial for 0 <= x <= Pi/4 and another one for Pi/4<x<=Pi/2
    // both branches will be computed
    emm2 = _mm_and_si128 (emm2, TWO_PI32);
    emm2 = _mm_cmpeq_epi32 (emm2, _mm_setzero_si128 ());
  
    __m128 sign_bit = _mm_castsi128_ps (emm0);
    __m128 poly_mask = _mm_castsi128_ps (emm2);
  
    // the magic pass: "Extended precision modular arithmetic"
    // x = ((x - y * DP1) - y * DP2) - y * DP3
    xmm1 = _mm_mul_ps (y, DP1_PS);
    xmm2 = _mm_mul_ps (y, DP2_PS);
    xmm3 = _mm_mul_ps (y, DP3_PS);
    x = _mm_add_ps (x, xmm1);
    x = _mm_add_ps (x, xmm2);
    x = _mm_add_ps (x, xmm3);

    // evaluate the first polynomial (0 <= x <= Pi/4)
    y = COSCOF_P0_PS;
    __m128 z = _mm_mul_ps (x, x);

    y = _mm_mul_ps (y, z);
    y = _mm_add_ps (y, COSCOF_P1_PS);
    y = _mm_mul_ps (y, z);
    y = _mm_add_ps (y, COSCOF_P2_PS);
    y = _mm_mul_ps (y, z);
    y = _mm_mul_ps (y, z);
    __m128 tmp = _mm_mul_ps (z, HALF_PS);
    y = _mm_sub_ps (y, tmp);
    y = _mm_add_ps (y, ONE_PS);
  
    // evaluate the second polynomial (Pi/4 <= x <= 0)
    __m128 y2 = SINCOF_P0_PS;
    y2 = _mm_mul_ps (y2, z);
    y2 = _mm_add_ps (y2, SINCOF_P1_PS);
    y2 = _mm_mul_ps (y2, z);
    y2 = _mm_add_ps (y2, SINCOF_P2_PS);
    y2 = _mm_mul_ps (y2, z);
    y2 = _mm_mul_ps (y2, x);
    y2 = _mm_add_ps (y2, x);

    // select the correct result from the two polynomials
    y = sel_ps (y, y2, poly_mask);

    // update the sign
    y = _mm_xor_ps (y, sign_bit);

    return y;
}

__m128d cos_pd (__m128d x)
{
    __m128d xmm1, xmm2 = _mm_setzero_pd (), xmm3, y;
    __m128i emm0, emm2;

    // take the absolute value
    x = abs_pd (x);
  
    // scale by 4/Pi
    y = _mm_mul_pd (x, FOUR_OVER_PI_PD);

    // store the integer part of y in mm0
    emm2 = _mm_cvttpd_epi32 (y);
    // j=(j+1) & (~1) (see the cephes sources)
    emm2 = _mm_add_epi32 (emm2, ONE_PI32);
    emm2 = _mm_and_si128 (emm2, INVONE_PI32);
    y = _mm_cvtepi32_pd (emm2);
    
    // get the swap sign flag
    emm2 = _mm_sub_epi32 (emm2, TWO_PI32);
    emm0 = _mm_andnot_si128 (emm2, FOUR_PI32);
    emm0 = _mm_slli_epi32 (emm0, 29);
    // emm0 contains data (the sign flag 0x00000000 or 0x80000000) in the first two 32-bit words;
    // create two 64-bit words (0x0000000000000000 or 0x8000000000000000)
    // (the last two words contain 0x00000000)
    emm0 = (__m128i) _mm_shuffle_ps ((__m128) emm0, (__m128) emm0, _MM_SHUFFLE (1, 2, 0, 2));
    
    // get the polynomial selection mask:
    // there is one polynomial for 0 <= x <= Pi/4 and another one for Pi/4<x<=Pi/2
    // both branches will be computed
    emm2 = _mm_and_si128 (emm2, TWO_PI32);
    emm2 = _mm_cmpeq_epi32 (emm2, _mm_setzero_si128 ());
    // emm2 contains data (0x00000000 or 0xffffffff) in the first two 32-bit words; create two 64-bit words (0x0000000000000000 or 0xffffffffffffffff)
    emm2 = (__m128i) _mm_shuffle_ps ((__m128) emm2, (__m128) emm2, _MM_SHUFFLE (1, 1, 0, 0));
  
    __m128d sign_bit = _mm_castsi128_pd (emm0);
    __m128d poly_mask = _mm_castsi128_pd (emm2);
  
    // the magic pass: "Extended precision modular arithmetic"
    // x = ((x - y * DP1) - y * DP2) - y * DP3
    xmm1 = _mm_mul_pd (y, DP1_PD);
    xmm2 = _mm_mul_pd (y, DP2_PD);
    xmm3 = _mm_mul_pd (y, DP3_PD);
    x = _mm_add_pd (x, xmm1);
    x = _mm_add_pd (x, xmm2);
    x = _mm_add_pd (x, xmm3);

    // evaluate the first polynomial (0 <= x <= Pi/4)
    y = COSCOF_P0_PD;
    __m128d z = _mm_mul_pd (x, x);

    y = _mm_mul_pd (y, z);
    y = _mm_add_pd (y, COSCOF_P1_PD);
    y = _mm_mul_pd (y, z);
    y = _mm_add_pd (y, COSCOF_P2_PD);
    y = _mm_mul_pd (y, z);
    y = _mm_mul_pd (y, z);
    __m128d tmp = _mm_mul_pd (z, HALF_PD);
    y = _mm_sub_pd (y, tmp);
    y = _mm_add_pd (y, ONE_PD);
  
    // evaluate the second polynomial (Pi/4 <= x <= 0)
    __m128d y2 = SINCOF_P0_PD;
    y2 = _mm_mul_pd (y2, z);
    y2 = _mm_add_pd (y2, SINCOF_P1_PD);
    y2 = _mm_mul_pd (y2, z);
    y2 = _mm_add_pd (y2, SINCOF_P2_PD);
    y2 = _mm_mul_pd (y2, z);
    y2 = _mm_mul_pd (y2, x);
    y2 = _mm_add_pd (y2, x);

    // select the correct result from the two polynomials
    y = sel_pd (y, y2, poly_mask);

    // update the sign
    y = _mm_xor_pd (y, sign_bit);

    return y;
}

void sincos_ps (__m128 x, __m128* s, __m128* c)
{
    __m128 xmm1, xmm2 = _mm_setzero_ps (), xmm3, sign_bit_sin, y;
    __m128i emm0, emm2, emm4;

    sign_bit_sin = x;
    // take the absolute value
    x = abs_ps (x);
    // extract the sign bit (upper one)
    sign_bit_sin = _mm_and_ps (sign_bit_sin, (__m128) SIGN_MASK_SINGLE);
  
    // scale by 4/Pi
    y = _mm_mul_ps (x, FOUR_OVER_PI_PS);

    // store the integer part of y in mm0
    emm2 = _mm_cvttps_epi32 (y);
    // j=(j+1) & (~1) (see the cephes sources)
    emm2 = _mm_add_epi32 (emm2, ONE_PI32);
    emm2 = _mm_and_si128 (emm2, INVONE_PI32);
    emm4 = emm2;
    y = _mm_cvtepi32_ps (emm2);
    
    // get the swap sign flag for sine
    emm0 = _mm_and_si128 (emm2, FOUR_PI32);
    emm0 = _mm_slli_epi32 (emm0, 29);
    __m128 swap_sign_bit_sin = _mm_castsi128_ps (emm0);
    
    // get the polynomial selection mask:
    // there is one polynomial for 0 <= x <= Pi/4 and another one for Pi/4<x<=Pi/2
    // both branches will be computed
    emm2 = _mm_and_si128 (emm2, TWO_PI32);
    emm2 = _mm_cmpeq_epi32 (emm2, _mm_setzero_si128 ());
    __m128 poly_mask = _mm_castsi128_ps (emm2);
  
    // the magic pass: "Extended precision modular arithmetic"
    // x = ((x - y * DP1) - y * DP2) - y * DP3
    xmm1 = _mm_mul_ps (y, DP1_PS);
    xmm2 = _mm_mul_ps (y, DP2_PS);
    xmm3 = _mm_mul_ps (y, DP3_PS);
    x = _mm_add_ps (x, xmm1);
    x = _mm_add_ps (x, xmm2);
    x = _mm_add_ps (x, xmm3);
    
    emm4 = _mm_sub_epi32 (emm4, TWO_PI32);
    emm4 = _mm_andnot_si128 (emm4, FOUR_PI32);
    emm4 = _mm_slli_epi32 (emm4, 29);
    __m128 sign_bit_cos = _mm_castsi128_ps (emm4);
    sign_bit_sin = _mm_xor_ps (sign_bit_sin, swap_sign_bit_sin);

    // evaluate the first polynomial (0 <= x <= Pi/4)
    y = COSCOF_P0_PS;
    __m128 z = _mm_mul_ps (x, x);

    y = _mm_mul_ps (y, z);
    y = _mm_add_ps (y, COSCOF_P1_PS);
    y = _mm_mul_ps (y, z);
    y = _mm_add_ps (y, COSCOF_P2_PS);
    y = _mm_mul_ps (y, z);
    y = _mm_mul_ps (y, z);
    __m128 tmp = _mm_mul_ps (z, HALF_PS);
    y = _mm_sub_ps (y, tmp);
    y = _mm_add_ps (y, ONE_PS);
  
    // evaluate the second polynomial (Pi/4 <= x <= 0)
    __m128 y2 = SINCOF_P0_PS;
    y2 = _mm_mul_ps (y2, z);
    y2 = _mm_add_ps (y2, SINCOF_P1_PS);
    y2 = _mm_mul_ps (y2, z);
    y2 = _mm_add_ps (y2, SINCOF_P2_PS);
    y2 = _mm_mul_ps (y2, z);
    y2 = _mm_mul_ps (y2, x);
    y2 = _mm_add_ps (y2, x);

    // select the correct result from the two polynomials
    __m128 ysin2 = _mm_and_ps (poly_mask, y2);
    __m128 ysin1 = _mm_andnot_ps (poly_mask, y);
    y2 = _mm_sub_ps (y2,ysin2);
    y = _mm_sub_ps (y, ysin1);
    xmm1 = _mm_add_ps (ysin1, ysin2);
    xmm2 = _mm_add_ps (y, y2);
  
    // update the sign
    *s = _mm_xor_ps (xmm1, sign_bit_sin);
    *c = _mm_xor_ps (xmm2, sign_bit_cos);
}

void sincos_pd (__m128d x, __m128d* s, __m128d* c)
{
    __m128d xmm1, xmm2 = _mm_setzero_pd (), xmm3, sign_bit_sin, y;
    __m128i emm0, emm2, emm4;

    sign_bit_sin = x;
    // take the absolute value
    x = abs_pd (x);
    // extract the sign bit (upper one)
    sign_bit_sin = _mm_and_pd (sign_bit_sin, (__m128d) SIGN_MASK_DOUBLE);
  
    // scale by 4/Pi
    y = _mm_mul_pd (x, FOUR_OVER_PI_PD);

    // store the integer part of y in emm2
    emm2 = _mm_cvttpd_epi32 (y);
    // j=(j+1) & (~1) (see the cephes sources)
    emm2 = _mm_add_epi32 (emm2, ONE_PI32);
    emm2 = _mm_and_si128 (emm2, INVONE_PI32);
    emm4 = emm2;
    y = _mm_cvtepi32_pd (emm2);
    
    // get the swap sign flag for sine
    emm0 = _mm_and_si128 (emm2, FOUR_PI32);
    emm0 = _mm_slli_epi32 (emm0, 29);
    // emm0 contains data (the sign flag 0x00000000 or 0x80000000) in the first two 32-bit words;
    // create two 64-bit words (0x0000000000000000 or 0x8000000000000000)
    // (the last two words contain 0x00000000)
    emm0 = (__m128i) _mm_shuffle_ps ((__m128) emm0, (__m128) emm0, _MM_SHUFFLE (1, 2, 0, 2));
    __m128d swap_sign_bit_sin = _mm_castsi128_pd (emm0);
    
    // get the polynomial selection mask:
    // there is one polynomial for 0 <= x <= Pi/4 and another one for Pi/4<x<=Pi/2
    // both branches will be computed
    emm2 = _mm_and_si128 (emm2, TWO_PI32);
    emm2 = _mm_cmpeq_epi32 (emm2, _mm_setzero_si128 ());
    // emm2 contains data (0x00000000 or 0xffffffff) in the first two 32-bit words; create two 64-bit words (0x0000000000000000 or 0xffffffffffffffff)
    emm2 = (__m128i) _mm_shuffle_ps ((__m128) emm2, (__m128) emm2, _MM_SHUFFLE (1, 1, 0, 0));
    __m128d poly_mask = _mm_castsi128_pd (emm2);
  
    // the magic pass: "Extended precision modular arithmetic"
    // x = ((x - y * DP1) - y * DP2) - y * DP3
    xmm1 = _mm_mul_pd (y, DP1_PD);
    xmm2 = _mm_mul_pd (y, DP2_PD);
    xmm3 = _mm_mul_pd (y, DP3_PD);
    x = _mm_add_pd (x, xmm1);
    x = _mm_add_pd (x, xmm2);
    x = _mm_add_pd (x, xmm3);
    
    emm4 = _mm_sub_epi32 (emm4, TWO_PI32);
    emm4 = _mm_andnot_si128 (emm4, FOUR_PI32);
    emm4 = _mm_slli_epi32 (emm4, 29);
    // emm4 contains data (the sign flag 0x00000000 or 0x80000000) in the first two 32-bit words;
    // create two 64-bit words (0x0000000000000000 or 0x8000000000000000)
    // (the last two words contain 0x00000000)
    emm4 = (__m128i) _mm_shuffle_ps ((__m128) emm4, (__m128) emm4, _MM_SHUFFLE (1, 2, 0, 2));
    __m128d sign_bit_cos = _mm_castsi128_pd (emm4);
    sign_bit_sin = _mm_xor_pd (sign_bit_sin, swap_sign_bit_sin);

    // evaluate the first polynomial (0 <= x <= Pi/4)
    y = COSCOF_P0_PD;
    __m128d z = _mm_mul_pd (x, x);

    y = _mm_mul_pd (y, z);
    y = _mm_add_pd (y, COSCOF_P1_PD);
    y = _mm_mul_pd (y, z);
    y = _mm_add_pd (y, COSCOF_P2_PD);
    y = _mm_mul_pd (y, z);
    y = _mm_mul_pd (y, z);
    __m128d tmp = _mm_mul_pd (z, HALF_PD);
    y = _mm_sub_pd (y, tmp);
    y = _mm_add_pd (y, ONE_PD);
  
    // evaluate the second polynomial (Pi/4 <= x <= 0)
    __m128d y2 = SINCOF_P0_PD;
    y2 = _mm_mul_pd (y2, z);
    y2 = _mm_add_pd (y2, SINCOF_P1_PD);
    y2 = _mm_mul_pd (y2, z);
    y2 = _mm_add_pd (y2, SINCOF_P2_PD);
    y2 = _mm_mul_pd (y2, z);
    y2 = _mm_mul_pd (y2, x);
    y2 = _mm_add_pd (y2, x);

    // select the correct result from the two polynomials
    __m128d ysin2 = _mm_and_pd (poly_mask, y2);
    __m128d ysin1 = _mm_andnot_pd (poly_mask, y);
    y2 = _mm_sub_pd (y2, ysin2);
    y = _mm_sub_pd (y, ysin1);
    xmm1 = _mm_add_pd (ysin1, ysin2);
    xmm2 = _mm_add_pd (y, y2);
  
    // update the sign
    *s = _mm_xor_pd (xmm1, sign_bit_sin);
    *c = _mm_xor_pd (xmm2, sign_bit_cos);
}

__m128 tan_ps (__m128 x)
{
    __m128 sine, cosine;
    sincos_ps (x, &sine, &cosine);
    return _mm_div_ps (sine, cosine);
}

__m128d tan_pd (__m128d x)
{
    __m128d sine, cosine;
    sincos_pd (x, &sine, &cosine);
    return _mm_div_pd (sine, cosine);
}

__m128 asin_ps (__m128 x)
{
    __m128 sign_bit = _mm_and_ps (x, (__m128) SIGN_MASK_SINGLE);
    __m128 abs_x = abs_ps (x);
    __m128 invalid_mask = _mm_cmpgt_ps (abs_x, ONE_PS);

    __m128 zz1 = _mm_sub_ps (ONE_PS, abs_x);
    __m128 p1 = ASIN_R0_PS;
    p1 = _mm_mul_ps (p1, zz1);
    p1 = _mm_add_ps (p1, ASIN_R1_PS);
    p1 = _mm_mul_ps (p1, zz1);
    p1 = _mm_add_ps (p1, ASIN_R2_PS);
    p1 = _mm_mul_ps (p1, zz1);
    p1 = _mm_add_ps (p1, ASIN_R3_PS);
    p1 = _mm_mul_ps (p1, zz1);
    p1 = _mm_add_ps (p1, ASIN_R4_PS);
    __m128 q1 = zz1;
    q1 = _mm_add_ps (q1, ASIN_S1_PS);
    q1 = _mm_mul_ps (q1, zz1);
    q1 = _mm_add_ps (q1, ASIN_S2_PS);
    q1 = _mm_mul_ps (q1, zz1);
    q1 = _mm_add_ps (q1, ASIN_S3_PS);
    q1 = _mm_mul_ps (q1, zz1);
    q1 = _mm_add_ps (q1, ASIN_S4_PS);
    p1 = _mm_mul_ps (zz1, _mm_div_ps (p1, q1));
    zz1 = _mm_sqrt_ps (_mm_add_ps (zz1, zz1));
    __m128 z1 = _mm_sub_ps (PI_OVER_FOUR_PS, zz1);
    zz1 = _mm_sub_ps (_mm_mul_ps (zz1, p1), ASIN_MOREBITS_PS);
    z1 = _mm_sub_ps (z1, zz1);
    z1 = _mm_add_ps (z1, PI_OVER_FOUR_PS);
    
    __m128 zz2 = _mm_mul_ps (abs_x, abs_x);
    __m128 p2 = ASIN_P0_PS;
    p2 = _mm_mul_ps (p2, zz2);
    p2 = _mm_add_ps (p2, ASIN_P1_PS);
    p2 = _mm_mul_ps (p2, zz2);
    p2 = _mm_add_ps (p2, ASIN_P2_PS);
    p2 = _mm_mul_ps (p2, zz2);
    p2 = _mm_add_ps (p2, ASIN_P3_PS);
    p2 = _mm_mul_ps (p2, zz2);
    p2 = _mm_add_ps (p2, ASIN_P4_PS);
    p2 = _mm_mul_ps (p2, zz2);
    p2 = _mm_add_ps (p2, ASIN_P5_PS);
    __m128 q2 = zz2;
    q2 = _mm_add_ps (q2, ASIN_Q1_PS);
    q2 = _mm_mul_ps (q2, zz2);
    q2 = _mm_add_ps (q2, ASIN_Q2_PS);
    q2 = _mm_mul_ps (q2, zz2);
    q2 = _mm_add_ps (q2, ASIN_Q3_PS);
    q2 = _mm_mul_ps (q2, zz2);
    q2 = _mm_add_ps (q2, ASIN_Q4_PS);
    q2 = _mm_mul_ps (q2, zz2);
    q2 = _mm_add_ps (q2, ASIN_Q5_PS);
    __m128 z2 = _mm_mul_ps (zz2, _mm_div_ps (p2, q2));
    z2 = _mm_add_ps (_mm_mul_ps (abs_x, z2), abs_x);
    
    return _mm_or_ps (
        _mm_xor_ps (
            sel_ps (
                z1,
                sel_ps (abs_x, z2, _mm_cmplt_ps (ASIN_BRANCH2_PS, abs_x)),
                _mm_cmpgt_ps (ASIN_BRANCH1_PS, abs_x)),
            sign_bit
        ),
        invalid_mask
    );
}

__m128d asin_pd (__m128d x)
{
    __m128d sign_bit = _mm_and_pd (x, (__m128d) SIGN_MASK_DOUBLE);
    __m128d abs_x = abs_pd (x);
    __m128d invalid_mask = _mm_cmpgt_pd (abs_x, ONE_PD);

    __m128d zz1 = _mm_sub_pd (ONE_PD, abs_x);
    __m128d p1 = ASIN_R0_PD;
    p1 = _mm_mul_pd (p1, zz1);
    p1 = _mm_add_pd (p1, ASIN_R1_PD);
    p1 = _mm_mul_pd (p1, zz1);
    p1 = _mm_add_pd (p1, ASIN_R2_PD);
    p1 = _mm_mul_pd (p1, zz1);
    p1 = _mm_add_pd (p1, ASIN_R3_PD);
    p1 = _mm_mul_pd (p1, zz1);
    p1 = _mm_add_pd (p1, ASIN_R4_PD);
    __m128d q1 = zz1;
    q1 = _mm_add_pd (q1, ASIN_S1_PD);
    q1 = _mm_mul_pd (q1, zz1);
    q1 = _mm_add_pd (q1, ASIN_S2_PD);
    q1 = _mm_mul_pd (q1, zz1);
    q1 = _mm_add_pd (q1, ASIN_S3_PD);
    q1 = _mm_mul_pd (q1, zz1);
    q1 = _mm_add_pd (q1, ASIN_S4_PD);
    p1 = _mm_mul_pd (zz1, _mm_div_pd (p1, q1));
    zz1 = _mm_sqrt_pd (_mm_add_pd (zz1, zz1));
    __m128d z1 = _mm_sub_pd (PI_OVER_FOUR_PD, zz1);
    zz1 = _mm_sub_pd (_mm_mul_pd (zz1, p1), ASIN_MOREBITS_PD);
    z1 = _mm_sub_pd (z1, zz1);
    z1 = _mm_add_pd (z1, PI_OVER_FOUR_PD);
    
    __m128d zz2 = _mm_mul_pd (abs_x, abs_x);
    __m128d p2 = ASIN_P0_PD;
    p2 = _mm_mul_pd (p2, zz2);
    p2 = _mm_add_pd (p2, ASIN_P1_PD);
    p2 = _mm_mul_pd (p2, zz2);
    p2 = _mm_add_pd (p2, ASIN_P2_PD);
    p2 = _mm_mul_pd (p2, zz2);
    p2 = _mm_add_pd (p2, ASIN_P3_PD);
    p2 = _mm_mul_pd (p2, zz2);
    p2 = _mm_add_pd (p2, ASIN_P4_PD);
    p2 = _mm_mul_pd (p2, zz2);
    p2 = _mm_add_pd (p2, ASIN_P5_PD);
    __m128d q2 = zz2;
    q2 = _mm_add_pd (q2, ASIN_Q1_PD);
    q2 = _mm_mul_pd (q2, zz2);
    q2 = _mm_add_pd (q2, ASIN_Q2_PD);
    q2 = _mm_mul_pd (q2, zz2);
    q2 = _mm_add_pd (q2, ASIN_Q3_PD);
    q2 = _mm_mul_pd (q2, zz2);
    q2 = _mm_add_pd (q2, ASIN_Q4_PD);
    q2 = _mm_mul_pd (q2, zz2);
    q2 = _mm_add_pd (q2, ASIN_Q5_PD);
    __m128d z2 = _mm_mul_pd (zz2, _mm_div_pd (p2, q2));
    z2 = _mm_add_pd (_mm_mul_pd (abs_x, z2), abs_x);
    
    return _mm_or_pd (
        _mm_xor_pd (
            sel_pd (
                z1,
                sel_pd (abs_x, z2, _mm_cmplt_pd (ASIN_BRANCH2_PD, abs_x)),
                _mm_cmpgt_pd (ASIN_BRANCH1_PD, abs_x)),
            sign_bit
        ),
        invalid_mask
    );
}

__m128 acos_ps (__m128 x)
{
    __m128 invalid_mask = _mm_cmpgt_ps (abs_ps (x), ONE_PS);

    __m128 y1 = _mm_mul_ps (TWO_PS, asin_ps (_mm_sqrt_ps (_mm_sub_ps (HALF_PS, _mm_mul_ps (HALF_PS, x)))));
    __m128 y2 = _mm_sub_ps (PI_OVER_FOUR_PS, asin_ps (x));
    y2 = _mm_add_ps (y2, ASIN_MOREBITS_PS);
    y2 = _mm_add_ps (y2, PI_OVER_FOUR_PS);
    
    return _mm_or_ps (sel_ps (y1, y2, _mm_cmpgt_ps (HALF_PS, x)), invalid_mask);
}

__m128d acos_pd (__m128d x)
{
    __m128d invalid_mask = _mm_cmpgt_pd (abs_pd (x), ONE_PD);

    __m128d y1 = _mm_mul_pd (TWO_PD, asin_pd (_mm_sqrt_pd (_mm_sub_pd (HALF_PD, _mm_mul_pd (HALF_PD, x)))));
    __m128d y2 = _mm_sub_pd (PI_OVER_FOUR_PD, asin_pd (x));
    y2 = _mm_add_pd (y2, ASIN_MOREBITS_PD);
    y2 = _mm_add_pd (y2, PI_OVER_FOUR_PD);
    
    return _mm_or_pd (sel_pd (y1, y2, _mm_cmpgt_pd (HALF_PD, x)), invalid_mask);
}

__m128 atan_ps (__m128 x)
{
    __m128 sign_bit = _mm_and_ps (x, (__m128) SIGN_MASK_SINGLE);
    x = abs_ps (x);

    // range reduction
    __m128 mask_range_large = _mm_cmpgt_ps (THREE_PI_OVER_EIGHT_PS, x);
    __m128 mask_range_small = _mm_cmple_ps (ATAN_BRANCH_PS, x);
    
    x = sel_ps (
        _mm_div_ps (NEGONE_PS, x),                                          // if x > THREE_PI_OVER_EIGHT_PS
        sel_ps (
            x,                                                              // else if x <= 0.66
            _mm_div_ps (_mm_sub_ps (x, ONE_PS), _mm_add_ps (x, ONE_PS)),    // else
            mask_range_small
        ), mask_range_large);
        
    __m128 y = sel_ps (
        PI_OVER_TWO_PS,         // if x > THREE_PI_OVER_EIGHT_PS
        sel_ps (
            _mm_setzero_ps (),  // else if x <= 0.66
            PI_OVER_FOUR_PS,    // else
            mask_range_small
        ), mask_range_large);
    
    // evaluate polynomials
    __m128 z = _mm_mul_ps (x, x);
    __m128 p = ATAN_P0_PS;
    p = _mm_mul_ps (p, z);
    p = _mm_add_ps (p, ATAN_P1_PS);
    p = _mm_mul_ps (p, z);
    p = _mm_add_ps (p, ATAN_P2_PS);
    p = _mm_mul_ps (p, z);
    p = _mm_add_ps (p, ATAN_P3_PS);
    p = _mm_mul_ps (p, z);
    p = _mm_add_ps (p, ATAN_P4_PS);
    __m128 q = z;
    q = _mm_add_ps (q, ATAN_Q1_PS);
    q = _mm_mul_ps (q, z);
    q = _mm_add_ps (q, ATAN_Q2_PS);
    q = _mm_mul_ps (q, z);
    q = _mm_add_ps (q, ATAN_Q3_PS);
    q = _mm_mul_ps (q, z);
    q = _mm_add_ps (q, ATAN_Q4_PS);
    q = _mm_mul_ps (q, z);
    q = _mm_add_ps (q, ATAN_Q5_PS);
    z = _mm_mul_ps (z, _mm_div_ps (p, q));
    z = _mm_add_ps (_mm_mul_ps (x, z), x);
    
    z = sel_ps (
        _mm_add_ps (z, ASIN_MOREBITS_PS),                            // if x > THREE_PI_OVER_EIGHT_PS [flag=1]
        sel_ps (
            z,                                                       // else if x <= 0.66 [flag=0]
            _mm_add_ps (z, _mm_mul_ps (HALF_PS, ASIN_MOREBITS_PS)),  // else [flag=2]
            mask_range_small
        ), mask_range_large);
        
    return _mm_xor_ps (_mm_add_ps (y, z), sign_bit);
}

__m128d atan_pd (__m128d x)
{
    __m128d sign_bit = _mm_and_pd (x, (__m128d) SIGN_MASK_DOUBLE);
    x = abs_pd (x);

    // range reduction
    __m128d mask_range_large = _mm_cmpgt_pd (THREE_PI_OVER_EIGHT_PD, x);
    __m128d mask_range_small = _mm_cmple_pd (ATAN_BRANCH_PD, x);
    
    x = sel_pd (
        _mm_div_pd (NEGONE_PD, x),                                          // if x > THREE_PI_OVER_EIGHT_PS
        sel_pd (
            x,                                                              // else if x <= 0.66
            _mm_div_pd (_mm_sub_pd (x, ONE_PD), _mm_add_pd (x, ONE_PD)),    // else
            mask_range_small
        ), mask_range_large);
        
    __m128d y = sel_pd (
        PI_OVER_TWO_PD,         // if x > THREE_PI_OVER_EIGHT_PS
        sel_pd (
            _mm_setzero_pd (),  // else if x <= 0.66
            PI_OVER_FOUR_PD,    // else
            mask_range_small
        ), mask_range_large);
    
    // evaluate polynomials
    __m128d z = _mm_mul_pd (x, x);
    __m128d p = ATAN_P0_PD;
    p = _mm_mul_pd (p, z);
    p = _mm_add_pd (p, ATAN_P1_PD);
    p = _mm_mul_pd (p, z);
    p = _mm_add_pd (p, ATAN_P2_PD);
    p = _mm_mul_pd (p, z);
    p = _mm_add_pd (p, ATAN_P3_PD);
    p = _mm_mul_pd (p, z);
    p = _mm_add_pd (p, ATAN_P4_PD);
    __m128d q = z;
    q = _mm_add_pd (q, ATAN_Q1_PD);
    q = _mm_mul_pd (q, z);
    q = _mm_add_pd (q, ATAN_Q2_PD);
    q = _mm_mul_pd (q, z);
    q = _mm_add_pd (q, ATAN_Q3_PD);
    q = _mm_mul_pd (q, z);
    q = _mm_add_pd (q, ATAN_Q4_PD);
    q = _mm_mul_pd (q, z);
    q = _mm_add_pd (q, ATAN_Q5_PD);
    z = _mm_mul_pd (z, _mm_div_pd (p, q));
    z = _mm_add_pd (_mm_mul_pd (x, z), x);
    
    z = sel_pd (
        _mm_add_pd (z, ASIN_MOREBITS_PD),                            // if x > THREE_PI_OVER_EIGHT_PS [flag=1]
        sel_pd (
            z,                                                       // else if x <= 0.66 [flag=0]
            _mm_add_pd (z, _mm_mul_pd (HALF_PD, ASIN_MOREBITS_PD)),  // else [flag=2]
            mask_range_small
        ), mask_range_large);
        
    return _mm_xor_pd (_mm_add_pd (y, z), sign_bit);
}

__m128 atan2_ps (__m128 y, __m128 x)
{
    __m128 z = atan_ps (_mm_div_ps (y, x));
    __m128 mask_x_pos_zero = _mm_cmpeq_ps (x, _mm_setzero_ps ());
    
    return sel_ps (
        // x != 0
        sel_ps (
            z,                                          // atan(y/x) if x > 0
            sel_ps (
                _mm_add_ps (z, PI_PS),                  // PI+atan(y/x) if y >= 0, x < 0
                _mm_sub_ps (z, PI_PS),                  // -PI+atan(y/x) if y < 0, x < 0
                _mm_cmpge_ps (_mm_setzero_ps (), y)),
            _mm_cmpgt_ps (_mm_setzero_ps (), x)),
            
        // x == 0
        sel_ps (
            sel_ps (PI_OVER_TWO_PS, NEG_PI_OVER_TWO_PS, _mm_cmpgt_ps (_mm_setzero_ps (), y)),   // +/- PI/2 if x=0, y >/< 0

            // atan(+0,+0)=+0, atan(+0,-0)=+PI, atan(-0,+0)=-0, atan(-0,-0)=-PI
            sel_ps (
                // y == 0x80000000 <=> y = -0
                sel_ps (NEG_PI_PS, NEG_ZERO_PS, mask_x_pos_zero),
                // y == 0x00000000 <=> y = +0
                sel_ps (PI_PS, _mm_setzero_ps (), mask_x_pos_zero),
                _mm_cmpeq_ps (y, _mm_setzero_ps ())),
            _mm_cmpeq_ps (y, _mm_setzero_ps ())),
        _mm_cmpeq_ps (x, _mm_setzero_ps ())
    );
}

__m128d atan2_pd (__m128d y, __m128d x)
{
    __m128d z = atan_pd (_mm_div_pd (y, x));
    __m128d mask_x_pos_zero = _mm_cmpeq_pd (x, _mm_setzero_pd ());
    
    return sel_pd (
        // x != 0
        sel_pd (
            z,                                          // atan(y/x) if x > 0
            sel_pd (
                _mm_add_pd (z, PI_PD),                  // PI+atan(y/x) if y >= 0, x < 0
                _mm_sub_pd (z, PI_PD),                  // -PI+atan(y/x) if y < 0, x < 0
                _mm_cmpge_pd (_mm_setzero_pd (), y)),
            _mm_cmpgt_pd (_mm_setzero_pd (), x)),
            
        // x == 0
        sel_pd (
            sel_pd (PI_OVER_TWO_PD, NEG_PI_OVER_TWO_PD, _mm_cmpgt_pd (_mm_setzero_pd (), y)),   // +/- PI/2 if x=0, y >/< 0

            // atan(+0,+0)=+0, atan(+0,-0)=+PI, atan(-0,+0)=-0, atan(-0,-0)=-PI
            sel_pd (
                // y == 0x80000000 <=> y = -0
                sel_pd (NEG_PI_PD, NEG_ZERO_PD, mask_x_pos_zero),
                // y == 0x00000000 <=> y = +0
                sel_pd (PI_PD, _mm_setzero_pd (), mask_x_pos_zero),
                _mm_cmpeq_pd (y, _mm_setzero_pd ())),
            _mm_cmpeq_pd (y, _mm_setzero_pd ())),
        _mm_cmpeq_pd (x, _mm_setzero_pd ())
    );
}

