#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

void print_vector_ps (char* szLabel, __m256 v)
{
    float a[8];
    __asm__ (
        "vmovups (%0),%%ymm0\n\t"
        "vmovups %%ymm0,(%1)\n\t"
        :
        : "r"(&v), "r"(a)
    );
    printf ("%s: %f, %f, %f, %f, %f, %f, %f, %f\n", szLabel,
        a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
    );
}

int main (int argc, char** argv)
{
    __m256 expr1 = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __m256 expr2 = { 8, 9, 10, 11, 12, 13, 14, 15 };
    
    print_vector_ps ("1", (__m256) _mm256_shuffle_pd ((__m256d) expr1, (__m256d) _mm256_permute2f128_ps (expr1, expr2, 33), 5));
    
    return 0;
}
