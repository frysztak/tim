#ifndef SIMD_MATH
#define SIMD_MATH

#include <emmintrin.h>

inline __m128 exp_approx_ps(__m128 x)
{
    // this approximation is not meant for general use.
    // it closely approximates exp(x) for a very small range of x = [-1, 0].
    // y = 0.30883*x^2 + 0.93096*x + 0.99466
    
    __m128 y = _mm_set1_ps(9.94663531855e-01);
    __m128 constant = _mm_set1_ps(9.30963170380e-01);
    y = _mm_add_ps(y, _mm_mul_ps(constant, x));
    constant = _mm_set1_ps(3.08826533369e-01);
    y = _mm_add_ps(y, _mm_mul_ps(constant, _mm_mul_ps(x, x)));

    return y;
}

inline __m128 log_approx_ps(__m128 x)
{
    // this approximation is not meant for general use.
    // it closely approximates 1.5*log(x) for a very small range of x = [4, 5].
    // y = -0.0372383*x^2 + 0.6693217*x - 0.0017514 
    
    __m128 y = _mm_set1_ps(-1.75139092193e-03);
    __m128 constant = _mm_set1_ps(6.69321654748e-01);
    y = _mm_add_ps(y, _mm_mul_ps(constant, x));
    constant = _mm_set1_ps(-3.72382626847e-02);
    y = _mm_add_ps(y, _mm_mul_ps(constant, _mm_mul_ps(x, x)));

    return y;
}


#endif

/* vim: set ft=cpp ts=4 sw=4 sts=4 tw=0 fenc=utf-8 et: */
