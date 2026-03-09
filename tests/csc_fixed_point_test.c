/*
 * CSC (Color Space Conversion) Fixed-Point Verification Test
 *
 * Implements the core CSC color gamut conversion (RGB->XYZ->RGB)
 * using fixed-point integer arithmetic (no double), and verifies
 * that the maximum pixel error does not exceed +/-1 compared to
 * a float-based reference implementation.
 *
 * Pipeline:
 *   RGB_src -> [RGB2XYZ_src] -> XYZ -> [adapt] -> [XYZ2RGB_dst] -> RGB_dst
 *
 * The combined conversion is a single 3x3 matrix multiply:
 *   [R_dst]   [m00 m01 m02] [R_src]
 *   [G_dst] = [m10 m11 m12] [G_src]
 *   [B_dst]   [m20 m21 m22] [B_src]
 *
 * Fixed-point pixel processing (same as FFmpeg multiply3x3):
 *   result = clip((coeff[0]*v0 + coeff[1]*v1 + coeff[2]*v2 + 8192) >> 14)
 *
 * Usage: ./csc_fixed_point_test
 *
 * Copyright (c) 2024 FFmpeg project
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* ================================================================
 * Fixed-point configuration
 * ================================================================
 * Q28: 28 fractional bits for matrix computation (~1e-8 precision)
 * Q14: 14 fractional bits for pixel processing coefficients
 * Pixel range: [0, 28672] maps to [0.0, 1.0] (same as FFmpeg)
 */
#define FP_BITS   28
#define FP_ONE    (1LL << FP_BITS)
#define FP_HALF   (1LL << (FP_BITS - 1))

#define COEFF_BITS 14
#define COEFF_HALF (1 << (COEFF_BITS - 1))

#define PIXEL_SCALE 28672

/* ================================================================
 * Integer-only rational type for color primary storage
 * ================================================================ */
typedef struct {
    int32_t num, den;
} Rat;

/* ================================================================
 * Fixed-point 3x3 matrix operations (Q20 format, int64_t)
 * ================================================================ */
typedef struct {
    int64_t m[3][3];
} FPMat3;

/* In-place matrix inversion using adjugate method.
 * Input/output in Q28 format. Uses __int128 for all intermediate
 * computations to avoid overflow (Q28*Q28=Q56 exceeds int64_t). */
static void fp_mat3_invert(FPMat3 *mat)
{
#ifdef __SIZEOF_INT128__
    __int128 m00 = mat->m[0][0], m01 = mat->m[0][1], m02 = mat->m[0][2];
    __int128 m10 = mat->m[1][0], m11 = mat->m[1][1], m12 = mat->m[1][2];
    __int128 m20 = mat->m[2][0], m21 = mat->m[2][1], m22 = mat->m[2][2];

    /* Cofactors in Q56 (products of two Q28 values) */
    __int128 a00 =  (m11 * m22 - m21 * m12);
    __int128 a01 = -(m01 * m22 - m21 * m02);
    __int128 a02 =  (m01 * m12 - m11 * m02);
    __int128 a10 = -(m10 * m22 - m20 * m12);
    __int128 a11 =  (m00 * m22 - m20 * m02);
    __int128 a12 = -(m00 * m12 - m10 * m02);
    __int128 a20 =  (m10 * m21 - m20 * m11);
    __int128 a21 = -(m00 * m21 - m20 * m01);
    __int128 a22 =  (m00 * m11 - m10 * m01);

    /* Determinant in Q84 = Q28 * Q56 */
    __int128 det_q84 = m00 * a00 + m10 * a01 + m20 * a02;

    /* inverse[i][j] = cofactor[i][j] / det
     * inverse_Q28 = cofactor_Q56 * 2^56 / det_Q84 */
    __int128 scale = ((__int128)1) << 56;
    mat->m[0][0] = (int64_t)(a00 * scale / det_q84);
    mat->m[0][1] = (int64_t)(a01 * scale / det_q84);
    mat->m[0][2] = (int64_t)(a02 * scale / det_q84);
    mat->m[1][0] = (int64_t)(a10 * scale / det_q84);
    mat->m[1][1] = (int64_t)(a11 * scale / det_q84);
    mat->m[1][2] = (int64_t)(a12 * scale / det_q84);
    mat->m[2][0] = (int64_t)(a20 * scale / det_q84);
    mat->m[2][1] = (int64_t)(a21 * scale / det_q84);
    mat->m[2][2] = (int64_t)(a22 * scale / det_q84);
#else
#error "__int128 required for Q28 matrix inversion"
#endif
}

/* Matrix multiply: a = a * b (both Q28)
 * Uses __int128 for Q28*Q28=Q56 products that exceed int64_t range */
static void fp_mat3_mul(FPMat3 *a, const FPMat3 *b)
{
#ifdef __SIZEOF_INT128__
    __int128 a00 = a->m[0][0], a01 = a->m[0][1], a02 = a->m[0][2];
    __int128 a10 = a->m[1][0], a11 = a->m[1][1], a12 = a->m[1][2];
    __int128 a20 = a->m[2][0], a21 = a->m[2][1], a22 = a->m[2][2];
    int i;

    for (i = 0; i < 3; i++) {
        __int128 bi0 = b->m[0][i], bi1 = b->m[1][i], bi2 = b->m[2][i];
        a->m[0][i] = (int64_t)((a00 * bi0 + a01 * bi1 + a02 * bi2
                                 + FP_HALF) >> FP_BITS);
        a->m[1][i] = (int64_t)((a10 * bi0 + a11 * bi1 + a12 * bi2
                                 + FP_HALF) >> FP_BITS);
        a->m[2][i] = (int64_t)((a20 * bi0 + a21 * bi1 + a22 * bi2
                                 + FP_HALF) >> FP_BITS);
    }
#else
#error "__int128 required for Q28 matrix multiplication"
#endif
}

/* ================================================================
 * CIE chromaticity data structures
 * ================================================================ */
typedef struct {
    Rat x, y;
} CIExy;

typedef struct {
    CIExy r, g, b;
} PrimCoeffs;

typedef struct {
    CIExy wp;
    PrimCoeffs prim;
} ColorPrimDesc;

/* ================================================================
 * Color primaries database
 * (same values as FFmpeg libavutil/csp.c, encoded as rationals)
 * ================================================================ */
/* Macro to create rational from decimal value (matching FFmpeg's AVR macro).
 * Uses integer arithmetic: R(0.640) -> {64000, 100000} */
#define R(int_part, frac_5digits) { (int_part) * 100000 + (frac_5digits), 100000 }
/* Helper for values < 1.0 with up to 5 decimal digits */
#define R0(frac) { frac, 100000 }
/* Helper for values specified as numerator/denominator */
#define RQ(n, d) { n, d }

static const ColorPrimDesc prim_bt709 = {
    .wp   = { R0(31270), R0(32900) },
    .prim = {
        { R0(64000), R0(33000) },
        { R0(30000), R0(60000) },
        { R0(15000), R0( 6000) },
    },
};

static const ColorPrimDesc prim_bt2020 = {
    .wp   = { R0(31270), R0(32900) },
    .prim = {
        { R0(70800), R0(29200) },
        { R0(17000), R0(79700) },
        { R0(13100), R0( 4600) },
    },
};

static const ColorPrimDesc prim_smpte170m = {
    .wp   = { R0(31270), R0(32900) },
    .prim = {
        { R0(63000), R0(34000) },
        { R0(31000), R0(59500) },
        { R0(15500), R0( 7000) },
    },
};

static const ColorPrimDesc prim_bt470bg = {
    .wp   = { R0(31270), R0(32900) },
    .prim = {
        { R0(64000), R0(33000) },
        { R0(29000), R0(60000) },
        { R0(15000), R0( 6000) },
    },
};

static const ColorPrimDesc prim_smpte432 = {  /* Display P3 */
    .wp   = { R0(31270), R0(32900) },
    .prim = {
        { R0(68000), R0(32000) },
        { R0(26500), R0(69000) },
        { R0(15000), R0( 6000) },
    },
};

static const ColorPrimDesc prim_bt470m = {
    .wp   = { R0(31000), R0(31600) },
    .prim = {
        { R0(67000), R0(33000) },
        { R0(21000), R0(71000) },
        { R0(14000), R0( 8000) },
    },
};

/* ================================================================
 * Fixed-point RGB <-> XYZ matrix computation
 * ================================================================ */

/* Compute X/Y from CIE xy: X = x/y (result in Q28)
 * Uses int64_t to avoid overflow with 100000-denominator rationals */
static int64_t cie_X_fp(CIExy xy)
{
    int64_t num = (int64_t)xy.x.num * xy.y.den;
    int64_t den = (int64_t)xy.x.den * xy.y.num;
    return (num << FP_BITS) / den;
}

/* Compute Z/Y from CIE xy: Z = (1-x-y)/y (result in Q28)
 * Uses int64_t to avoid overflow */
static int64_t cie_Z_fp(CIExy xy)
{
    int64_t z_num = (int64_t)xy.x.den * xy.y.den
                  - (int64_t)xy.x.num * xy.y.den
                  - (int64_t)xy.y.num * xy.x.den;
    int64_t z_den = (int64_t)xy.x.den * xy.y.num;
    return (z_num << FP_BITS) / z_den;
}

/* Compute RGB->XYZ matrix in Q28 fixed-point */
static FPMat3 fp_rgb2xyz(const ColorPrimDesc *desc)
{
    FPMat3 out;
    int64_t S[3], X[3], Z[3], Xw, Zw;
    int i;

    memset(&out, 0, sizeof(out));

    X[0] = cie_X_fp(desc->prim.r);
    X[1] = cie_X_fp(desc->prim.g);
    X[2] = cie_X_fp(desc->prim.b);

    Z[0] = cie_Z_fp(desc->prim.r);
    Z[1] = cie_Z_fp(desc->prim.g);
    Z[2] = cie_Z_fp(desc->prim.b);

    Xw = cie_X_fp(desc->wp);
    Zw = cie_Z_fp(desc->wp);

    /* Build [X; 1; Z] matrix */
    for (i = 0; i < 3; i++) {
        out.m[0][i] = X[i];
        out.m[1][i] = FP_ONE;
        out.m[2][i] = Z[i];
    }

    /* Invert to get scaling factors: S = [X;1;Z]^-1 * [Xw;1;Zw] */
    fp_mat3_invert(&out);

    /* S[i] = row_i of inverse * [Xw, 1, Zw]
     * Use __int128 for Q28*Q28 = Q56 products */
    for (i = 0; i < 3; i++)
        S[i] = (int64_t)(((__int128)out.m[i][0] * Xw +
                           (__int128)out.m[i][1] * FP_ONE +
                           (__int128)out.m[i][2] * Zw + FP_HALF) >> FP_BITS);

    /* Build final matrix: M = [S*X; S; S*Z] */
    for (i = 0; i < 3; i++) {
        out.m[0][i] = (int64_t)(((__int128)S[i] * X[i] + FP_HALF) >> FP_BITS);
        out.m[1][i] = S[i];
        out.m[2][i] = (int64_t)(((__int128)S[i] * Z[i] + FP_HALF) >> FP_BITS);
    }

    return out;
}

/* Compute XYZ->RGB matrix in Q28 fixed-point */
static FPMat3 fp_xyz2rgb(const ColorPrimDesc *desc)
{
    FPMat3 out = fp_rgb2xyz(desc);
    fp_mat3_invert(&out);
    return out;
}

/* ================================================================
 * Bradford chromatic adaptation in Q20 fixed-point
 * ================================================================ */

/* Bradford matrix in Q28 (pre-computed integer constants, no float)
 * Values: round(coefficient * 268435456)
 *   0.8951 * 2^28 = 240284570,  0.2664 * 2^28 = 71508940, -0.1614 * 2^28 = -43325347
 *  -0.7502 * 2^28 = -201372089, 1.7135 * 2^28 = 459956630, 0.0367 * 2^28 = 9851582
 *   0.0389 * 2^28 = 10442131,  -0.0685 * 2^28 = -18387840, 1.0296 * 2^28 = 276379796
 */
static const FPMat3 bradford_mat = {{
    {  240284570,   71508940,  -43325347 },
    { -201372089,  459956630,    9851582 },
    {   10442131,  -18387840,  276379796 },
}};

/* Compute chromatic adaptation matrix (Bradford method)
 * Result: xyz2rgb_dst := xyz2rgb_dst * M_adapt
 * where M_adapt converts from src white point XYZ to dst white point XYZ */
static void fp_apply_chromatic_adaptation(CIExy wp_src, CIExy wp_dst,
                                          FPMat3 *mat)
{
    FPMat3 brad = bradford_mat;
    FPMat3 brad_inv = bradford_mat;
    FPMat3 diag;
    int64_t srcX, srcZ, dstX, dstZ;
    int64_t src_cone[3], dst_cone[3];
    int i;

    /* Check if white points are the same */
    if (wp_src.x.num * wp_dst.x.den == wp_dst.x.num * wp_src.x.den &&
        wp_src.y.num * wp_dst.y.den == wp_dst.y.num * wp_src.y.den)
        return;

    srcX = cie_X_fp(wp_src);
    srcZ = cie_Z_fp(wp_src);
    dstX = cie_X_fp(wp_dst);
    dstZ = cie_Z_fp(wp_dst);

    /* Invert Bradford matrix */
    fp_mat3_invert(&brad_inv);

    /* Compute source and destination cone responses
     * Use __int128 for Q28*Q28 = Q56 products */
    for (i = 0; i < 3; i++) {
        src_cone[i] = (int64_t)(((__int128)brad.m[i][0] * srcX +
                                  (__int128)brad.m[i][1] * FP_ONE +
                                  (__int128)brad.m[i][2] * srcZ +
                                  FP_HALF) >> FP_BITS);
        dst_cone[i] = (int64_t)(((__int128)brad.m[i][0] * dstX +
                                  (__int128)brad.m[i][1] * FP_ONE +
                                  (__int128)brad.m[i][2] * dstZ +
                                  FP_HALF) >> FP_BITS);
    }

    /* Build diagonal scaling matrix: diag[i][i] = dst_cone[i] / src_cone[i]
     * Result in Q28 */
    memset(&diag, 0, sizeof(diag));
    for (i = 0; i < 3; i++)
        diag.m[i][i] = ((__int128)dst_cone[i] * FP_ONE) / src_cone[i];

    /* tmp = diag * brad */
    fp_mat3_mul(&diag, &brad);

    /* mat = mat * brad_inv * diag_brad */
    fp_mat3_mul(mat, &brad_inv);
    fp_mat3_mul(mat, &diag);
}

/* ================================================================
 * Compute the full RGB->RGB gamut conversion matrix (fixed-point)
 * ================================================================ */
static FPMat3 fp_compute_rgb2rgb(const ColorPrimDesc *src,
                                  const ColorPrimDesc *dst)
{
    FPMat3 rgb2xyz_src = fp_rgb2xyz(src);
    FPMat3 xyz2rgb_dst = fp_xyz2rgb(dst);

    /* Apply chromatic adaptation if white points differ */
    fp_apply_chromatic_adaptation(src->wp, dst->wp, &xyz2rgb_dst);

    /* Combined: xyz2rgb_dst * rgb2xyz_src */
    fp_mat3_mul(&xyz2rgb_dst, &rgb2xyz_src);

    return xyz2rgb_dst;
}

/* Quantize Q28 matrix to Q14 int16_t coefficients (same as FFmpeg) */
static void fp_quantize_coeffs(const FPMat3 *mat, int16_t coeffs[3][3])
{
    int i, j;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++) {
            /* Convert Q28 to real value * 16384:
             * coeff = round(mat_q28 / 2^28 * 16384)
             *       = round(mat_q28 / 2^14)
             *       = (mat_q28 + 2^13) >> 14 */
            int64_t val = (mat->m[i][j] + (1LL << (FP_BITS - COEFF_BITS - 1)))
                          >> (FP_BITS - COEFF_BITS);
            if (val > 32767)  val = 32767;
            if (val < -32768) val = -32768;
            coeffs[i][j] = (int16_t)val;
        }
}

/* ================================================================
 * Fixed-point pixel processing (same as FFmpeg multiply3x3)
 * ================================================================ */
static void fp_apply_matrix(const int16_t m[3][3],
                             int16_t r, int16_t g, int16_t b,
                             int16_t *ro, int16_t *go, int16_t *bo)
{
    int32_t v0 = r, v1 = g, v2 = b;
    int32_t out0 = (m[0][0] * v0 + m[0][1] * v1 + m[0][2] * v2 + COEFF_HALF) >> COEFF_BITS;
    int32_t out1 = (m[1][0] * v0 + m[1][1] * v1 + m[1][2] * v2 + COEFF_HALF) >> COEFF_BITS;
    int32_t out2 = (m[2][0] * v0 + m[2][1] * v1 + m[2][2] * v2 + COEFF_HALF) >> COEFF_BITS;

    /* Clip to int16_t range (same as FFmpeg av_clip_int16) */
    *ro = (int16_t)(out0 > 32767 ? 32767 : (out0 < -32768 ? -32768 : out0));
    *go = (int16_t)(out1 > 32767 ? 32767 : (out1 < -32768 ? -32768 : out1));
    *bo = (int16_t)(out2 > 32767 ? 32767 : (out2 < -32768 ? -32768 : out2));
}

/* ================================================================
 * Float reference implementation (for comparison only)
 * Note: uses float, NOT double (double is prohibited)
 * ================================================================ */
typedef struct {
    float m[3][3];
} FloatMat3;

static FloatMat3 ref_rgb2xyz(const ColorPrimDesc *desc)
{
    FloatMat3 out;
    float X[3], Z[3], S[3], Xw, Zw;
    int i;

    for (i = 0; i < 3; i++) {
        const CIExy *p = (i == 0) ? &desc->prim.r :
                          (i == 1) ? &desc->prim.g : &desc->prim.b;
        X[i] = (float)p->x.num / p->x.den / ((float)p->y.num / p->y.den);
        Z[i] = (1.0f - (float)p->x.num / p->x.den - (float)p->y.num / p->y.den)
               / ((float)p->y.num / p->y.den);
    }
    Xw = (float)desc->wp.x.num / desc->wp.x.den /
         ((float)desc->wp.y.num / desc->wp.y.den);
    Zw = (1.0f - (float)desc->wp.x.num / desc->wp.x.den -
          (float)desc->wp.y.num / desc->wp.y.den) /
         ((float)desc->wp.y.num / desc->wp.y.den);

    for (i = 0; i < 3; i++) {
        out.m[0][i] = X[i];
        out.m[1][i] = 1.0f;
        out.m[2][i] = Z[i];
    }

    /* Invert */
    {
        float m00 = out.m[0][0], m01 = out.m[0][1], m02 = out.m[0][2];
        float m10 = out.m[1][0], m11 = out.m[1][1], m12 = out.m[1][2];
        float m20 = out.m[2][0], m21 = out.m[2][1], m22 = out.m[2][2];
        float a00 =  (m11*m22 - m21*m12), a01 = -(m01*m22 - m21*m02);
        float a02 =  (m01*m12 - m11*m02), a10 = -(m10*m22 - m20*m12);
        float a11 =  (m00*m22 - m20*m02), a12 = -(m00*m12 - m10*m02);
        float a20 =  (m10*m21 - m20*m11), a21 = -(m00*m21 - m20*m01);
        float a22 =  (m00*m11 - m10*m01);
        float det = m00*a00 + m10*a01 + m20*a02;
        det = 1.0f / det;
        out.m[0][0] = det*a00; out.m[0][1] = det*a01; out.m[0][2] = det*a02;
        out.m[1][0] = det*a10; out.m[1][1] = det*a11; out.m[1][2] = det*a12;
        out.m[2][0] = det*a20; out.m[2][1] = det*a21; out.m[2][2] = det*a22;
    }

    for (i = 0; i < 3; i++)
        S[i] = out.m[i][0] * Xw + out.m[i][1] * 1.0f + out.m[i][2] * Zw;

    for (i = 0; i < 3; i++) {
        out.m[0][i] = S[i] * X[i];
        out.m[1][i] = S[i];
        out.m[2][i] = S[i] * Z[i];
    }
    return out;
}

static void ref_mat3_invert(FloatMat3 *mat)
{
    float m00 = mat->m[0][0], m01 = mat->m[0][1], m02 = mat->m[0][2];
    float m10 = mat->m[1][0], m11 = mat->m[1][1], m12 = mat->m[1][2];
    float m20 = mat->m[2][0], m21 = mat->m[2][1], m22 = mat->m[2][2];
    float a00 =  (m11*m22 - m21*m12), a01 = -(m01*m22 - m21*m02);
    float a02 =  (m01*m12 - m11*m02), a10 = -(m10*m22 - m20*m12);
    float a11 =  (m00*m22 - m20*m02), a12 = -(m00*m12 - m10*m02);
    float a20 =  (m10*m21 - m20*m11), a21 = -(m00*m21 - m20*m01);
    float a22 =  (m00*m11 - m10*m01);
    float det = m00*a00 + m10*a01 + m20*a02;
    det = 1.0f / det;
    mat->m[0][0] = det*a00; mat->m[0][1] = det*a01; mat->m[0][2] = det*a02;
    mat->m[1][0] = det*a10; mat->m[1][1] = det*a11; mat->m[1][2] = det*a12;
    mat->m[2][0] = det*a20; mat->m[2][1] = det*a21; mat->m[2][2] = det*a22;
}

static void ref_mat3_mul(FloatMat3 *a, const FloatMat3 *b)
{
    float a00 = a->m[0][0], a01 = a->m[0][1], a02 = a->m[0][2];
    float a10 = a->m[1][0], a11 = a->m[1][1], a12 = a->m[1][2];
    float a20 = a->m[2][0], a21 = a->m[2][1], a22 = a->m[2][2];
    int i;
    for (i = 0; i < 3; i++) {
        a->m[0][i] = a00 * b->m[0][i] + a01 * b->m[1][i] + a02 * b->m[2][i];
        a->m[1][i] = a10 * b->m[0][i] + a11 * b->m[1][i] + a12 * b->m[2][i];
        a->m[2][i] = a20 * b->m[0][i] + a21 * b->m[1][i] + a22 * b->m[2][i];
    }
}

static void ref_apply_chromatic_adaptation(CIExy wp_src, CIExy wp_dst,
                                            FloatMat3 *mat)
{
    static const FloatMat3 brad = {{
        {  0.8951f,  0.2664f, -0.1614f },
        { -0.7502f,  1.7135f,  0.0367f },
        {  0.0389f, -0.0685f,  1.0296f },
    }};
    FloatMat3 brad_inv = brad;
    FloatMat3 diag;
    float srcX, srcZ, dstX, dstZ;
    float src_cone[3], dst_cone[3];
    int i;

    if (wp_src.x.num * wp_dst.x.den == wp_dst.x.num * wp_src.x.den &&
        wp_src.y.num * wp_dst.y.den == wp_dst.y.num * wp_src.y.den)
        return;

    srcX = (float)wp_src.x.num / wp_src.x.den / ((float)wp_src.y.num / wp_src.y.den);
    srcZ = (1.0f - (float)wp_src.x.num / wp_src.x.den -
            (float)wp_src.y.num / wp_src.y.den) / ((float)wp_src.y.num / wp_src.y.den);
    dstX = (float)wp_dst.x.num / wp_dst.x.den / ((float)wp_dst.y.num / wp_dst.y.den);
    dstZ = (1.0f - (float)wp_dst.x.num / wp_dst.x.den -
            (float)wp_dst.y.num / wp_dst.y.den) / ((float)wp_dst.y.num / wp_dst.y.den);

    ref_mat3_invert(&brad_inv);

    for (i = 0; i < 3; i++) {
        src_cone[i] = brad.m[i][0] * srcX + brad.m[i][1] * 1.0f + brad.m[i][2] * srcZ;
        dst_cone[i] = brad.m[i][0] * dstX + brad.m[i][1] * 1.0f + brad.m[i][2] * dstZ;
    }

    memset(&diag, 0, sizeof(diag));
    for (i = 0; i < 3; i++)
        diag.m[i][i] = dst_cone[i] / src_cone[i];

    ref_mat3_mul(&diag, &brad);
    ref_mat3_mul(mat, &brad_inv);
    ref_mat3_mul(mat, &diag);
}

static FloatMat3 ref_compute_rgb2rgb(const ColorPrimDesc *src,
                                      const ColorPrimDesc *dst)
{
    FloatMat3 rgb2xyz_src = ref_rgb2xyz(src);
    FloatMat3 xyz2rgb_dst = ref_rgb2xyz(dst);
    ref_mat3_invert(&xyz2rgb_dst);

    ref_apply_chromatic_adaptation(src->wp, dst->wp, &xyz2rgb_dst);
    ref_mat3_mul(&xyz2rgb_dst, &rgb2xyz_src);
    return xyz2rgb_dst;
}

static void ref_quantize_coeffs(const FloatMat3 *mat, int16_t coeffs[3][3])
{
    int i, j;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++) {
            int32_t val = (int32_t)(mat->m[i][j] * 16384.0f + (mat->m[i][j] >= 0 ? 0.5f : -0.5f));
            if (val > 32767)  val = 32767;
            if (val < -32768) val = -32768;
            coeffs[i][j] = (int16_t)val;
        }
}

static void ref_apply_matrix(const int16_t m[3][3],
                              int16_t r, int16_t g, int16_t b,
                              int16_t *ro, int16_t *go, int16_t *bo)
{
    int32_t v0 = r, v1 = g, v2 = b;
    int32_t out0 = (m[0][0] * v0 + m[0][1] * v1 + m[0][2] * v2 + COEFF_HALF) >> COEFF_BITS;
    int32_t out1 = (m[1][0] * v0 + m[1][1] * v1 + m[1][2] * v2 + COEFF_HALF) >> COEFF_BITS;
    int32_t out2 = (m[2][0] * v0 + m[2][1] * v1 + m[2][2] * v2 + COEFF_HALF) >> COEFF_BITS;
    *ro = (int16_t)(out0 > 32767 ? 32767 : (out0 < -32768 ? -32768 : out0));
    *go = (int16_t)(out1 > 32767 ? 32767 : (out1 < -32768 ? -32768 : out1));
    *bo = (int16_t)(out2 > 32767 ? 32767 : (out2 < -32768 ? -32768 : out2));
}

/* ================================================================
 * Test harness
 * ================================================================ */
typedef struct {
    const char *name;
    const ColorPrimDesc *desc;
} PrimEntry;

static const PrimEntry primaries_table[] = {
    { "BT.709",     &prim_bt709 },
    { "BT.2020",    &prim_bt2020 },
    { "SMPTE170M",  &prim_smpte170m },
    { "BT.470BG",   &prim_bt470bg },
    { "Display P3", &prim_smpte432 },
    { "BT.470M",    &prim_bt470m },
};
#define NUM_PRIMS (sizeof(primaries_table) / sizeof(primaries_table[0]))

static int test_conversion(const char *src_name, const ColorPrimDesc *src,
                            const char *dst_name, const ColorPrimDesc *dst)
{
    FPMat3 fp_mat;
    FloatMat3 ref_mat;
    int16_t fp_coeffs[3][3], ref_coeffs[3][3];
    int coeff_diff_max = 0;
    int pixel_diff_max = 0;
    int total_tested = 0;
    int total_diffs = 0;
    int worst_r = 0, worst_g = 0, worst_b = 0;
    int i, j, r, g, b;

    printf("Testing: %s -> %s\n", src_name, dst_name);

    /* Compute conversion matrices */
    fp_mat = fp_compute_rgb2rgb(src, dst);
    ref_mat = ref_compute_rgb2rgb(src, dst);

    /* Quantize to Q14 coefficients */
    fp_quantize_coeffs(&fp_mat, fp_coeffs);
    ref_quantize_coeffs(&ref_mat, ref_coeffs);

    /* Report matrix values */
    printf("  Fixed-point Q20 matrix:\n");
    for (i = 0; i < 3; i++) {
        printf("    [");
        for (j = 0; j < 3; j++)
            printf(" %10lld", (long long)fp_mat.m[i][j]);
        printf(" ]\n");
    }

    printf("  Float reference matrix:\n");
    for (i = 0; i < 3; i++) {
        printf("    [");
        for (j = 0; j < 3; j++)
            printf(" %10.6f", (float)ref_mat.m[i][j]);
        printf(" ]\n");
    }

    printf("  Q14 coefficients (fixed-point / reference / diff):\n");
    for (i = 0; i < 3; i++) {
        printf("    [");
        for (j = 0; j < 3; j++) {
            int diff = fp_coeffs[i][j] - ref_coeffs[i][j];
            printf(" %6d/%6d/%+d", fp_coeffs[i][j], ref_coeffs[i][j], diff);
            if (diff < 0) diff = -diff;
            if (diff > coeff_diff_max) coeff_diff_max = diff;
        }
        printf(" ]\n");
    }

    printf("  Max coefficient difference: %d\n", coeff_diff_max);

    /* Test pixel conversions for all 8-bit RGB values
     * Map 8-bit [0,255] to internal range [0,28672]:
     * pixel = value * 28672 / 255 ≈ value * 112.4 */
    for (r = 0; r < 256; r++) {
        for (g = 0; g < 256; g++) {
            for (b = 0; b < 256; b++) {
                int16_t ri = (int16_t)((int32_t)r * PIXEL_SCALE / 255);
                int16_t gi = (int16_t)((int32_t)g * PIXEL_SCALE / 255);
                int16_t bi = (int16_t)((int32_t)b * PIXEL_SCALE / 255);
                int16_t fp_ro, fp_go, fp_bo;
                int16_t ref_ro, ref_go, ref_bo;
                int dr, dg, db, dmax;

                fp_apply_matrix(fp_coeffs, ri, gi, bi, &fp_ro, &fp_go, &fp_bo);
                ref_apply_matrix(ref_coeffs, ri, gi, bi, &ref_ro, &ref_go, &ref_bo);

                dr = fp_ro - ref_ro; if (dr < 0) dr = -dr;
                dg = fp_go - ref_go; if (dg < 0) dg = -dg;
                db = fp_bo - ref_bo; if (db < 0) db = -db;

                dmax = dr;
                if (dg > dmax) dmax = dg;
                if (db > dmax) dmax = db;

                if (dmax > 0) total_diffs++;
                total_tested++;

                if (dmax > pixel_diff_max) {
                    pixel_diff_max = dmax;
                    worst_r = r;
                    worst_g = g;
                    worst_b = b;
                }
            }
        }
    }

    printf("  Pixel test: %d values tested, %d with differences\n",
           total_tested, total_diffs);
    printf("  Max pixel difference: %d", pixel_diff_max);
    if (pixel_diff_max > 0) {
        printf(" (at RGB=%d,%d,%d)", worst_r, worst_g, worst_b);
    }
    printf("\n");

    if (pixel_diff_max > 1) {
        printf("  *** FAIL: max pixel error %d exceeds tolerance of 1 ***\n",
               pixel_diff_max);
        return 1;
    }
    printf("  PASS\n\n");
    return 0;
}

int main(void)
{
    int failures = 0;
    int total = 0;
    size_t i, j;

    printf("=== CSC Fixed-Point Verification Test ===\n");
    printf("Fixed-point format: Q%d (%lld = 1.0)\n", FP_BITS, (long long)FP_ONE);
    printf("Coefficient format: Q%d (%d = 1.0)\n", COEFF_BITS, 1 << COEFF_BITS);
    printf("Pixel scale: %d\n", PIXEL_SCALE);
    printf("Tolerance: max pixel error <= 1\n\n");

    for (i = 0; i < NUM_PRIMS; i++) {
        for (j = 0; j < NUM_PRIMS; j++) {
            if (i == j) continue;
            total++;
            failures += test_conversion(
                primaries_table[i].name, primaries_table[i].desc,
                primaries_table[j].name, primaries_table[j].desc);
        }
    }

    printf("=== Summary: %d/%d tests passed ===\n",
           total - failures, total);

    return failures > 0 ? 1 : 0;
}
