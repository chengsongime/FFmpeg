/*
 * CSC (Color Space Conversion) Fixed-Point Verification Tool
 *
 * This tool reproduces the color gamut conversion pipeline from
 * libavfilter/vf_colorspace.c using fixed-point arithmetic (Q30 format)
 * instead of double precision floating-point.
 *
 * The pipeline under test:
 *   1. YUV -> RGB  (fixed-point 3x3 matrix, 14-bit coefficients)
 *   2. Linearize   (gamma removal via LUT, using float for pow only)
 *   3. RGB -> RGB   (gamut conversion, fixed-point 3x3 matrix)
 *   4. Delinearize (gamma application via LUT)
 *   5. RGB -> YUV  (fixed-point 3x3 matrix)
 *
 * All matrix coefficient computations use Q30 fixed-point with __int128
 * for intermediate products. Double is never used. Float is used only
 * for pow() in gamma LUT generation (unavoidable transcendental).
 *
 * Verification: maximum per-pixel error must not exceed ±1 compared
 * to the reference (double-based) implementation.
 *
 * Copyright (c) 2024 FFmpeg developers
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "libavutil/avutil.h"
#include "libavutil/common.h"
#include "libavutil/csp.h"
#include "libavutil/pixfmt.h"
#include "libavutil/rational.h"

/* ================================================================
 * Fixed-Point Arithmetic (Q30 format)
 *
 * We use Q30: 30 fractional bits stored in int64_t.
 * Range: roughly [-2^33, 2^33) with precision ~10^-9.
 * Intermediate products use __int128 to avoid overflow.
 * ================================================================ */

#define FP_BITS 30
#define FP_ONE  ((int64_t)1 << FP_BITS)
#define FP_HALF ((int64_t)1 << (FP_BITS - 1))

typedef int64_t fp_t;

static fp_t fp_from_rational(AVRational r)
{
    return ((int64_t)r.num << FP_BITS) / r.den;
}

static fp_t fp_mul(fp_t a, fp_t b)
{
    return (fp_t)(((__int128)a * b + FP_HALF) >> FP_BITS);
}

static fp_t fp_div(fp_t a, fp_t b)
{
    return (fp_t)(((__int128)a << FP_BITS) / b);
}

/* Convert Q30 to float (for display only) */
static float fp_to_float(fp_t v)
{
    return (float)v / (float)FP_ONE;
}

/* ================================================================
 * Fixed-Point 3x3 Matrix Operations
 * ================================================================ */

typedef struct {
    fp_t m[3][3];
} FPMatrix3x3;

static void fp_matrix_invert(const FPMatrix3x3 *in, FPMatrix3x3 *out)
{
    fp_t m00 = in->m[0][0], m01 = in->m[0][1], m02 = in->m[0][2];
    fp_t m10 = in->m[1][0], m11 = in->m[1][1], m12 = in->m[1][2];
    fp_t m20 = in->m[2][0], m21 = in->m[2][1], m22 = in->m[2][2];
    fp_t det;
    int i, j;

    /* Cofactor matrix (each element is Q30 after fp_mul) */
    out->m[0][0] =  (fp_mul(m11, m22) - fp_mul(m21, m12));
    out->m[0][1] = -(fp_mul(m01, m22) - fp_mul(m21, m02));
    out->m[0][2] =  (fp_mul(m01, m12) - fp_mul(m11, m02));
    out->m[1][0] = -(fp_mul(m10, m22) - fp_mul(m20, m12));
    out->m[1][1] =  (fp_mul(m00, m22) - fp_mul(m20, m02));
    out->m[1][2] = -(fp_mul(m00, m12) - fp_mul(m10, m02));
    out->m[2][0] =  (fp_mul(m10, m21) - fp_mul(m20, m11));
    out->m[2][1] = -(fp_mul(m00, m21) - fp_mul(m20, m01));
    out->m[2][2] =  (fp_mul(m00, m11) - fp_mul(m10, m01));

    /* Determinant (Q30) */
    det = fp_mul(m00, out->m[0][0]) +
          fp_mul(m10, out->m[0][1]) +
          fp_mul(m20, out->m[0][2]);

    /* Divide each cofactor by determinant */
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            out->m[i][j] = fp_div(out->m[i][j], det);
}

static void fp_matrix_mul(FPMatrix3x3 *dst,
                          const FPMatrix3x3 *src1,
                          const FPMatrix3x3 *src2)
{
    int m, n;
    FPMatrix3x3 tmp;

    for (m = 0; m < 3; m++)
        for (n = 0; n < 3; n++)
            tmp.m[m][n] = fp_mul(src2->m[m][0], src1->m[0][n]) +
                          fp_mul(src2->m[m][1], src1->m[1][n]) +
                          fp_mul(src2->m[m][2], src1->m[2][n]);

    *dst = tmp;
}

/* ================================================================
 * CSC Coefficient Computation (Fixed-Point, No Double)
 *
 * Reproduces ff_fill_rgb2xyz_table() and ff_fill_rgb2yuv_table()
 * using Q30 fixed-point arithmetic.
 * ================================================================ */

static void fp_fill_rgb2xyz(const AVPrimaryCoefficients *coeffs,
                            const AVWhitepointCoefficients *wp,
                            FPMatrix3x3 *rgb2xyz)
{
    FPMatrix3x3 inv;
    fp_t sr, sg, sb, zw;
    fp_t xr = fp_from_rational(coeffs->r.x), yr = fp_from_rational(coeffs->r.y);
    fp_t xg = fp_from_rational(coeffs->g.x), yg = fp_from_rational(coeffs->g.y);
    fp_t xb = fp_from_rational(coeffs->b.x), yb = fp_from_rational(coeffs->b.y);
    fp_t xw = fp_from_rational(wp->x),       yw = fp_from_rational(wp->y);

    /* Build chromaticity matrix */
    rgb2xyz->m[0][0] = fp_div(xr, yr);
    rgb2xyz->m[0][1] = fp_div(xg, yg);
    rgb2xyz->m[0][2] = fp_div(xb, yb);
    rgb2xyz->m[1][0] = rgb2xyz->m[1][1] = rgb2xyz->m[1][2] = FP_ONE;
    rgb2xyz->m[2][0] = fp_div(FP_ONE - xr - yr, yr);
    rgb2xyz->m[2][1] = fp_div(FP_ONE - xg - yg, yg);
    rgb2xyz->m[2][2] = fp_div(FP_ONE - xb - yb, yb);

    /* Invert to find scaling factors */
    fp_matrix_invert(rgb2xyz, &inv);
    zw = FP_ONE - xw - yw;
    sr = fp_mul(inv.m[0][0], xw) + fp_mul(inv.m[0][1], yw) + fp_mul(inv.m[0][2], zw);
    sg = fp_mul(inv.m[1][0], xw) + fp_mul(inv.m[1][1], yw) + fp_mul(inv.m[1][2], zw);
    sb = fp_mul(inv.m[2][0], xw) + fp_mul(inv.m[2][1], yw) + fp_mul(inv.m[2][2], zw);

    /* Apply scaling */
    rgb2xyz->m[0][0] = fp_mul(rgb2xyz->m[0][0], sr);
    rgb2xyz->m[0][1] = fp_mul(rgb2xyz->m[0][1], sg);
    rgb2xyz->m[0][2] = fp_mul(rgb2xyz->m[0][2], sb);
    rgb2xyz->m[1][0] = sr;
    rgb2xyz->m[1][1] = sg;
    rgb2xyz->m[1][2] = sb;
    rgb2xyz->m[2][0] = fp_mul(rgb2xyz->m[2][0], sr);
    rgb2xyz->m[2][1] = fp_mul(rgb2xyz->m[2][1], sg);
    rgb2xyz->m[2][2] = fp_mul(rgb2xyz->m[2][2], sb);
}

static void fp_fill_rgb2yuv(const AVLumaCoefficients *coeffs,
                            FPMatrix3x3 *rgb2yuv)
{
    fp_t cr = fp_from_rational(coeffs->cr);
    fp_t cg = fp_from_rational(coeffs->cg);
    fp_t cb = fp_from_rational(coeffs->cb);
    fp_t bscale, rscale;

    rgb2yuv->m[0][0] = cr;
    rgb2yuv->m[0][1] = cg;
    rgb2yuv->m[0][2] = cb;
    bscale = fp_div(FP_HALF, cb - FP_ONE);   /* 0.5 / (cb - 1.0) */
    rscale = fp_div(FP_HALF, cr - FP_ONE);   /* 0.5 / (cr - 1.0) */
    rgb2yuv->m[1][0] = fp_mul(bscale, cr);
    rgb2yuv->m[1][1] = fp_mul(bscale, cg);
    rgb2yuv->m[1][2] = FP_HALF;              /* 0.5 */
    rgb2yuv->m[2][0] = FP_HALF;
    rgb2yuv->m[2][1] = fp_mul(rscale, cg);
    rgb2yuv->m[2][2] = fp_mul(rscale, cb);
}

/* ================================================================
 * Whitepoint Adaptation (Bradford)
 *
 * Reproduces fill_whitepoint_conv_table() using fixed-point.
 * ================================================================ */

static void fp_fill_whitepoint_conv(FPMatrix3x3 *out,
                                    const AVWhitepointCoefficients *wp_src,
                                    const AVWhitepointCoefficients *wp_dst)
{
    /* Bradford chromatic adaptation matrix */
    static const float ma_f[3][3] = {
        {  0.8951f,  0.2664f, -0.1614f },
        { -0.7502f,  1.7135f,  0.0367f },
        {  0.0389f, -0.0685f,  1.0296f },
    };
    FPMatrix3x3 ma, mai, fac, tmp;
    fp_t xw_src, yw_src, zw_src, xw_dst, yw_dst, zw_dst;
    fp_t rs, gs, bs, rd, gd, bd;
    int i, j;

    /* Convert Bradford matrix from float to Q30 (no double used) */
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            ma.m[i][j] = (fp_t)(ma_f[i][j] * (float)FP_ONE + (ma_f[i][j] >= 0 ? 0.5f : -0.5f));

    xw_src = fp_from_rational(wp_src->x);
    yw_src = fp_from_rational(wp_src->y);
    zw_src = FP_ONE - xw_src - yw_src;
    xw_dst = fp_from_rational(wp_dst->x);
    yw_dst = fp_from_rational(wp_dst->y);
    zw_dst = FP_ONE - xw_dst - yw_dst;

    fp_matrix_invert(&ma, &mai);

    rs = fp_mul(ma.m[0][0], xw_src) + fp_mul(ma.m[0][1], yw_src) + fp_mul(ma.m[0][2], zw_src);
    gs = fp_mul(ma.m[1][0], xw_src) + fp_mul(ma.m[1][1], yw_src) + fp_mul(ma.m[1][2], zw_src);
    bs = fp_mul(ma.m[2][0], xw_src) + fp_mul(ma.m[2][1], yw_src) + fp_mul(ma.m[2][2], zw_src);
    rd = fp_mul(ma.m[0][0], xw_dst) + fp_mul(ma.m[0][1], yw_dst) + fp_mul(ma.m[0][2], zw_dst);
    gd = fp_mul(ma.m[1][0], xw_dst) + fp_mul(ma.m[1][1], yw_dst) + fp_mul(ma.m[1][2], zw_dst);
    bd = fp_mul(ma.m[2][0], xw_dst) + fp_mul(ma.m[2][1], yw_dst) + fp_mul(ma.m[2][2], zw_dst);

    memset(&fac, 0, sizeof(fac));
    fac.m[0][0] = fp_div(rd, rs);
    fac.m[1][1] = fp_div(gd, gs);
    fac.m[2][2] = fp_div(bd, bs);

    fp_matrix_mul(&tmp, &ma, &fac);
    fp_matrix_mul(out, &tmp, &mai);
}

/* ================================================================
 * Gamma LUT Generation
 *
 * Uses float for pow() only - this is the minimal floating-point
 * usage needed for transcendental functions. Everything else is
 * integer/fixed-point.
 * ================================================================ */

struct TransferChar {
    float alpha, beta, gamma, delta;
};

static const struct TransferChar trc_table[] = {
    [AVCOL_TRC_BT709]        = { 1.099f,  0.018f,  0.45f, 4.5f },
    [AVCOL_TRC_GAMMA22]      = { 1.0f,    0.0f,    1.0f / 2.2f, 0.0f },
    [AVCOL_TRC_GAMMA28]      = { 1.0f,    0.0f,    1.0f / 2.8f, 0.0f },
    [AVCOL_TRC_SMPTE170M]    = { 1.099f,  0.018f,  0.45f, 4.5f },
    [AVCOL_TRC_SMPTE240M]    = { 1.1115f, 0.0228f, 0.45f, 4.0f },
    [AVCOL_TRC_LINEAR]       = { 1.0f,    0.0f,    1.0f,  0.0f },
    [AVCOL_TRC_IEC61966_2_1] = { 1.055f,  0.0031308f, 1.0f / 2.4f, 12.92f },
    [AVCOL_TRC_IEC61966_2_4] = { 1.099f,  0.018f,  0.45f, 4.5f },
    [AVCOL_TRC_BT2020_10]    = { 1.099f,  0.018f,  0.45f, 4.5f },
    [AVCOL_TRC_BT2020_12]    = { 1.0993f, 0.0181f, 0.45f, 4.5f },
};

static int fill_gamma_lut(int16_t **lin_lut_out, int16_t **delin_lut_out,
                          const struct TransferChar *in_trc,
                          const struct TransferChar *out_trc)
{
    int16_t *lin_lut, *delin_lut;
    int n;

    lin_lut = malloc(sizeof(*lin_lut) * 32768 * 2);
    if (!lin_lut)
        return -1;
    delin_lut = lin_lut + 32768;

    for (n = 0; n < 32768; n++) {
        float v = (n - 2048.0f) / 28672.0f;
        float d, l;
        int d_rounded, l_rounded;

        /* Delinearize (using output TRC) */
        if (v <= -out_trc->beta)
            d = -out_trc->alpha * powf(-v, out_trc->gamma) + (out_trc->alpha - 1.0f);
        else if (v < out_trc->beta)
            d = out_trc->delta * v;
        else
            d = out_trc->alpha * powf(v, out_trc->gamma) - (out_trc->alpha - 1.0f);
        d_rounded = lrintf(d * 28672.0f);
        delin_lut[n] = av_clip_int16(d_rounded);

        /* Linearize (using input TRC) */
        if (in_trc->delta != 0.0f) {
            float ialpha = 1.0f / in_trc->alpha;
            float igamma = 1.0f / in_trc->gamma;
            float idelta = 1.0f / in_trc->delta;

            if (v <= -in_trc->beta * in_trc->delta)
                l = -powf((1.0f - in_trc->alpha - v) * ialpha, igamma);
            else if (v < in_trc->beta * in_trc->delta)
                l = v * idelta;
            else
                l = powf((v + in_trc->alpha - 1.0f) * ialpha, igamma);
        } else {
            /* gamma-only (no linear segment) */
            float igamma = 1.0f / in_trc->gamma;
            if (v < 0)
                l = -powf(-v, igamma);
            else
                l = powf(v, igamma);
        }
        l_rounded = lrintf(l * 28672.0f);
        lin_lut[n] = av_clip_int16(l_rounded);
    }

    *lin_lut_out = lin_lut;
    *delin_lut_out = delin_lut;
    return 0;
}

/* ================================================================
 * Per-Pixel Processing (Integer Only)
 *
 * Reproduces the colorspacedsp_template.c pipeline.
 * All operations are integer arithmetic.
 * ================================================================ */

/* YUV -> internal RGB (15-bit signed representation)
 * Coefficients are 14-bit fixed-point. */
static void yuv2rgb_pixel(int y_val, int u_val, int v_val,
                          int *r, int *g, int *b,
                          const int16_t yuv2rgb_coeffs[3][3],
                          int yuv_offset, int depth)
{
    int y0 = y_val - yuv_offset;
    int u0 = u_val - (128 << (depth - 8));
    int v0 = v_val - (128 << (depth - 8));
    int sh = depth - 1;
    int rnd = 1 << (sh - 1);

    int cy  = yuv2rgb_coeffs[0][0];
    int crv = yuv2rgb_coeffs[0][2];
    int cgu = yuv2rgb_coeffs[1][1];
    int cgv = yuv2rgb_coeffs[1][2];
    int cbu = yuv2rgb_coeffs[2][1];

    *r = av_clip_int16((y0 * cy + crv * v0 + rnd) >> sh);
    *g = av_clip_int16((y0 * cy + cgu * u0 + cgv * v0 + rnd) >> sh);
    *b = av_clip_int16((y0 * cy + cbu * u0 + rnd) >> sh);
}

/* Apply LUT to linearize/delinearize */
static int16_t apply_lut_val(int16_t val, const int16_t *lut)
{
    return lut[av_clip_uintp2(2048 + val, 15)];
}

/* 3x3 matrix multiply on internal RGB (14-bit coefficients) */
static void multiply3x3_pixel(int *r, int *g, int *b,
                               const int16_t m[3][3])
{
    int v0 = *r, v1 = *g, v2 = *b;
    *r = av_clip_int16((m[0][0] * v0 + m[0][1] * v1 + m[0][2] * v2 + 8192) >> 14);
    *g = av_clip_int16((m[1][0] * v0 + m[1][1] * v1 + m[1][2] * v2 + 8192) >> 14);
    *b = av_clip_int16((m[2][0] * v0 + m[2][1] * v1 + m[2][2] * v2 + 8192) >> 14);
}

/* Internal RGB -> YUV (output pixel) */
static void rgb2yuv_pixel(int r, int g, int b,
                          int *y_out, int *u_out, int *v_out,
                          const int16_t rgb2yuv_coeffs[3][3],
                          int yuv_offset, int depth)
{
    int sh = 29 - depth;
    int rnd = 1 << (sh - 1);
    int uv_offset = 128 << (depth - 8);

    *y_out = av_clip_uint8(yuv_offset +
                           ((r * rgb2yuv_coeffs[0][0] + g * rgb2yuv_coeffs[0][1] +
                             b * rgb2yuv_coeffs[0][2] + rnd) >> sh));
    *u_out = av_clip_uint8(uv_offset +
                           ((r * rgb2yuv_coeffs[1][0] + g * rgb2yuv_coeffs[1][1] +
                             b * rgb2yuv_coeffs[1][2] + rnd) >> sh));
    *v_out = av_clip_uint8(uv_offset +
                           ((r * rgb2yuv_coeffs[2][0] + g * rgb2yuv_coeffs[2][1] +
                             b * rgb2yuv_coeffs[2][2] + rnd) >> sh));
}

/* ================================================================
 * Reference Implementation (using double for comparison ONLY)
 *
 * This section uses double to compute the "ground truth" coefficients.
 * It follows the exact same algorithm as vf_colorspace.c.
 * ================================================================ */

static void ref_fill_rgb2xyz(const AVPrimaryCoefficients *coeffs,
                             const AVWhitepointCoefficients *wp,
                             double rgb2xyz[3][3])
{
    double inv[3][3], sr, sg, sb, zw;
    double xr = av_q2d(coeffs->r.x), yr = av_q2d(coeffs->r.y);
    double xg = av_q2d(coeffs->g.x), yg = av_q2d(coeffs->g.y);
    double xb = av_q2d(coeffs->b.x), yb = av_q2d(coeffs->b.y);
    double xw = av_q2d(wp->x),       yw = av_q2d(wp->y);
    double m00, m01, m02, m10, m11, m12, m20, m21, m22;
    double det;
    int i, j;

    rgb2xyz[0][0] = xr / yr;
    rgb2xyz[0][1] = xg / yg;
    rgb2xyz[0][2] = xb / yb;
    rgb2xyz[1][0] = rgb2xyz[1][1] = rgb2xyz[1][2] = 1.0;
    rgb2xyz[2][0] = (1.0 - xr - yr) / yr;
    rgb2xyz[2][1] = (1.0 - xg - yg) / yg;
    rgb2xyz[2][2] = (1.0 - xb - yb) / yb;

    /* Invert */
    m00 = rgb2xyz[0][0]; m01 = rgb2xyz[0][1]; m02 = rgb2xyz[0][2];
    m10 = rgb2xyz[1][0]; m11 = rgb2xyz[1][1]; m12 = rgb2xyz[1][2];
    m20 = rgb2xyz[2][0]; m21 = rgb2xyz[2][1]; m22 = rgb2xyz[2][2];

    inv[0][0] =  (m11 * m22 - m21 * m12);
    inv[0][1] = -(m01 * m22 - m21 * m02);
    inv[0][2] =  (m01 * m12 - m11 * m02);
    inv[1][0] = -(m10 * m22 - m20 * m12);
    inv[1][1] =  (m00 * m22 - m20 * m02);
    inv[1][2] = -(m00 * m12 - m10 * m02);
    inv[2][0] =  (m10 * m21 - m20 * m11);
    inv[2][1] = -(m00 * m21 - m20 * m01);
    inv[2][2] =  (m00 * m11 - m10 * m01);

    det = m00 * inv[0][0] + m10 * inv[0][1] + m20 * inv[0][2];
    det = 1.0 / det;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            inv[i][j] *= det;

    zw = 1.0 - xw - yw;
    sr = inv[0][0] * xw + inv[0][1] * yw + inv[0][2] * zw;
    sg = inv[1][0] * xw + inv[1][1] * yw + inv[1][2] * zw;
    sb = inv[2][0] * xw + inv[2][1] * yw + inv[2][2] * zw;

    rgb2xyz[0][0] *= sr; rgb2xyz[0][1] *= sg; rgb2xyz[0][2] *= sb;
    rgb2xyz[1][0] *= sr; rgb2xyz[1][1] *= sg; rgb2xyz[1][2] *= sb;
    rgb2xyz[2][0] *= sr; rgb2xyz[2][1] *= sg; rgb2xyz[2][2] *= sb;
}

static void ref_matrix_invert(const double in[3][3], double out[3][3])
{
    double m00 = in[0][0], m01 = in[0][1], m02 = in[0][2],
           m10 = in[1][0], m11 = in[1][1], m12 = in[1][2],
           m20 = in[2][0], m21 = in[2][1], m22 = in[2][2];
    double det;
    int i, j;

    out[0][0] =  (m11 * m22 - m21 * m12);
    out[0][1] = -(m01 * m22 - m21 * m02);
    out[0][2] =  (m01 * m12 - m11 * m02);
    out[1][0] = -(m10 * m22 - m20 * m12);
    out[1][1] =  (m00 * m22 - m20 * m02);
    out[1][2] = -(m00 * m12 - m10 * m02);
    out[2][0] =  (m10 * m21 - m20 * m11);
    out[2][1] = -(m00 * m21 - m20 * m01);
    out[2][2] =  (m00 * m11 - m10 * m01);

    det = m00 * out[0][0] + m10 * out[0][1] + m20 * out[0][2];
    det = 1.0 / det;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            out[i][j] *= det;
}

static void ref_matrix_mul(double dst[3][3],
                           const double src1[3][3],
                           const double src2[3][3])
{
    int m, n;
    for (m = 0; m < 3; m++)
        for (n = 0; n < 3; n++)
            dst[m][n] = src2[m][0] * src1[0][n] +
                        src2[m][1] * src1[1][n] +
                        src2[m][2] * src1[2][n];
}

static void ref_fill_rgb2yuv(const AVLumaCoefficients *coeffs,
                             double rgb2yuv[3][3])
{
    double cr = av_q2d(coeffs->cr), cg = av_q2d(coeffs->cg), cb = av_q2d(coeffs->cb);
    double bscale, rscale;

    rgb2yuv[0][0] = cr;
    rgb2yuv[0][1] = cg;
    rgb2yuv[0][2] = cb;
    bscale = 0.5 / (cb - 1.0);
    rscale = 0.5 / (cr - 1.0);
    rgb2yuv[1][0] = bscale * cr;
    rgb2yuv[1][1] = bscale * cg;
    rgb2yuv[1][2] = 0.5;
    rgb2yuv[2][0] = 0.5;
    rgb2yuv[2][1] = rscale * cg;
    rgb2yuv[2][2] = rscale * cb;
}

static void ref_fill_whitepoint_conv(double out[3][3],
                                     const AVWhitepointCoefficients *wp_src,
                                     const AVWhitepointCoefficients *wp_dst)
{
    static const double ma[3][3] = {
        {  0.8951,  0.2664, -0.1614 },
        { -0.7502,  1.7135,  0.0367 },
        {  0.0389, -0.0685,  1.0296 },
    };
    double mai[3][3], fac[3][3], tmp[3][3];
    double xw_src = av_q2d(wp_src->x), yw_src = av_q2d(wp_src->y);
    double xw_dst = av_q2d(wp_dst->x), yw_dst = av_q2d(wp_dst->y);
    double zw_src = 1.0 - xw_src - yw_src;
    double zw_dst = 1.0 - xw_dst - yw_dst;
    double rs, gs, bs, rd, gd, bd;

    ref_matrix_invert(ma, mai);

    rs = ma[0][0] * xw_src + ma[0][1] * yw_src + ma[0][2] * zw_src;
    gs = ma[1][0] * xw_src + ma[1][1] * yw_src + ma[1][2] * zw_src;
    bs = ma[2][0] * xw_src + ma[2][1] * yw_src + ma[2][2] * zw_src;
    rd = ma[0][0] * xw_dst + ma[0][1] * yw_dst + ma[0][2] * zw_dst;
    gd = ma[1][0] * xw_dst + ma[1][1] * yw_dst + ma[1][2] * zw_dst;
    bd = ma[2][0] * xw_dst + ma[2][1] * yw_dst + ma[2][2] * zw_dst;

    memset(fac, 0, sizeof(fac));
    fac[0][0] = rd / rs;
    fac[1][1] = gd / gs;
    fac[2][2] = bd / bs;

    ref_matrix_mul(tmp, (const double (*)[3])ma, fac);
    ref_matrix_mul(out, tmp, mai);
}

/* ================================================================
 * Quantize Coefficients (Fixed-Point, No Double)
 *
 * Convert Q30 matrix to int16_t coefficients matching the format
 * used by the existing per-pixel DSP functions.
 * ================================================================ */

static void fp_quantize_yuv2rgb(const FPMatrix3x3 *yuv2rgb_fp,
                                int16_t out[3][3],
                                int depth, int y_rng, int uv_rng)
{
    int n, m;
    int bits = 1 << (depth - 1);

    for (n = 0; n < 3; n++) {
        int in_rng = y_rng;
        for (m = 0; m < 3; m++, in_rng = uv_rng) {
            /* coeff = round(28672 * bits * yuv2rgb[n][m] / in_rng)
             *       = round(yuv2rgb_fp * 28672 * bits / (in_rng * FP_ONE))
             *       = (yuv2rgb_fp * 28672 * bits / in_rng + FP_HALF) >> FP_BITS */
            int64_t val = yuv2rgb_fp->m[n][m] * (int64_t)28672 * bits / in_rng;
            out[n][m] = (int16_t)((val + FP_HALF) >> FP_BITS);
        }
    }
}

static void fp_quantize_rgb2yuv(const FPMatrix3x3 *rgb2yuv_fp,
                                int16_t out[3][3],
                                int depth, int y_rng, int uv_rng)
{
    int n, m;
    int bits = 1 << (29 - depth);

    int out_rng = y_rng;
    for (n = 0; n < 3; n++, out_rng = uv_rng) {
        for (m = 0; m < 3; m++) {
            /* coeff = round(bits * out_rng * rgb2yuv[n][m] / 28672) */
            int64_t val = rgb2yuv_fp->m[n][m] * (int64_t)bits * out_rng / 28672;
            out[n][m] = (int16_t)((val + FP_HALF) >> FP_BITS);
        }
    }
}

static void fp_quantize_lrgb2lrgb(const FPMatrix3x3 *rgb2rgb_fp,
                                  int16_t out[3][3])
{
    int m, n;
    for (m = 0; m < 3; m++)
        for (n = 0; n < 3; n++)
            /* coeff = round(16384 * rgb2rgb[m][n])
             *       = (rgb2rgb_fp * 16384 + FP_HALF) >> FP_BITS
             *       = (rgb2rgb_fp + (1 << 15)) >> 16 */
            out[m][n] = (int16_t)((rgb2rgb_fp->m[m][n] + (1LL << 15)) >> 16);
}

/* ================================================================
 * Reference Quantization (using double for comparison)
 * ================================================================ */

static void ref_quantize_yuv2rgb(const double yuv2rgb[3][3],
                                 int16_t out[3][3],
                                 int depth, int y_rng, int uv_rng)
{
    int n, m;
    int bits = 1 << (depth - 1);
    for (n = 0; n < 3; n++) {
        int in_rng = y_rng;
        for (m = 0; m < 3; m++, in_rng = uv_rng)
            out[n][m] = (int16_t)lrint(28672.0 * bits * yuv2rgb[n][m] / in_rng);
    }
}

static void ref_quantize_rgb2yuv(const double rgb2yuv[3][3],
                                 int16_t out[3][3],
                                 int depth, int y_rng, int uv_rng)
{
    int n, m;
    int bits = 1 << (29 - depth);
    int out_rng = y_rng;
    for (n = 0; n < 3; n++, out_rng = uv_rng)
        for (m = 0; m < 3; m++)
            out[n][m] = (int16_t)lrint((double)bits * out_rng * rgb2yuv[n][m] / 28672.0);
}

static void ref_quantize_lrgb2lrgb(const double rgb2rgb[3][3],
                                   int16_t out[3][3])
{
    int m, n;
    for (m = 0; m < 3; m++)
        for (n = 0; n < 3; n++)
            out[m][n] = (int16_t)lrint(16384.0 * rgb2rgb[m][n]);
}

/* ================================================================
 * Test Framework
 * ================================================================ */

struct CSCTestCase {
    const char *name;
    enum AVColorPrimaries in_prm, out_prm;
    enum AVColorSpace in_csp, out_csp;
    enum AVColorTransferCharacteristic in_trc, out_trc;
};

static const struct CSCTestCase test_cases[] = {
    { "BT.709 -> BT.2020",
      AVCOL_PRI_BT709,    AVCOL_PRI_BT2020,
      AVCOL_SPC_BT709,    AVCOL_SPC_BT2020_NCL,
      AVCOL_TRC_BT709,    AVCOL_TRC_BT2020_10 },
    { "BT.2020 -> BT.709",
      AVCOL_PRI_BT2020,   AVCOL_PRI_BT709,
      AVCOL_SPC_BT2020_NCL, AVCOL_SPC_BT709,
      AVCOL_TRC_BT2020_10, AVCOL_TRC_BT709 },
    { "BT.709 -> SMPTE170M",
      AVCOL_PRI_BT709,    AVCOL_PRI_SMPTE170M,
      AVCOL_SPC_BT709,    AVCOL_SPC_SMPTE170M,
      AVCOL_TRC_BT709,    AVCOL_TRC_SMPTE170M },
    { "SMPTE170M -> BT.709",
      AVCOL_PRI_SMPTE170M, AVCOL_PRI_BT709,
      AVCOL_SPC_SMPTE170M, AVCOL_SPC_BT709,
      AVCOL_TRC_SMPTE170M, AVCOL_TRC_BT709 },
    { "BT.470BG -> BT.2020",
      AVCOL_PRI_BT470BG,  AVCOL_PRI_BT2020,
      AVCOL_SPC_BT470BG,  AVCOL_SPC_BT2020_NCL,
      AVCOL_TRC_GAMMA28,  AVCOL_TRC_BT2020_10 },
    { "BT.709 -> BT.470M",
      AVCOL_PRI_BT709,    AVCOL_PRI_BT470M,
      AVCOL_SPC_BT709,    AVCOL_SPC_SMPTE170M,
      AVCOL_TRC_BT709,    AVCOL_TRC_GAMMA22 },
};

static int run_test(const struct CSCTestCase *tc)
{
    const AVColorPrimariesDesc *in_primaries, *out_primaries;
    const AVLumaCoefficients *in_lumacoef, *out_lumacoef;
    const struct TransferChar *in_trc_data, *out_trc_data;
    int depth = 8;
    int y_rng = 219, uv_rng = 224;
    int yuv_offset = 16;
    int pass = 1;
    int max_err_y = 0, max_err_u = 0, max_err_v = 0;
    int max_coeff_err = 0;

    /* Fixed-point matrices */
    FPMatrix3x3 fp_rgb2xyz_in, fp_xyz2rgb_out, fp_rgb2rgb;
    FPMatrix3x3 fp_rgb2yuv_in, fp_yuv2rgb_in;
    FPMatrix3x3 fp_rgb2yuv_out;
    int16_t fp_yuv2rgb_q[3][3], fp_rgb2yuv_q[3][3], fp_lrgb2lrgb_q[3][3];

    /* Reference (double) matrices */
    double ref_rgb2xyz_in[3][3], ref_xyz2rgb_out[3][3], ref_rgb2rgb[3][3];
    double ref_rgb2yuv_in[3][3], ref_yuv2rgb_in[3][3];
    double ref_rgb2yuv_out[3][3];
    int16_t ref_yuv2rgb_q[3][3], ref_rgb2yuv_q[3][3], ref_lrgb2lrgb_q[3][3];

    /* Gamma LUTs */
    int16_t *lin_lut = NULL, *delin_lut = NULL;
    int need_gamma;

    printf("\n=== Test: %s ===\n", tc->name);

    /* Get color space data */
    in_primaries  = av_csp_primaries_desc_from_id(tc->in_prm);
    out_primaries = av_csp_primaries_desc_from_id(tc->out_prm);
    in_lumacoef   = av_csp_luma_coeffs_from_avcsp(tc->in_csp);
    out_lumacoef  = av_csp_luma_coeffs_from_avcsp(tc->out_csp);

    if (!in_primaries || !out_primaries || !in_lumacoef || !out_lumacoef) {
        printf("SKIP: unsupported color space parameters\n");
        return 0;
    }

    /* --- Step 1: Compute YUV<->RGB matrices --- */

    /* Fixed-point */
    fp_fill_rgb2yuv(in_lumacoef, &fp_rgb2yuv_in);
    fp_matrix_invert(&fp_rgb2yuv_in, &fp_yuv2rgb_in);
    fp_fill_rgb2yuv(out_lumacoef, &fp_rgb2yuv_out);

    /* Reference */
    ref_fill_rgb2yuv(in_lumacoef, ref_rgb2yuv_in);
    ref_matrix_invert(ref_rgb2yuv_in, ref_yuv2rgb_in);
    ref_fill_rgb2yuv(out_lumacoef, ref_rgb2yuv_out);

    /* --- Step 2: Compute RGB<->RGB gamut matrix --- */
    {
        FPMatrix3x3 fp_rgb2xyz_out;

        fp_fill_rgb2xyz(&in_primaries->prim,  &in_primaries->wp,  &fp_rgb2xyz_in);
        fp_fill_rgb2xyz(&out_primaries->prim, &out_primaries->wp, &fp_rgb2xyz_out);
        fp_matrix_invert(&fp_rgb2xyz_out, &fp_xyz2rgb_out);

        /* Handle whitepoint adaptation if needed */
        if (memcmp(&in_primaries->wp, &out_primaries->wp,
                   sizeof(in_primaries->wp)) != 0) {
            FPMatrix3x3 wpconv, tmp;
            fp_fill_whitepoint_conv(&wpconv, &in_primaries->wp, &out_primaries->wp);
            fp_matrix_mul(&tmp, &fp_rgb2xyz_in, &wpconv);
            fp_matrix_mul(&fp_rgb2rgb, &tmp, &fp_xyz2rgb_out);
        } else {
            fp_matrix_mul(&fp_rgb2rgb, &fp_rgb2xyz_in, &fp_xyz2rgb_out);
        }
    }

    /* Reference */
    {
        double ref_rgb2xyz_out[3][3];
        ref_fill_rgb2xyz(&in_primaries->prim,  &in_primaries->wp,  ref_rgb2xyz_in);
        ref_fill_rgb2xyz(&out_primaries->prim, &out_primaries->wp, ref_rgb2xyz_out);
        ref_matrix_invert(ref_rgb2xyz_out, ref_xyz2rgb_out);

        if (memcmp(&in_primaries->wp, &out_primaries->wp,
                   sizeof(in_primaries->wp)) != 0) {
            double wpconv[3][3], tmp[3][3];
            ref_fill_whitepoint_conv(wpconv, &in_primaries->wp, &out_primaries->wp);
            ref_matrix_mul(tmp, ref_rgb2xyz_in, wpconv);
            ref_matrix_mul(ref_rgb2rgb, tmp, ref_xyz2rgb_out);
        } else {
            ref_matrix_mul(ref_rgb2rgb, ref_rgb2xyz_in, ref_xyz2rgb_out);
        }
    }

    /* --- Step 3: Quantize to int16 coefficients --- */
    fp_quantize_yuv2rgb(&fp_yuv2rgb_in, fp_yuv2rgb_q, depth, y_rng, uv_rng);
    fp_quantize_rgb2yuv(&fp_rgb2yuv_out, fp_rgb2yuv_q, depth, y_rng, uv_rng);
    fp_quantize_lrgb2lrgb(&fp_rgb2rgb, fp_lrgb2lrgb_q);

    ref_quantize_yuv2rgb(ref_yuv2rgb_in, ref_yuv2rgb_q, depth, y_rng, uv_rng);
    ref_quantize_rgb2yuv(ref_rgb2yuv_out, ref_rgb2yuv_q, depth, y_rng, uv_rng);
    ref_quantize_lrgb2lrgb(ref_rgb2rgb, ref_lrgb2lrgb_q);

    /* --- Step 4: Compare quantized coefficients --- */
    printf("  Coefficient comparison (fixed-point vs double):\n");
    {
        const char *names[] = {"yuv2rgb", "rgb2yuv", "lrgb2lrgb"};
        const int16_t (*fp_q[3])[3]  = { fp_yuv2rgb_q, fp_rgb2yuv_q, fp_lrgb2lrgb_q };
        const int16_t (*ref_q[3])[3] = { ref_yuv2rgb_q, ref_rgb2yuv_q, ref_lrgb2lrgb_q };
        int k;

        for (k = 0; k < 3; k++) {
            int max_e = 0;
            int m, n;
            for (m = 0; m < 3; m++)
                for (n = 0; n < 3; n++) {
                    int e = abs(fp_q[k][m][n] - ref_q[k][m][n]);
                    if (e > max_e) max_e = e;
                    if (e > max_coeff_err) max_coeff_err = e;
                }
            printf("    %-10s: max coeff error = %d %s\n",
                   names[k], max_e, max_e <= 1 ? "OK" : "FAIL");
            if (max_e > 1) pass = 0;
        }
    }

    /* --- Step 5: Determine if gamma correction is needed --- */
    in_trc_data  = (tc->in_trc < FF_ARRAY_ELEMS(trc_table) && trc_table[tc->in_trc].alpha != 0)
                   ? &trc_table[tc->in_trc] : NULL;
    out_trc_data = (tc->out_trc < FF_ARRAY_ELEMS(trc_table) && trc_table[tc->out_trc].alpha != 0)
                   ? &trc_table[tc->out_trc] : NULL;
    need_gamma = in_trc_data && out_trc_data;

    if (need_gamma) {
        if (fill_gamma_lut(&lin_lut, &delin_lut, in_trc_data, out_trc_data) < 0) {
            printf("  ERROR: failed to allocate gamma LUT\n");
            return -1;
        }
    }

    /* --- Step 6: Per-pixel verification --- */
    printf("  Per-pixel verification (8-bit, MPEG range):\n");
    {
        int y_val, u_val, v_val;
        int test_count = 0;

        /* Test all Y values with fixed U,V;
         * and grid of U,V values with fixed Y */
        for (y_val = 16; y_val <= 235; y_val++) {
            for (u_val = 16; u_val <= 240; u_val += 8) {
                for (v_val = 16; v_val <= 240; v_val += 8) {
                    int fp_r, fp_g, fp_b;
                    int ref_r, ref_g, ref_b;
                    int fp_yo, fp_uo, fp_vo;
                    int ref_yo, ref_uo, ref_vo;

                    /* Fixed-point path */
                    yuv2rgb_pixel(y_val, u_val, v_val,
                                 &fp_r, &fp_g, &fp_b,
                                 fp_yuv2rgb_q, yuv_offset, depth);

                    if (need_gamma) {
                        fp_r = apply_lut_val(fp_r, lin_lut);
                        fp_g = apply_lut_val(fp_g, lin_lut);
                        fp_b = apply_lut_val(fp_b, lin_lut);
                    }

                    multiply3x3_pixel(&fp_r, &fp_g, &fp_b, fp_lrgb2lrgb_q);

                    if (need_gamma) {
                        fp_r = apply_lut_val(fp_r, delin_lut);
                        fp_g = apply_lut_val(fp_g, delin_lut);
                        fp_b = apply_lut_val(fp_b, delin_lut);
                    }

                    rgb2yuv_pixel(fp_r, fp_g, fp_b,
                                 &fp_yo, &fp_uo, &fp_vo,
                                 fp_rgb2yuv_q, yuv_offset, depth);

                    /* Reference path (identical pipeline, different coefficients) */
                    yuv2rgb_pixel(y_val, u_val, v_val,
                                 &ref_r, &ref_g, &ref_b,
                                 ref_yuv2rgb_q, yuv_offset, depth);

                    if (need_gamma) {
                        ref_r = apply_lut_val(ref_r, lin_lut);
                        ref_g = apply_lut_val(ref_g, lin_lut);
                        ref_b = apply_lut_val(ref_b, lin_lut);
                    }

                    multiply3x3_pixel(&ref_r, &ref_g, &ref_b, ref_lrgb2lrgb_q);

                    if (need_gamma) {
                        ref_r = apply_lut_val(ref_r, delin_lut);
                        ref_g = apply_lut_val(ref_g, delin_lut);
                        ref_b = apply_lut_val(ref_b, delin_lut);
                    }

                    rgb2yuv_pixel(ref_r, ref_g, ref_b,
                                 &ref_yo, &ref_uo, &ref_vo,
                                 ref_rgb2yuv_q, yuv_offset, depth);

                    /* Check error */
                    {
                        int ey = abs(fp_yo - ref_yo);
                        int eu = abs(fp_uo - ref_uo);
                        int ev = abs(fp_vo - ref_vo);
                        if (ey > max_err_y) max_err_y = ey;
                        if (eu > max_err_u) max_err_u = eu;
                        if (ev > max_err_v) max_err_v = ev;
                    }
                    test_count++;
                }
            }
        }

        printf("    Tested %d pixel combinations\n", test_count);
        printf("    Max pixel error: Y=%d U=%d V=%d\n",
               max_err_y, max_err_u, max_err_v);

        if (max_err_y > 1 || max_err_u > 1 || max_err_v > 1) {
            printf("    FAIL: pixel error exceeds ±1\n");
            pass = 0;
        } else {
            printf("    PASS: all pixel errors within ±1\n");
        }
    }

    free(lin_lut);

    printf("  Overall: %s (max coeff err=%d, max pixel err Y=%d U=%d V=%d)\n",
           pass ? "PASS" : "FAIL", max_coeff_err,
           max_err_y, max_err_u, max_err_v);

    return pass ? 0 : 1;
}

/* ================================================================
 * Print detailed matrix analysis
 * ================================================================ */

static void print_matrix_comparison(const char *name,
                                    const FPMatrix3x3 *fp,
                                    const double ref[3][3])
{
    int i, j;
    printf("  %s matrix (fixed-point vs double):\n", name);
    for (i = 0; i < 3; i++) {
        printf("    [");
        for (j = 0; j < 3; j++) {
            float fp_val  = fp_to_float(fp->m[i][j]);
            float ref_val = (float)ref[i][j];
            float err = fp_val - ref_val;
            printf(" %10.6f(%+.1e)", fp_val, err);
        }
        printf(" ]\n");
    }
}

int main(int argc, char *argv[])
{
    int ret = 0;
    int i;

    printf("CSC Fixed-Point Verification Tool\n");
    printf("==================================\n");
    printf("Fixed-point format: Q%d (%d fractional bits)\n", FP_BITS, FP_BITS);
    printf("Precision: ~%.1e\n", 1.0 / (double)FP_ONE);
    printf("Depth: 8-bit, Range: MPEG (16-235/240)\n");
    printf("Number of test cases: %d\n",
           (int)(sizeof(test_cases) / sizeof(test_cases[0])));

    /* Print detailed matrix analysis for the first test case */
    {
        const struct CSCTestCase *tc = &test_cases[0];
        const AVColorPrimariesDesc *in_prm  = av_csp_primaries_desc_from_id(tc->in_prm);
        const AVLumaCoefficients *in_luma = av_csp_luma_coeffs_from_avcsp(tc->in_csp);

        if (in_prm && in_luma) {
            FPMatrix3x3 fp_m;
            double ref_m[3][3];

            printf("\n--- Detailed Analysis: %s ---\n", tc->name);

            fp_fill_rgb2xyz(&in_prm->prim, &in_prm->wp, &fp_m);
            ref_fill_rgb2xyz(&in_prm->prim, &in_prm->wp, ref_m);
            print_matrix_comparison("RGB->XYZ (input)", &fp_m, ref_m);

            fp_fill_rgb2yuv(in_luma, &fp_m);
            ref_fill_rgb2yuv(in_luma, ref_m);
            print_matrix_comparison("RGB->YUV (input)", &fp_m, ref_m);
        }
    }

    /* Run all test cases */
    for (i = 0; i < (int)(sizeof(test_cases) / sizeof(test_cases[0])); i++) {
        int err = run_test(&test_cases[i]);
        if (err != 0) ret = 1;
    }

    printf("\n==================================\n");
    printf("Final result: %s\n", ret == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return ret;
}
