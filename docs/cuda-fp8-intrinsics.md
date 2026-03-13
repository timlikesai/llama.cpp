# CUDA FP8 Conversion and Data Movement API
Source: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html

## Type Definitions
```cpp
typedef unsigned char      __nv_fp8_storage_t;    // 1 FP8 value
typedef unsigned short int __nv_fp8x2_storage_t;  // 2 FP8 values
typedef unsigned int       __nv_fp8x4_storage_t;  // 4 FP8 values
```

## Enumerations
```cpp
enum __nv_fp8_interpretation_t { __NV_E4M3, __NV_E5M2 };
enum __nv_saturation_t { __NV_NOSAT, __NV_SATFINITE };
```

## Dequant Functions (FP8 → half)
```cpp
__half_raw  __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t x, __nv_fp8_interpretation_t);
__half2_raw __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t x, __nv_fp8_interpretation_t);
// NOTE: No fp8x4 → float4 __nv_cvt function. Use class operator instead (see below).
```

## E8M0 Functions (Scaling Factor Conversions)

E8M0 is an 8-bit pure exponent format (0 mantissa bits) with bias 127.
Encodes values 2^(-127) through 2^(127) (stored as 0-254), with 255 = NaN.
Used as shared scale factor per block of 32 elements in MX (Microscaling) format.

### Standalone Conversion Functions
```cpp
// Encode: float/double/bf16 → E8M0
__nv_fp8_storage_t   __nv_cvt_float_to_e8m0(float x, __nv_saturation_t, cudaRoundMode);
__nv_fp8_storage_t   __nv_cvt_double_to_e8m0(double x, __nv_saturation_t, cudaRoundMode);
__nv_fp8_storage_t   __nv_cvt_bfloat16raw_to_e8m0(__nv_bfloat16_raw x, __nv_saturation_t, cudaRoundMode);

// Encode x2: pair of values → E8M0 pair
__nv_fp8x2_storage_t __nv_cvt_float2_to_e8m0x2(float2 x, __nv_saturation_t, cudaRoundMode);
__nv_fp8x2_storage_t __nv_cvt_double2_to_e8m0x2(double2 x, __nv_saturation_t, cudaRoundMode);
__nv_fp8x2_storage_t __nv_cvt_bfloat162raw_to_e8m0x2(__nv_bfloat162_raw x, __nv_saturation_t, cudaRoundMode);

// Decode: E8M0 → bf16 ONLY (no direct → float or → half)
__nv_bfloat16_raw    __nv_cvt_e8m0_to_bf16raw(__nv_fp8_storage_t x);
__nv_bfloat162_raw   __nv_cvt_e8m0x2_to_bf162raw(__nv_fp8x2_storage_t x);
// NOTE: No __nv_cvt x4 functions exist. Use struct operators instead.
// NOTE: No __nv_cvt_e8m0_to_halfraw — must go through bf16 or float.
```

### Saturation and Rounding
- `__NV_SATFINITE`: Clamp ±Inf to max finite E8M0 (254). NaN behavior unspecified.
- `__NV_NOSAT`: No saturation (Inf/NaN may produce E8M0 NaN = 255).
- `cudaRoundNearest`: Round to nearest, ties to even.
- `cudaRoundPosInf`: Round toward positive infinity (used by struct constructors).

### WARNING: __nv_cvt_float_to_e8m0 rounding mismatch
**DO NOT USE `__nv_cvt_float_to_e8m0` as a drop-in for `round(log2(x)) + 127`.**
The intrinsic rounds to the "closest power of two" in LINEAR space (arithmetic
midpoint = 1.5 × 2^n), while our E8M0 scale computation rounds in LOG space
(geometric midpoint = sqrt(2) × 2^n ≈ 1.414 × 2^n). For ~8.6% of positive floats
(those between the geometric and arithmetic midpoints of consecutive powers of 2),
the intrinsic returns an exponent 1 lower than log-space rounding — a 2× scale error.

Example: x = 5.7
- Log-space: log2(5.7)=2.51 → round to 3 → E8M0=130 (scale=8)
- Linear-space: |5.7-4|=1.7 < |5.7-8|=2.3 → nearest is 4 → E8M0=129 (scale=4)

When used in `compute_e8m0_scale` (quantize.cu), this caused catastrophic PPL
regression (11.5 → 62,524,600) because ~8.6% of MXFP4 weight blocks had scales
that were 2× too small, causing quantized values to overflow their FP4 range.
The portable IEEE bit extraction (Schraudolph 1999) is used instead.
The **decode** intrinsics (`__nv_cvt_e8m0_to_bf16raw`, `__nv_cvt_e8m0x2_to_bf162raw`)
are safe — they are simple lookups with no rounding ambiguity.

## Quant Functions (float/half/bf16 → FP8)
```cpp
__nv_fp8_storage_t  __nv_cvt_float_to_fp8(float x, __nv_saturation_t, __nv_fp8_interpretation_t);
__nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(float2 x, __nv_saturation_t, __nv_fp8_interpretation_t);
__nv_fp8_storage_t  __nv_cvt_halfraw_to_fp8(__half_raw x, __nv_saturation_t, __nv_fp8_interpretation_t);
__nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(__half2_raw x, __nv_saturation_t, __nv_fp8_interpretation_t);
__nv_fp8_storage_t  __nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw x, __nv_saturation_t, __nv_fp8_interpretation_t);
__nv_fp8x2_storage_t __nv_cvt_bfloat16raw2_to_fp8x2(__nv_bfloat162_raw x, __nv_saturation_t, __nv_fp8_interpretation_t);
__nv_fp8_storage_t  __nv_cvt_double_to_fp8(double x, __nv_saturation_t, __nv_fp8_interpretation_t);
__nv_fp8x2_storage_t __nv_cvt_double2_to_fp8x2(double2 x, __nv_saturation_t, __nv_fp8_interpretation_t);
```

## Classes

### __nv_fp8_e4m3 / __nv_fp8_e5m2 (single value)
Constructors from: float, double, __half, __nv_bfloat16, int, long, long long, unsigned variants, short.
Conversion operators: float, double, __half, __nv_bfloat16, int, long, long long, short, char, signed/unsigned variants, bool.

### __nv_fp8_e8m0 (single value)
```cpp
// Constructors from: float, double, __half, __nv_bfloat16, int, long, long long,
//   short, unsigned variants. All use __NV_SATFINITE + cudaRoundPosInf.
__nv_fp8_e8m0(const float f);
__nv_fp8_e8m0(const double f);
__nv_fp8_e8m0(const __half f);
__nv_fp8_e8m0(const __nv_bfloat16 f);
__nv_fp8_e8m0(const int val);          // integer constructors use cudaRoundPosInf
// ... and all other integer types

// Conversion operators:
operator float() const;
operator double() const;
operator __half() const;
operator __nv_bfloat16() const;
operator int() const;                  // clamps, NaN → 0
operator bool() const;                 // always returns true
// ... and all other integer types (clamp, NaN → 0)
```

### __nv_fp8x2_e4m3 / __nv_fp8x2_e5m2 (pair)
```cpp
Constructors from: float2, double2, __half2, __nv_bfloat162.
operator float2() const;   // DEQUANT: 2 FP8 → float2
operator __half2() const;  // DEQUANT: 2 FP8 → half2
```

### __nv_fp8x2_e8m0 (pair, uint16_t storage)
```cpp
// Constructors: all use __NV_SATFINITE.
__nv_fp8x2_e8m0(const __half2 f);
__nv_fp8x2_e8m0(const __nv_bfloat162 f);
__nv_fp8x2_e8m0(const double2 f);
__nv_fp8x2_e8m0(const float2 f);

// Decode operators:
operator float2() const;
operator __half2() const;
operator __nv_bfloat162() const;
```

### __nv_fp8x4_e4m3 / __nv_fp8x4_e5m2 (tetrad, uint32_t)
```cpp
__nv_fp8x4_e4m3();
__nv_fp8x4_e4m3(float4 f);               // quantize
__nv_fp8x4_e4m3(double4 f);
__nv_fp8x4_e4m3(double4_16a f);
__nv_fp8x4_e4m3(double4_32a f);
__nv_fp8x4_e4m3(__half2 flo, __half2 fhi);  // quantize
__nv_fp8x4_e4m3(__nv_bfloat162 flo, __nv_bfloat162 fhi);
operator float4() const;                 // DEQUANT: 4 FP8 → float4 ✓
// NOTE: No operator __half4() — only float4 output on x4
```

### __nv_fp8x4_e8m0 (tetrad, uint32_t storage)
```cpp
// Constructors: all use __NV_SATFINITE.
__nv_fp8x4_e8m0(const float4 f);
__nv_fp8x4_e8m0(const double4 f);
__nv_fp8x4_e8m0(const double4_16a f);
__nv_fp8x4_e8m0(const double4_32a f);
__nv_fp8x4_e8m0(const __half2 flo, const __half2 fhi);
__nv_fp8x4_e8m0(const __nv_bfloat162 flo, const __nv_bfloat162 fhi);

// Decode operator:
operator float4() const;   // DEQUANT: 4 E8M0 → float4
// NOTE: No operator __half4 or __half2+__half2. Only float4 output.
// NOTE: No standalone __nv_cvt x4 functions — struct operator is the only x4 path.
```
