# CUDA FP6 Conversion and Data Movement API
Source: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6__MISC.html

## Type Definitions
```cpp
typedef __nv_fp8_storage_t   __nv_fp6_storage_t;     // uint8_t  — 1 FP6 value (byte-padded)
typedef __nv_fp8x2_storage_t __nv_fp6x2_storage_t;   // uint16_t — 2 FP6 values
typedef __nv_fp8x4_storage_t __nv_fp6x4_storage_t;   // uint32_t — 4 FP6 values
```

## Enumeration
```cpp
enum __nv_fp6_interpretation_t { __NV_E2M3, __NV_E3M2 };
```

## Dequant Functions (FP6 → half)
```cpp
__half_raw  __nv_cvt_fp6_to_halfraw(__nv_fp6_storage_t x, __nv_fp6_interpretation_t);
__half2_raw __nv_cvt_fp6x2_to_halfraw2(__nv_fp6x2_storage_t x, __nv_fp6_interpretation_t);
// NOTE: No fp6x4 → float4 dequant function or operator. x4 types are quantization-only.
```

## Quant Functions (float/half/bf16 → FP6)
```cpp
__nv_fp6_storage_t  __nv_cvt_float_to_fp6(float x, __nv_fp6_interpretation_t, cudaRoundMode);
__nv_fp6x2_storage_t __nv_cvt_float2_to_fp6x2(float2 x, __nv_fp6_interpretation_t, cudaRoundMode);
__nv_fp6_storage_t  __nv_cvt_halfraw_to_fp6(__half_raw x, __nv_fp6_interpretation_t, cudaRoundMode);
__nv_fp6x2_storage_t __nv_cvt_halfraw2_to_fp6x2(__half2_raw x, __nv_fp6_interpretation_t, cudaRoundMode);
__nv_fp6_storage_t  __nv_cvt_bfloat16raw_to_fp6(__nv_bfloat16_raw x, __nv_fp6_interpretation_t, cudaRoundMode);
__nv_fp6x2_storage_t __nv_cvt_bfloat16raw2_to_fp6x2(__nv_bfloat162_raw x, __nv_fp6_interpretation_t, cudaRoundMode);
__nv_fp6_storage_t  __nv_cvt_double_to_fp6(double x, __nv_fp6_interpretation_t, cudaRoundMode);
__nv_fp6x2_storage_t __nv_cvt_double2_to_fp6x2(double2 x, __nv_fp6_interpretation_t, cudaRoundMode);
```

## Classes

### __nv_fp6_e2m3 / __nv_fp6_e3m2 (single value)
Constructors from: float, double, __half, __nv_bfloat16, int, long, long long, unsigned variants, short.
No conversion operators listed — use __nv_cvt_fp6_to_halfraw for dequant.

### __nv_fp6x2_e2m3 / __nv_fp6x2_e3m2 (pair)
```cpp
Constructors from: float2, double2, __half2, __nv_bfloat162.
// NOTE: No operator float2() or operator __half2() listed in docs.
// For dequant, use __nv_cvt_fp6x2_to_halfraw2.
```

### __nv_fp6x4_e2m3 / __nv_fp6x4_e3m2 (tetrad, uint32_t)
```cpp
__nv_fp6x4_e2m3();
__nv_fp6x4_e2m3(const float4 f);               // quantize
__nv_fp6x4_e2m3(const double4 f);
__nv_fp6x4_e2m3(const double4_16a f);
__nv_fp6x4_e2m3(const double4_32a f);
__nv_fp6x4_e2m3(const __half2 flo, const __half2 fhi);  // quantize
__nv_fp6x4_e2m3(const __nv_bfloat162 flo, const __nv_bfloat162 fhi);
// NOTE: NO operator float4() — x4 is QUANTIZATION ONLY for FP6.
// For dequant, must use 2x __nv_cvt_fp6x2_to_halfraw2 on the low/high halves.
```
