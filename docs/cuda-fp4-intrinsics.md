# CUDA FP4 Conversion and Data Movement API
Source: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html

## Type Definitions
```cpp
typedef __nv_fp8_storage_t  __nv_fp4_storage_t;      // uint8_t  — 1 FP4 value (low nibble)
typedef __nv_fp8_storage_t  __nv_fp4x2_storage_t;     // uint8_t  — 2 FP4 values (both nibbles)
typedef __nv_fp8x2_storage_t __nv_fp4x4_storage_t;    // uint16_t — 4 FP4 values (2 bytes)
```

## Enumeration
```cpp
enum __nv_fp4_interpretation_t { __NV_E2M1 };
```

## Dequant Functions (FP4 → half)
```cpp
__half_raw  __nv_cvt_fp4_to_halfraw(const __nv_fp4_storage_t x, const __nv_fp4_interpretation_t fp4_interpretation);
__half2_raw __nv_cvt_fp4x2_to_halfraw2(const __nv_fp4x2_storage_t x, const __nv_fp4_interpretation_t fp4_interpretation);
// NOTE: No fp4x4 → float4 dequant function exists. x4 types are quantization-only.
```

## Quant Functions (float/half/bf16 → FP4)
```cpp
__nv_fp4_storage_t  __nv_cvt_float_to_fp4(float x, __nv_fp4_interpretation_t, cudaRoundMode);
__nv_fp4x2_storage_t __nv_cvt_float2_to_fp4x2(float2 x, __nv_fp4_interpretation_t, cudaRoundMode);
__nv_fp4_storage_t  __nv_cvt_halfraw_to_fp4(__half_raw x, __nv_fp4_interpretation_t, cudaRoundMode);
__nv_fp4x2_storage_t __nv_cvt_halfraw2_to_fp4x2(__half2_raw x, __nv_fp4_interpretation_t, cudaRoundMode);
__nv_fp4_storage_t  __nv_cvt_bfloat16raw_to_fp4(__nv_bfloat16_raw x, __nv_fp4_interpretation_t, cudaRoundMode);
__nv_fp4x2_storage_t __nv_cvt_bfloat16raw2_to_fp4x2(__nv_bfloat162_raw x, __nv_fp4_interpretation_t, cudaRoundMode);
__nv_fp4_storage_t  __nv_cvt_double_to_fp4(double x, __nv_fp4_interpretation_t, cudaRoundMode);
__nv_fp4x2_storage_t __nv_cvt_double2_to_fp4x2(double2 x, __nv_fp4_interpretation_t, cudaRoundMode);
```

## Classes

### __nv_fp4_e2m1 (single value)
Constructors from: float, double, __half, __nv_bfloat16, int, long, long long, unsigned variants, short.
No conversion operators listed — use __nv_cvt_fp4_to_halfraw for dequant.

### __nv_fp4x2_e2m1 (pair, 1 byte)
```cpp
__nv_fp4x2_e2m1();
__nv_fp4x2_e2m1(const float2 f);
__nv_fp4x2_e2m1(const double2 f);
__nv_fp4x2_e2m1(const __half2 f);
__nv_fp4x2_e2m1(const __nv_bfloat162 f);
// No conversion operators — use __nv_cvt_fp4x2_to_halfraw2 for dequant.
```

### __nv_fp4x4_e2m1 (tetrad, 2 bytes = uint16_t)
```cpp
__nv_fp4x4_e2m1();
__nv_fp4x4_e2m1(const float4 f);               // quantize float4 → fp4x4
__nv_fp4x4_e2m1(const double4 f);
__nv_fp4x4_e2m1(const double4_16a f);
__nv_fp4x4_e2m1(const double4_32a f);
__nv_fp4x4_e2m1(const __half2 flo, const __half2 fhi);  // quantize half2+half2 → fp4x4
__nv_fp4x4_e2m1(const __nv_bfloat162 flo, const __nv_bfloat162 fhi);
// NOTE: NO operator float4() or similar — x4 is QUANTIZATION ONLY for FP4.
```
