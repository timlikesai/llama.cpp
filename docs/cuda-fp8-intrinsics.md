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

## E8M0 Functions
```cpp
__nv_fp8_storage_t  __nv_cvt_float_to_e8m0(float x, __nv_saturation_t, cudaRoundMode);
__nv_fp8_storage_t  __nv_cvt_double_to_e8m0(double x, __nv_saturation_t, cudaRoundMode);
__nv_fp8_storage_t  __nv_cvt_bfloat16raw_to_e8m0(__nv_bfloat16_raw x, __nv_saturation_t, cudaRoundMode);
__nv_fp8x2_storage_t __nv_cvt_float2_to_e8m0x2(float2 x, __nv_saturation_t, cudaRoundMode);
__nv_fp8x2_storage_t __nv_cvt_double2_to_e8m0x2(double2 x, __nv_saturation_t, cudaRoundMode);
__nv_fp8x2_storage_t __nv_cvt_bfloat162raw_to_e8m0x2(__nv_bfloat162_raw x, __nv_saturation_t, cudaRoundMode);
__nv_bfloat16_raw  __nv_cvt_e8m0_to_bf16raw(__nv_fp8_storage_t x);
__nv_bfloat162_raw __nv_cvt_e8m0x2_to_bf162raw(__nv_fp8x2_storage_t x);
```

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
Same constructors and conversion operators as above.

### __nv_fp8x2_e4m3 / __nv_fp8x2_e5m2 (pair)
```cpp
Constructors from: float2, double2, __half2, __nv_bfloat162.
operator float2() const;   // DEQUANT: 2 FP8 → float2
operator __half2() const;  // DEQUANT: 2 FP8 → half2
```

### __nv_fp8x2_e8m0 (pair)
```cpp
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

### __nv_fp8x4_e8m0 (tetrad)
```cpp
operator float4() const;   // DEQUANT: 4 E8M0 → float4
```
