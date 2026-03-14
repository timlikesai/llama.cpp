 » 3. FP8 Intrinsics » 3.9. FP8 Conversion and Data Movementv13.2 | PDF | Archive  
3.9. FP8 Conversion and Data Movement
To use these functions, include the header file cuda_fp8.h in your program.

Enumerations

__nv_fp8_interpretation_t
Enumerates the possible interpretations of the 8-bit values when referring to them as fp8 types.
__nv_saturation_t
Enumerates the modes applicable when performing a narrowing conversion to fp8 destination types.
Functions

__host__ __device__ __nv_fp8x2_storage_t__nv_cvt_bfloat162raw_to_e8m0x2(const __nv_bfloat162_raw x, const __nv_saturation_t saturate, const enum cudaRoundMode rounding)
Converts a pair of bfloat16 values into a pair of scaling factors of e8m0 kind.
__host__ __device__ __nv_fp8x2_storage_t__nv_cvt_bfloat16raw2_to_fp8x2(const __nv_bfloat162_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input vector of two nv_bfloat16 precision numbers packed in __nv_bfloat162_raw x into a vector of two values of fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.
__host__ __device__ __nv_fp8_storage_t__nv_cvt_bfloat16raw_to_e8m0(const __nv_bfloat16_raw x, const __nv_saturation_t saturate, const enum cudaRoundMode rounding)
Converts input bfloat16 input into a scaling factor of e8m0 kind.
__host__ __device__ __nv_fp8_storage_t__nv_cvt_bfloat16raw_to_fp8(const __nv_bfloat16_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input nv_bfloat16 precision x to fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.
__host__ __device__ __nv_fp8x2_storage_t__nv_cvt_double2_to_e8m0x2(const double2 x, const __nv_saturation_t saturate, const enum cudaRoundMode rounding)
Converts a pair of double values into a pair of scaling factors of e8m0 kind.
__host__ __device__ __nv_fp8x2_storage_t__nv_cvt_double2_to_fp8x2(const double2 x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input vector of two double precision numbers packed in double2 x into a vector of two values of fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.
__host__ __device__ __nv_fp8_storage_t__nv_cvt_double_to_e8m0(const double x, const __nv_saturation_t saturate, const enum cudaRoundMode rounding)
Converts input double value into a scaling factor of e8m0 kind.
__host__ __device__ __nv_fp8_storage_t__nv_cvt_double_to_fp8(const double x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input double precision x to fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.
__host__ __device__ __nv_bfloat16_raw__nv_cvt_e8m0_to_bf16raw(const __nv_fp8_storage_t x)
Converts input scaling factor value of e8m0 kind into bfloat16 .
__host__ __device__ __nv_bfloat162_raw__nv_cvt_e8m0x2_to_bf162raw(const __nv_fp8x2_storage_t x)
Converts input pair of scaling factors of e8m0 kind into a pair of bfloat16 values.
__host__ __device__ __nv_fp8x2_storage_t__nv_cvt_float2_to_e8m0x2(const float2 x, const __nv_saturation_t saturate, const enum cudaRoundMode rounding)
Converts a pair of float values into a pair of scaling factors of e8m0 kind.
__host__ __device__ __nv_fp8x2_storage_t__nv_cvt_float2_to_fp8x2(const float2 x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input vector of two single precision numbers packed in float2 x into a vector of two values of fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.
__host__ __device__ __nv_fp8_storage_t__nv_cvt_float_to_e8m0(const float x, const __nv_saturation_t saturate, const enum cudaRoundMode rounding)
Converts input float value into a scaling factor of e8m0 kind.
__host__ __device__ __nv_fp8_storage_t__nv_cvt_float_to_fp8(const float x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input single precision x to fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.
__host__ __device__ __half_raw__nv_cvt_fp8_to_halfraw(const __nv_fp8_storage_t x, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input fp8 x of the specified kind to half precision.
__host__ __device__ __half2_raw__nv_cvt_fp8x2_to_halfraw2(const __nv_fp8x2_storage_t x, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input vector of two fp8 values of the specified kind to a vector of two half precision values packed in __half2_raw structure.
__host__ __device__ __nv_fp8x2_storage_t__nv_cvt_halfraw2_to_fp8x2(const __half2_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input vector of two half precision numbers packed in __half2_raw x into a vector of two values of fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.
__host__ __device__ __nv_fp8_storage_t__nv_cvt_halfraw_to_fp8(const __half_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input half precision x to fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.
__host__ __device____nv_fp8_e4m3::__nv_fp8_e4m3(const int val)
Constructor from int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e4m3::__nv_fp8_e4m3(const unsigned long long int val)
Constructor from unsigned long long int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e4m3::__nv_fp8_e4m3(const __nv_bfloat16 f)
Constructor from __nv_bfloat16 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e4m3::__nv_fp8_e4m3(const long int val)
Constructor from long int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e4m3::__nv_fp8_e4m3(const long long int val)
Constructor from long long int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__nv_fp8_e4m3::__nv_fp8_e4m3()=default
Constructor by default.
__host__ __device____nv_fp8_e4m3::__nv_fp8_e4m3(const unsigned short int val)
Constructor from unsigned short int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e4m3::__nv_fp8_e4m3(const float f)
Constructor from float data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e4m3::__nv_fp8_e4m3(const __half f)
Constructor from __half data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e4m3::__nv_fp8_e4m3(const unsigned long int val)
Constructor from unsigned long int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e4m3::__nv_fp8_e4m3(const double f)
Constructor from double data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e4m3::__nv_fp8_e4m3(const short int val)
Constructor from short int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e4m3::__nv_fp8_e4m3(const unsigned int val)
Constructor from unsigned int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e4m3::operator __half() const
Conversion operator to __half data type.
__host__ __device____nv_fp8_e4m3::operator __nv_bfloat16() const
Conversion operator to __nv_bfloat16 data type.
__host__ __device____nv_fp8_e4m3::operator bool() const
Conversion operator to bool data type.
__host__ __device____nv_fp8_e4m3::operator char() const
Conversion operator to an implementation defined char data type.
__host__ __device____nv_fp8_e4m3::operator double() const
Conversion operator to double data type.
__host__ __device____nv_fp8_e4m3::operator float() const
Conversion operator to float data type.
__host__ __device____nv_fp8_e4m3::operator int() const
Conversion operator to int data type.
__host__ __device____nv_fp8_e4m3::operator long int() const
Conversion operator to long int data type.
__host__ __device____nv_fp8_e4m3::operator long long int() const
Conversion operator to long long int data type.
__host__ __device____nv_fp8_e4m3::operator short int() const
Conversion operator to short int data type.
__host__ __device____nv_fp8_e4m3::operator signed char() const
Conversion operator to signed char data type.
__host__ __device____nv_fp8_e4m3::operator unsigned char() const
Conversion operator to unsigned char data type.
__host__ __device____nv_fp8_e4m3::operator unsigned int() const
Conversion operator to unsigned int data type.
__host__ __device____nv_fp8_e4m3::operator unsigned long int() const
Conversion operator to unsigned long int data type.
__host__ __device____nv_fp8_e4m3::operator unsigned long long int() const
Conversion operator to unsigned long long int data type.
__host__ __device____nv_fp8_e4m3::operator unsigned short int() const
Conversion operator to unsigned short int data type.
__host__ __device____nv_fp8_e5m2::__nv_fp8_e5m2(const __half f)
Constructor from __half data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e5m2::__nv_fp8_e5m2(const long long int val)
Constructor from long long int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e5m2::__nv_fp8_e5m2(const unsigned int val)
Constructor from unsigned int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e5m2::__nv_fp8_e5m2(const float f)
Constructor from float data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e5m2::__nv_fp8_e5m2(const unsigned short int val)
Constructor from unsigned short int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__nv_fp8_e5m2::__nv_fp8_e5m2()=default
Constructor by default.
__host__ __device____nv_fp8_e5m2::__nv_fp8_e5m2(const int val)
Constructor from int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e5m2::__nv_fp8_e5m2(const long int val)
Constructor from long int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e5m2::__nv_fp8_e5m2(const double f)
Constructor from double data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e5m2::__nv_fp8_e5m2(const unsigned long int val)
Constructor from unsigned long int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e5m2::__nv_fp8_e5m2(const short int val)
Constructor from short int data type.
__host__ __device____nv_fp8_e5m2::__nv_fp8_e5m2(const __nv_bfloat16 f)
Constructor from __nv_bfloat16 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e5m2::__nv_fp8_e5m2(const unsigned long long int val)
Constructor from unsigned long long int data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8_e5m2::operator __half() const
Conversion operator to __half data type.
__host__ __device____nv_fp8_e5m2::operator __nv_bfloat16() const
Conversion operator to __nv_bfloat16 data type.
__host__ __device____nv_fp8_e5m2::operator bool() const
Conversion operator to bool data type.
__host__ __device____nv_fp8_e5m2::operator char() const
Conversion operator to an implementation defined char data type.
__host__ __device____nv_fp8_e5m2::operator double() const
Conversion operator to double data type.
__host__ __device____nv_fp8_e5m2::operator float() const
Conversion operator to float data type.
__host__ __device____nv_fp8_e5m2::operator int() const
Conversion operator to int data type.
__host__ __device____nv_fp8_e5m2::operator long int() const
Conversion operator to long int data type.
__host__ __device____nv_fp8_e5m2::operator long long int() const
Conversion operator to long long int data type.
__host__ __device____nv_fp8_e5m2::operator short int() const
Conversion operator to short int data type.
__host__ __device____nv_fp8_e5m2::operator signed char() const
Conversion operator to signed char data type.
__host__ __device____nv_fp8_e5m2::operator unsigned char() const
Conversion operator to unsigned char data type.
__host__ __device____nv_fp8_e5m2::operator unsigned int() const
Conversion operator to unsigned int data type.
__host__ __device____nv_fp8_e5m2::operator unsigned long int() const
Conversion operator to unsigned long int data type.
__host__ __device____nv_fp8_e5m2::operator unsigned long long int() const
Conversion operator to unsigned long long int data type.
__host__ __device____nv_fp8_e5m2::operator unsigned short int() const
Conversion operator to unsigned short int data type.
__host__ __device____nv_fp8_e8m0::__nv_fp8_e8m0(const long int val)
Constructor from long int data type, relies on cudaRoundPosInf rounding.
__host__ __device____nv_fp8_e8m0::__nv_fp8_e8m0(const int val)
Constructor from int data type, relies on cudaRoundPosInf rounding.
__nv_fp8_e8m0::__nv_fp8_e8m0()=default
Constructor by default.
__host__ __device____nv_fp8_e8m0::__nv_fp8_e8m0(const unsigned int val)
Constructor from unsigned int data type, relies on cudaRoundPosInf rounding.
__host__ __device____nv_fp8_e8m0::__nv_fp8_e8m0(const float f)
Constructor from float data type, relies on __NV_SATFINITE behavior behavior for large input values and cudaRoundPosInf for rounding.
__host__ __device____nv_fp8_e8m0::__nv_fp8_e8m0(const unsigned long long int val)
Constructor from unsigned long long int data type, relies on cudaRoundPosInf rounding.
__host__ __device____nv_fp8_e8m0::__nv_fp8_e8m0(const double f)
Constructor from double data type, relies on __NV_SATFINITE behavior for large input values and cudaRoundPosInf for rounding.
__host__ __device____nv_fp8_e8m0::__nv_fp8_e8m0(const __half f)
Constructor from __half data type, relies on __NV_SATFINITE behavior for large input values and cudaRoundPosInf for rounding.
__host__ __device____nv_fp8_e8m0::__nv_fp8_e8m0(const __nv_bfloat16 f)
Constructor from __nv_bfloat16 data type, relies on __NV_SATFINITE behavior for large input values and cudaRoundPosInf for rounding.
__host__ __device____nv_fp8_e8m0::__nv_fp8_e8m0(const unsigned long int val)
Constructor from unsigned long int data type, relies on cudaRoundPosInf rounding.
__host__ __device____nv_fp8_e8m0::__nv_fp8_e8m0(const unsigned short int val)
Constructor from unsigned short int data type, relies on cudaRoundPosInf rounding.
__host__ __device____nv_fp8_e8m0::__nv_fp8_e8m0(const long long int val)
Constructor from long long int data type, relies on cudaRoundPosInf rounding.
__host__ __device____nv_fp8_e8m0::__nv_fp8_e8m0(const short int val)
Constructor from short int data type, relies on cudaRoundPosInf rounding.
__host__ __device____nv_fp8_e8m0::operator __half() const
Conversion operator to __half data type.
__host__ __device____nv_fp8_e8m0::operator __nv_bfloat16() const
Conversion operator to __nv_bfloat16 data type.
__host__ __device____nv_fp8_e8m0::operator bool() const
Conversion operator to bool data type.
__host__ __device____nv_fp8_e8m0::operator char() const
Conversion operator to an implementation defined char data type.
__host__ __device____nv_fp8_e8m0::operator double() const
Conversion operator to double data type.
__host__ __device____nv_fp8_e8m0::operator float() const
Conversion operator to float data type.
__host__ __device____nv_fp8_e8m0::operator int() const
Conversion operator to int data type.
__host__ __device____nv_fp8_e8m0::operator long int() const
Conversion operator to long int data type.
__host__ __device____nv_fp8_e8m0::operator long long int() const
Conversion operator to long long int data type.
__host__ __device____nv_fp8_e8m0::operator short int() const
Conversion operator to short int data type.
__host__ __device____nv_fp8_e8m0::operator signed char() const
Conversion operator to signed char data type.
__host__ __device____nv_fp8_e8m0::operator unsigned char() const
Conversion operator to unsigned char data type.
__host__ __device____nv_fp8_e8m0::operator unsigned int() const
Conversion operator to unsigned int data type.
__host__ __device____nv_fp8_e8m0::operator unsigned long int() const
Conversion operator to unsigned long int data type.
__host__ __device____nv_fp8_e8m0::operator unsigned long long int() const
Conversion operator to unsigned long long int data type.
__host__ __device____nv_fp8_e8m0::operator unsigned short int() const
Conversion operator to unsigned short int data type.
__nv_fp8x2_e4m3::__nv_fp8x2_e4m3()=default
Constructor by default.
__host__ __device____nv_fp8x2_e4m3::__nv_fp8x2_e4m3(const __nv_bfloat162 f)
Constructor from __nv_bfloat162 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x2_e4m3::__nv_fp8x2_e4m3(const double2 f)
Constructor from double2 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x2_e4m3::__nv_fp8x2_e4m3(const __half2 f)
Constructor from __half2 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x2_e4m3::__nv_fp8x2_e4m3(const float2 f)
Constructor from float2 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x2_e4m3::operator __half2() const
Conversion operator to __half2 data type.
__host__ __device____nv_fp8x2_e4m3::operator float2() const
Conversion operator to float2 data type.
__host__ __device____nv_fp8x2_e5m2::__nv_fp8x2_e5m2(const __nv_bfloat162 f)
Constructor from __nv_bfloat162 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x2_e5m2::__nv_fp8x2_e5m2(const double2 f)
Constructor from double2 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x2_e5m2::__nv_fp8x2_e5m2(const __half2 f)
Constructor from __half2 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__nv_fp8x2_e5m2::__nv_fp8x2_e5m2()=default
Constructor by default.
__host__ __device____nv_fp8x2_e5m2::__nv_fp8x2_e5m2(const float2 f)
Constructor from float2 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x2_e5m2::operator __half2() const
Conversion operator to __half2 data type.
__host__ __device____nv_fp8x2_e5m2::operator float2() const
Conversion operator to float2 data type.
__host__ __device____nv_fp8x2_e8m0::__nv_fp8x2_e8m0(const __half2 f)
Constructor from __half2 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x2_e8m0::__nv_fp8x2_e8m0(const float2 f)
Constructor from float2 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x2_e8m0::__nv_fp8x2_e8m0(const __nv_bfloat162 f)
Constructor from __nv_bfloat162 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x2_e8m0::__nv_fp8x2_e8m0(const double2 f)
Constructor from double2 data type, relies on __NV_SATFINITE behavior for out-of-range values.
__nv_fp8x2_e8m0::__nv_fp8x2_e8m0()=default
Constructor by default.
__host__ __device____nv_fp8x2_e8m0::operator __half2() const
Conversion operator to __half2 data type.
__host__ __device____nv_fp8x2_e8m0::operator __nv_bfloat162() const
Conversion operator to __nv_bfloat162 data type.
__host__ __device____nv_fp8x2_e8m0::operator float2() const
Conversion operator to float2 data type.
__NV_SILENCE_DEPRECATION_END __host__ __device____nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const double4_16a f)
Constructor from double4_16a vector data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const __nv_bfloat162 flo, const __nv_bfloat162 fhi)
Constructor from a pair of __nv_bfloat162 data type values, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const double4_32a f)
Constructor from double4_32a vector data type, relies on __NV_SATFINITE behavior for out-of-range values.
__NV_SILENCE_DEPRECATION_BEGIN __host__ __device____nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const double4 f)
Constructor from double4 vector data type, relies on __NV_SATFINITE behavior for out-of-range values.
__nv_fp8x4_e4m3::__nv_fp8x4_e4m3()=default
Constructor by default.
__host__ __device____nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const float4 f)
Constructor from float4 vector data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x4_e4m3::__nv_fp8x4_e4m3(const __half2 flo, const __half2 fhi)
Constructor from a pair of __half2 data type values, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x4_e4m3::operator float4() const
Conversion operator to float4 vector data type.
__nv_fp8x4_e5m2::__nv_fp8x4_e5m2()=default
Constructor by default.
__host__ __device____nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const double4_32a f)
Constructor from double4_32a vector data type, relies on __NV_SATFINITE behavior for out-of-range values.
__NV_SILENCE_DEPRECATION_END __host__ __device____nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const double4_16a f)
Constructor from double4_16a vector data type, relies on __NV_SATFINITE behavior for out-of-range values.
__NV_SILENCE_DEPRECATION_BEGIN __host__ __device____nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const double4 f)
Constructor from double4 vector data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const float4 f)
Constructor from float4 vector data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const __nv_bfloat162 flo, const __nv_bfloat162 fhi)
Constructor from a pair of __nv_bfloat162 data type values, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x4_e5m2::__nv_fp8x4_e5m2(const __half2 flo, const __half2 fhi)
Constructor from a pair of __half2 data type values, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x4_e5m2::operator float4() const
Conversion operator to float4 vector data type.
__NV_SILENCE_DEPRECATION_BEGIN __host__ __device____nv_fp8x4_e8m0::__nv_fp8x4_e8m0(const double4 f)
Constructor from double4 vector data type, relies on __NV_SATFINITE behavior for out-of-range values.
__NV_SILENCE_DEPRECATION_END __host__ __device____nv_fp8x4_e8m0::__nv_fp8x4_e8m0(const double4_16a f)
Constructor from double4_16a vector data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x4_e8m0::__nv_fp8x4_e8m0(const __half2 flo, const __half2 fhi)
Constructor from a pair of __half2 data type values, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x4_e8m0::__nv_fp8x4_e8m0(const float4 f)
Constructor from float4 vector data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x4_e8m0::__nv_fp8x4_e8m0(const __nv_bfloat162 flo, const __nv_bfloat162 fhi)
Constructor from a pair of __nv_bfloat162 data type values, relies on __NV_SATFINITE behavior for out-of-range values.
__nv_fp8x4_e8m0::__nv_fp8x4_e8m0()=default
Constructor by default.
__host__ __device____nv_fp8x4_e8m0::__nv_fp8x4_e8m0(const double4_32a f)
Constructor from double4_32a vector data type, relies on __NV_SATFINITE behavior for out-of-range values.
__host__ __device____nv_fp8x4_e8m0::operator float4() const
Conversion operator to float4 vector data type.
Typedefs

__nv_fp8_storage_t
8-bit unsigned integer type abstraction used for fp8 floating-point numbers storage.
__nv_fp8x2_storage_t
16-bit unsigned integer type abstraction used for storage of pairs of fp8 floating-point numbers.
__nv_fp8x4_storage_t
32-bit unsigned integer type abstraction used for storage of tetrads of fp8 floating-point numbers.
3.9.1. Enumerations
enum __nv_fp8_interpretation_t
Enumerates the possible interpretations of the 8-bit values when referring to them as fp8 types.

Values:

enumerator __NV_E4M3
Stands for fp8 numbers of e4m3 kind.

enumerator __NV_E5M2
Stands for fp8 numbers of e5m2 kind.

enum __nv_saturation_t
Enumerates the modes applicable when performing a narrowing conversion to fp8 destination types.

Values:

enumerator __NV_NOSAT
Means no saturation to finite is performed when conversion results in rounding values outside the range of destination type.

NOTE: for fp8 type of e4m3 kind, the results that are larger than the maximum representable finite number of the target format become NaN.

enumerator __NV_SATFINITE
Means input larger than the maximum representable finite number MAXNORM of the target format round to the MAXNORM of the same sign as input.

3.9.2. Functions
__host__ __device__ __nv_fp8x2_storage_t __nv_cvt_bfloat162raw_to_e8m0x2(const __nv_bfloat162_raw x, const __nv_saturation_t saturate, const enum cudaRoundMode rounding)
Converts a pair of bfloat16 values into a pair of scaling factors of e8m0 kind.

See also

__nv_cvt_bfloat16raw_to_e8m0() for details of conversion.

Returns
The __nv_fp8x2_storage_t value holds the result of conversion.

__host__ __device__ __nv_fp8x2_storage_t __nv_cvt_bfloat16raw2_to_fp8x2(const __nv_bfloat162_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input vector of two nv_bfloat16 precision numbers packed in __nv_bfloat162_raw x into a vector of two values of fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.

Converts input vector x to a vector of two fp8 values of the kind specified by fp8_interpretation parameter, using round-to-nearest-even rounding and saturation mode specified by saturate parameter.

Returns
The __nv_fp8x2_storage_t value holds the result of conversion.

__host__ __device__ __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_e8m0(const __nv_bfloat16_raw x, const __nv_saturation_t saturate, const enum cudaRoundMode rounding)
Converts input bfloat16 input into a scaling factor of e8m0 kind.

Input number’s absolute value is rounded to the closest power of two in the direction specified via rounding parameter. Rounded results that are smaller than the smallest representable target format number 2^-127 are then clipped to 2^-127. Results that are larger than the largest representable target format number 2^127 are either clipped to 2^127 if saturate equals to __NV_SATFINITE, or convert to NaN otherwise. NaN inputs convert into NaN output, encoded as 0xFF in the target format.

Returns
The __nv_fp8_storage_t value holds the result of conversion.

__host__ __device__ __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_fp8(const __nv_bfloat16_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input nv_bfloat16 precision x to fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.

Converts input x to fp8 type of the kind specified by fp8_interpretation parameter, using round-to-nearest-even rounding and saturation mode specified by saturate parameter.

Returns
The __nv_fp8_storage_t value holds the result of conversion.

__host__ __device__ __nv_fp8x2_storage_t __nv_cvt_double2_to_e8m0x2(const double2 x, const __nv_saturation_t saturate, const enum cudaRoundMode rounding)
Converts a pair of double values into a pair of scaling factors of e8m0 kind.

See also

__nv_cvt_bfloat16raw_to_e8m0() for details of conversion.

Returns
The __nv_fp8x2_storage_t value holds the result of conversion.

__host__ __device__ __nv_fp8x2_storage_t __nv_cvt_double2_to_fp8x2(const double2 x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input vector of two double precision numbers packed in double2 x into a vector of two values of fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.

Converts input vector x to a vector of two fp8 values of the kind specified by fp8_interpretation parameter, using round-to-nearest-even rounding and saturation mode specified by saturate parameter.

Returns
The __nv_fp8x2_storage_t value holds the result of conversion.

__host__ __device__ __nv_fp8_storage_t __nv_cvt_double_to_e8m0(const double x, const __nv_saturation_t saturate, const enum cudaRoundMode rounding)
Converts input double value into a scaling factor of e8m0 kind.

See also

__nv_cvt_bfloat16raw_to_e8m0() for details of conversion.

Returns
The __nv_fp8_storage_t value holds the result of conversion.

__host__ __device__ __nv_fp8_storage_t __nv_cvt_double_to_fp8(const double x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input double precision x to fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.

Converts input x to fp8 type of the kind specified by fp8_interpretation parameter, using round-to-nearest-even rounding and saturation mode specified by saturate parameter.

Returns
The __nv_fp8_storage_t value holds the result of conversion.

__host__ __device__ __nv_bfloat16_raw __nv_cvt_e8m0_to_bf16raw(const __nv_fp8_storage_t x)
Converts input scaling factor value of e8m0 kind into bfloat16.

Input scales are exact powers of two or a NaN value, also representable in the target format.

Returns
The __nv_bfloat16_raw value holds the result of conversion.

__host__ __device__ __nv_bfloat162_raw __nv_cvt_e8m0x2_to_bf162raw(const __nv_fp8x2_storage_t x)
Converts input pair of scaling factors of e8m0 kind into a pair of bfloat16 values.

Returns
The __nv_bfloat162_raw value holds the result of conversion.

__host__ __device__ __nv_fp8x2_storage_t __nv_cvt_float2_to_e8m0x2(const float2 x, const __nv_saturation_t saturate, const enum cudaRoundMode rounding)
Converts a pair of float values into a pair of scaling factors of e8m0 kind.

See also

__nv_cvt_bfloat16raw_to_e8m0() for details of conversion.

Returns
The __nv_fp8x2_storage_t value holds the result of conversion.

__host__ __device__ __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(const float2 x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input vector of two single precision numbers packed in float2 x into a vector of two values of fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.

Converts input vector x to a vector of two fp8 values of the kind specified by fp8_interpretation parameter, using round-to-nearest-even rounding and saturation mode specified by saturate parameter.

Returns
The __nv_fp8x2_storage_t value holds the result of conversion.

__host__ __device__ __nv_fp8_storage_t __nv_cvt_float_to_e8m0(const float x, const __nv_saturation_t saturate, const enum cudaRoundMode rounding)
Converts input float value into a scaling factor of e8m0 kind.

See also

__nv_cvt_bfloat16raw_to_e8m0() for details of conversion.

Returns
The __nv_fp8_storage_t value holds the result of conversion.

__host__ __device__ __nv_fp8_storage_t __nv_cvt_float_to_fp8(const float x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input single precision x to fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.

Converts input x to fp8 type of the kind specified by fp8_interpretation parameter, using round-to-nearest-even rounding and saturation mode specified by saturate parameter.

Returns
The __nv_fp8_storage_t value holds the result of conversion.

__host__ __device__ __half_raw __nv_cvt_fp8_to_halfraw(const __nv_fp8_storage_t x, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input fp8 x of the specified kind to half precision.

Converts input x of fp8 type of the kind specified by fp8_interpretation parameter to half precision.

Returns
The __half_raw value holds the result of conversion.

__host__ __device__ __half2_raw __nv_cvt_fp8x2_to_halfraw2(const __nv_fp8x2_storage_t x, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input vector of two fp8 values of the specified kind to a vector of two half precision values packed in __half2_raw structure.

Converts input vector x of fp8 type of the kind specified by fp8_interpretation parameter to a vector of two half precision values and returns as __half2_raw structure.

Returns
The __half2_raw value holds the result of conversion.

__host__ __device__ __nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(const __half2_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input vector of two half precision numbers packed in __half2_raw x into a vector of two values of fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.

Converts input vector x to a vector of two fp8 values of the kind specified by fp8_interpretation parameter, using round-to-nearest-even rounding and saturation mode specified by saturate parameter.

Returns
The __nv_fp8x2_storage_t value holds the result of conversion.

__host__ __device__ __nv_fp8_storage_t __nv_cvt_halfraw_to_fp8(const __half_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation)
Converts input half precision x to fp8 type of the requested kind using round-to-nearest-even rounding and requested saturation mode.

Converts input x to fp8 type of the kind specified by fp8_interpretation parameter, using round-to-nearest-even rounding and saturation mode specified by saturate parameter.

Returns
The __nv_fp8_storage_t value holds the result of conversion.

3.9.3. Typedefs
typedef unsigned char __nv_fp8_storage_t
8-bit unsigned integer type abstraction used for fp8 floating-point numbers storage.

typedef unsigned short int __nv_fp8x2_storage_t
16-bit unsigned integer type abstraction used for storage of pairs of fp8 floating-point numbers.

typedef unsigned int __nv_fp8x4_storage_t
32-bit unsigned integer type abstraction used for storage of tetrads of fp8 floating-point numbers.

