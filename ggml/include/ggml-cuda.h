#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

/**
 * Shrink all memory pools across all CUDA backends to release unused
 * physical pages back to the driver (pools only, no async trim).
 * Caller should invoke between inference steps (not concurrently with
 * llama_decode). The VMM pool shrink only releases freed tail memory.
 * This is a no-op if VMM is not enabled or if pools are already empty.
 */
GGML_BACKEND_API void ggml_backend_cuda_pool_shrink_all(void);

/**
 * Shrink all pools AND trim the CUDA async allocator across all devices.
 * ONLY call when no inference is active — async trim can cause latency spikes.
 */
GGML_BACKEND_API void ggml_backend_cuda_shrink_all(void);

/**
 * Trim only the CUDA async allocator pools across all devices (no pool shrink).
 * ONLY call when no inference is active.
 */
GGML_BACKEND_API void ggml_backend_cuda_async_trim_all(void);

/**
 * Get aggregated pool stats for a single device.
 * Returns total pool_size (mapped/allocated) and pool_used (actively held)
 * across all contexts and streams.
 */
GGML_BACKEND_API void ggml_backend_cuda_get_pool_stats(int device, size_t * out_pool_size, size_t * out_pool_used);

#ifdef  __cplusplus
}
#endif
