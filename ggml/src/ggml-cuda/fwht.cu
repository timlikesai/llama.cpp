#include "common.cuh"
#include "fwht.cuh"

template <int N>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const float * src, float * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    // for N < warp_size (e.g. 32 on 64-wide waves) one warp does warp_size/N rows using N-wide shuffles
    constexpr int lanes_per_row = N < warp_size ? N : warp_size;
    constexpr int rows_per_warp = warp_size / lanes_per_row;

    const int lane = threadIdx.x % lanes_per_row;

    const int64_t r0 = ((int64_t) blockIdx.x * blockDim.y + threadIdx.y) * rows_per_warp + threadIdx.x / lanes_per_row;
    // clamp instead of early return so all lanes reach the shuffles
    const int64_t r = r0 < n_rows ? r0 : n_rows - 1;

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / lanes_per_row;
    float     reg[el_w];

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * lanes_per_row + lane] * scale;
    }

#pragma unroll
    for (int h = 1; h < lanes_per_row; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, lanes_per_row);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = lanes_per_row; h < N; h *= 2) {
        const int step = h / lanes_per_row;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    if (r0 < n_rows) {
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            dst[i * lanes_per_row + lane] = reg[i];
        }
    }
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const float * src_d = (const float *) src->data;
    float *       dst_d = (float *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;
    const int rows_per_warp  = n < warp_size ? warp_size / n : 1;

    const int64_t num_blocks = (rows + rows_per_block * rows_per_warp - 1) / (rows_per_block * rows_per_warp);

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const float scale = 1 / sqrtf(n);

    switch (n) {
        case 32:
            ggml_cuda_kernel_launch(fwht_cuda<32>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 64:
            ggml_cuda_kernel_launch(fwht_cuda<64>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 128:
            ggml_cuda_kernel_launch(fwht_cuda<128>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 256:
            ggml_cuda_kernel_launch(fwht_cuda<256>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 512:
            ggml_cuda_kernel_launch(fwht_cuda<512>, launch_params, src_d, dst_d, rows, scale);
            return true;
        default:
            return false;
    }
}
