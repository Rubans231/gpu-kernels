#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <c10/cuda/CUDAException.h>
#include <stdio.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor") // Checks if x is cuda

#define CHECK_CONTIGUOUS(x)                                                                                            \
    TORCH_CHECK(x.is_contiguous(),                                                                                     \
                #x " must be contiguous") // Checks if mem is allocated contiguous instead of fragmented

#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }
__global__ void matmul_k(float *m, float *n, float *out, int h, int w, int k) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= h || c >= w)
        return;
    float o = 0;
    for (int i = 0; i < k; i++) {
        o += m[r * k + i] * n[i * w + c];
    }
    out[r * w + c] = o;
}

torch::Tensor matmul(torch::Tensor m, torch::Tensor n) {
    CHECK_INPUT(m);
    CHECK_INPUT(n);
    int h = m.size(0);
    int k = m.size(1);
    int w = n.size(1);
    TORCH_CHECK(k == n.size(0), "Size mismatch!");

    auto output = torch::zeros({h, w}, m.options());

    dim3 tpb(16, 16);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));

    matmul_k<<<blocks, tpb>>>(m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, w, k);
    // Check for errors during kernel launch in dev phase
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

#define TILE 16

__global__ void matmul_k_tiled(float *m, float *n, float *out, int h, int w, int k) {
    // shared mem tiles
    __shared__ float tile_m[TILE][TILE];
    __shared__ float tile_n[TILE][TILE];

    // calculate the tile's row and column indices
    int r = blockIdx.y * TILE + threadIdx.y;
    int c = blockIdx.x * TILE + threadIdx.x;

    float o = 0.0f;

    // loop through the inner dimension k
    for (int t = 0; t < (k + TILE - 1) / TILE; t++) {

        // Load M to shared
        int m_col = t * TILE + threadIdx.x;

        if (r < h && m_col < k)
            tile_m[threadIdx.y][threadIdx.x] = m[r * k + m_col];
        else
            tile_m[threadIdx.y][threadIdx.x] = 0.0f;

        // Load N to shared
        int n_row = t * TILE + threadIdx.y;

        if (n_row < k && c < w)
            tile_n[threadIdx.y][threadIdx.x] = n[n_row * w + c];
        else
            tile_n[threadIdx.y][threadIdx.x] = 0.0f;

        // sync and wait whole tile to load
        __syncthreads();

        // out with shared mem now
        for (int i = 0; i < TILE; i++) {
            o += tile_m[threadIdx.y][i] * tile_n[i][threadIdx.x];
        }

        // make sure no overwrites by having each write wait previous read and inference to process
        __syncthreads();
    }

    // Write result to mem
    if (r < h && c < w)
        out[r * w + c] = o;
}
