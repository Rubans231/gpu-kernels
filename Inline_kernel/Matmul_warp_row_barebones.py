import torch
from torch.utils.cpp_extension import load_inline
import gzip, pickle
from urllib.request import urlretrieve
from pathlib import Path
from torch import tensor
# import time
# import os

# MNIST DATA PULL

MNIST_URL = "https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz?raw=true"
path_data = Path("data")
path_data.mkdir(exist_ok=True)
path_gz = path_data / "mnist.pkl.gz"
if not path_gz.exists():
    urlretrieve(MNIST_URL, path_gz)

with gzip.open(path_gz, "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
x_train, y_train, x_valid, y_valid = map(tensor, (x_train, y_train, x_valid, y_valid))
# x_train.shape, x_train.type()

torch.manual_seed(1)
weights = torch.randn(
    784, 10
)  # 28 x 28 = 784(The dataset has images of 28x28) and 10 because there are 10 classes in the dataset and thus requiring 10 cloumns

m1 = x_train
m2 = weights

# Best debugging practice for dev to check for errors
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# CUDA C
cuda_src = r"""
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor") // Checks if x is cuda

#define CHECK_CONTIGUOUS(x)                                                                                            \
    TORCH_CHECK(x.is_contiguous(),                                                                                     \
                #x " must be contiguous") // Checks if mem is allocated contiguous instead of fragmented

#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)

__host__ __device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

#define TILE_K 32   // one warp-width slab of K per tile -- lane index == row-in-tile
#define MAX_N 16    // headroom above this problem's 10 output columns

__global__ __launch_bounds__(256) void matmul_warp_per_row_kernel(const float* __restrict__ m, const float* __restrict__ n, float* __restrict__ out, int h, int w, int k) {
    // Staging area for one K-tile's worth of n's rows. Every warp in the
    // block reads from here instead of gathering scattered rows straight
    // from global memory -- that gather (n[kk*w+c] with kk varying per
    // lane) was the source of the uncoalesced-load finding.
    __shared__ float tile_n[TILE_K][MAX_N + 1];  // +1 padding to dodge bank conflicts

    int warp_in_block = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;

    int row = blockIdx.x * warps_per_block + warp_in_block;
    int lane = threadIdx.x & 31;

    float acc[MAX_N];
    #pragma unroll
    for (int c = 0; c < MAX_N; c++)
        acc[c] = 0.0f;

    int num_tiles = cdiv(k, TILE_K);
    for (int t = 0; t < num_tiles; t++) {
        int tile_base = t * TILE_K;

        // Cooperative load of this K-tile's n rows into shared memory.
        // Flat index over the WHOLE block so every thread does equal
        // work, and consecutive threadIdx.x map to consecutive
        // addresses in n -- this part IS coalesced.
        for (int idx = threadIdx.x; idx < TILE_K * w; idx += blockDim.x) {
            int kk = idx / w;
            int c = idx % w;
            int global_k = tile_base + kk;
            tile_n[kk][c] = (global_k < k) ? n[global_k * w + c] : 0.0f;
        }
        __syncthreads();  // unconditional -- every thread hits this, no deadlock risk

        if (row < h) {
            int global_k = tile_base + lane;
            float x_val = (global_k < k) ? __ldg(&m[row * k + global_k]) : 0.0f;  // coalesced across the warp
            #pragma unroll
            for (int c = 0; c < MAX_N; c++) {
                if (c < w)
                    acc[c] += x_val * tile_n[lane][c];  // shared-memory read, not global
            }
        }
        __syncthreads();  // make sure everyone's done reading before the next tile overwrites it
    }

    if (row < h) {
        // Reduce each column across the warp, then hand column c's result to
        // lane c instead of leaving everything on lane 0 -- that turns w
        // sequential single-thread stores into one coalesced store across
        // lanes 0..w-1.
        float row_result = 0.0f;
        #pragma unroll
        for (int c = 0; c < MAX_N; c++) {

            if (c >= w)
                continue;
            float v = acc[c];

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                v += __shfl_down_sync(0xffffffff, v, offset);   // true sum lands on lane 0
            v = __shfl_sync(0xffffffff, v, 0);                  // broadcast it to every lane
            if (lane == c)
                row_result = v;                                 // lane c keeps "its" column
        }
        if (lane < w)
            out[row * w + lane] = row_result;                   // single coalesced store
    }
}

torch::Tensor matmul_warp_per_row(torch::Tensor m, torch::Tensor n) {
    CHECK_INPUT(m);
    CHECK_INPUT(n);

    int h = m.size(0);
    int k = m.size(1);
    int w = n.size(1);

    TORCH_CHECK(k == n.size(0), "Size mismatch!");
    TORCH_CHECK(w <= MAX_N, "Output width exceeds MAX_N, raise the cap in the kernel");

    auto output = torch::empty({h, w}, m.options());

    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = cdiv(h, warps_per_block);

    matmul_warp_per_row_kernel<<<blocks, threads>>>(m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(),
                                                    h, w, k);

    return output;
}
"""
# Store the method to a variable to call into inline
cpp_src = "torch::Tensor matmul_warp_per_row(torch::Tensor m, torch::Tensor n);"

module = load_inline(
    name="matmul_warp_per_row",
    cuda_sources=[cuda_src],
    cpp_sources=[cpp_src],
    functions=["matmul_warp_per_row"],
    extra_cuda_cflags=["-O2"],
    verbose=True,
)

# m1 and m2 but now stored as contiguous tensors in the gpu instead of cpu
m1c, m2c = m1.contiguous().cuda(), m2.contiguous().cuda()


# ================================================
#       matmul_warp_per_row_kernel
# ================================================


for _ in range(10):
    module.matmul_warp_per_row(m1c, m2c)

torch.cuda.synchronize()

module.matmul_warp_per_row(m1c, m2c)

torch.cuda.synchronize()


# ================================================
#       Conclusion
# ================================================

# Build overhead is a thing in custom inline_cuda kernel but it is a one time thing. For pytorch it is initialization overhead.
# There is two forms of time benchmarking, one is the CPU's perf_counter which does have a bit of overhead though almost negligible, the second is CUDA events which is most accurate in this scenario

# Timing diff is not so bad for NAIVEE way, but lets make it faster, cuz why not
