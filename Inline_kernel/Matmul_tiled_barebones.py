import torch
from torch.utils.cpp_extension import load_inline
import gzip, pickle
from urllib.request import urlretrieve
from pathlib import Path
from torch import tensor
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

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

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

torch::Tensor matmul_tiled(torch::Tensor m, torch::Tensor n) {
    CHECK_INPUT(m);
    CHECK_INPUT(n);

    int h = m.size(0);
    int k = m.size(1);
    int w = n.size(1);

    TORCH_CHECK(k == n.size(0), "Size mismatch!");

    auto output = torch::zeros({h, w}, m.options());

    dim3 tpb(TILE, TILE);
    dim3 blocks(cdiv(w, TILE), cdiv(h, TILE));

    matmul_k_tiled<<<blocks, tpb>>>(m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, w, k);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return output;
}
"""
# Store the method to a variable to call into inline
cpp_src = "torch::Tensor matmul_tiled(torch::Tensor m, torch::Tensor n);"

module = load_inline(
    name="matmul_tiled_bare",
    cuda_sources=[cuda_src],
    cpp_sources=[cpp_src],
    functions=["matmul_tiled"],
    extra_cuda_cflags=["-O2"],
    verbose=True,
)

# m1 and m2 but now stored as contiguous tensors in the gpu instead of cpu
m1c, m2c = m1.contiguous().cuda(), m2.contiguous().cuda()

# ================================================
#       Matmul_tiled
# ================================================

# All warmup compressed in one space
for _ in range(10):
    module.matmul_tiled(m1c, m2c)

torch.cuda.synchronize()

module.matmul_tiled(m1c, m2c)

torch.cuda.synchronize()

# ================================================
#       Conclusion
# ================================================
