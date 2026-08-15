import torch, os, math
from torch.utils.cpp_extension import load_inline
import gzip, pickle
from urllib.request import urlretrieve
from pathlib import Path
from torch import tensor

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
x_train.shape, x_train.type()

imgs = x_train.reshape((-1, 28, 28))
imgs.shape

torch.manual_seed(1)
weights = torch.randn(784, 10)
weights

m1 = x_train
m2 = weights
m1.shape, m2.shape
tr = m1 @ m2

# Best debugging practice for dev to check for errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# CUDA C
cuda_src = r"""
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor") // Checks if x is cuda
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous") // Checks if mem is allocated contiguous instead of fragmented
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
__global__ void matmul_k(float* m, float* n, float* out, int h, int w, int k) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r>=h || c>=w) return;
    float o = 0;
    for (int i = 0; i < k; i++){
        o += m[r*k + i] * n[i*w +c];
    }
    out[r*w + c] = o;
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

    matmul_k<<<blocks, tpb>>>(
        m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr(), h, w, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
"""
# Store the method to a variable to call into inline
cpp_src = "torch::Tensor matmul(torch::Tensor m, torch::Tensor n);"

# Load_inline
module = load_inline(
    name="matmul_c",
    cuda_sources=[cuda_src],
    cpp_sources=[cpp_src],
    functions=["matmul"],
    extra_cuda_cflags=["-02"],
    verbose=True,
)

# m1 and m2 but now stored contiguous in cuda instead of cpu
m1c, m2c = m1.contiguous().cuda(), m2.contiguous().cuda()
