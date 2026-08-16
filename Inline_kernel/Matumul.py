import torch, os
from torch.utils.cpp_extension import load_inline
import gzip, pickle
from urllib.request import urlretrieve
from pathlib import Path
from torch import tensor
import time

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

torch.manual_seed(1)
weights = torch.randn(
    784, 10
)  # 28 x 28 = 784(The dataset has images of 28x28) and 10 because there are 10 classes in the dataset and thus requiring 10 cloumns

m1 = x_train
m2 = weights

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
        m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, w, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); // Check for errors during kernel launch in dev phase
    return output;
}
"""
# Store the method to a variable to call into inline
cpp_src = "torch::Tensor matmul(torch::Tensor m, torch::Tensor n);"

# Timed build
build_start = time.perf_counter()

module = load_inline(
    name="matmul_c",
    cuda_sources=[cuda_src],
    cpp_sources=[cpp_src],
    functions=["matmul"],
    extra_cuda_cflags=["-O2"],
    verbose=True,
)

build_end = time.perf_counter()

print(
    "================================ No Warmup ======================================"
)

print(f"Extension build/load: {(build_end - build_start):.3f}s")

# ================================================
#       Timed with no warmup
# ================================================

# timed cpu pytorch matmul
start = time.perf_counter()
tr = m1 @ m2
end = time.perf_counter()

# m1 and m2 but now stored as contiguous tensors in the gpu instead of cpu
m1c, m2c = m1.contiguous().cuda(), m2.contiguous().cuda()

# timed gpu pytorch matmul
# No warmup in this case means gpu-libraries initialization overhead time
torch.cuda.synchronize()
start = time.perf_counter()

tr_gpu = m1c @ m2c

torch.cuda.synchronize()
end = time.perf_counter()

print(f"Pytorch Gpu with no warmup: {(end - start) * 1000:.3f}ms")

# timed Custom kernel matmul
torch.cuda.synchronize()
start = time.perf_counter()

kernelcuda = module.matmul(m1c, m2c)

torch.cuda.synchronize()
end = time.perf_counter()

print(f"Custom kernel with no warmup: {(end - start) * 1000:.3f}ms")
print(
    "\n============== Value difference b/w pytorch-cpu and custom kernel ==================\n"
)

# abs difference between the value calculated by cuda and pytorch cpu
torch.cuda.synchronize()
print(
    "Is the value calculated by pytorch cpu close to the value calculated by my custom kernel: ",
    torch.allclose(kernelcuda.cpu(), tr, atol=1e-3),
)
print(
    "How much is the difference between the two values: ",
    (kernelcuda.cpu() - tr).abs().max().item(),
)

print(
    "\n================================= After warmup =====================================\n"
)

# ================================================
#       warmup
# ================================================

for _ in range(10):
    m1c @ m2c

for _ in range(10):
    module.matmul(m1c, m2c)

torch.cuda.synchronize()

# ================================================
#      CPU Timed and averaged for pytorch after warmup using perf_counter
# ================================================

start = time.perf_counter()

for _ in range(100):
    m1c @ m2c

torch.cuda.synchronize()
end = time.perf_counter()

print(
    f"Pytorch avg after warmup and multiple iterations: {(end - start) / 100 * 1000:.3f}ms"
)

# ================================================
#      CPU Timed and averaged for custom kernel after warmup using perf_counter
# ================================================

start = time.perf_counter()

for _ in range(100):
    module.matmul(m1c, m2c)

torch.cuda.synchronize()

end = time.perf_counter()
print(
    f"Custom kernel avg after warmup and multiple iterations: {(end - start) / 100 * 1000:.3f}ms"
)

# ================================================
#       CUDA events Timing instead of perf_counter for pytorch GPU
# ================================================

print(
    "\n================ CUDA events for pytorch gpu and custom kernel ==================\n"
)

for _ in range(10):
    m1c @ m2c

torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

for _ in range(100):
    m1c @ m2c

end.record()

torch.cuda.synchronize()

print(
    f"pytorch GPU time measured with CUDA events: {start.elapsed_time(end) / 100:.3f}ms"
)

# ================================================
#       CUDA events Timing for Custom kernel
# ================================================

for _ in range(10):
    module.matmul(m1c, m2c)

torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()

for _ in range(100):
    module.matmul(m1c, m2c)

end.record()

torch.cuda.synchronize()

print(
    f"Custom kernel time measured with CUDA events: {start.elapsed_time(end) / 100:.3f}ms"
)

# ================================================
#       Conclusion
# ================================================

# Build overhead is a thing in custom inline_cuda kernel but it is a one time thing. For pytorch it is initialization overhead.
# There is two forms of time benchmarking, one is the CPU's perf_counter which does have a bit of overhead though almost negligible, the second is CUDA events which is most accurate in this scenario

# Timing diff is not so bad for NAIVEE way, but lets make it faster, cuz why not
