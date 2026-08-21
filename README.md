# Matmul CUDA Kernel Optimization Log

A record of profiling and iterating on a hand-written CUDA matmul kernel for MNIST (`(50000, 784) @ (784, 10)`), using Nsight Compute (`ncu`) to drive every change. Each doc below covers one stage; every fix was made in response to a specific profiler finding, not guesswork.

## The shape that drives everything

- **M = 50,000** — number of training images

- **K = 784** — pixels per image (28×28)

- **N = 10** — number of classes

The defining fact about this problem is that **N is tiny**. A matmul this skinny doesn't behave like a typical square GEMM, and most of the optimization work here comes from recognizing where generic tiling assumptions break down at this shape, rather than from generic "GPU optimization" tricks.

## Repository layout

```
Python/  
├── .gitattributes                     \# \*.ncu-rep tracked via Git LFS  
├── .gitignore                         \# ignores data/ (MNIST cache)  
├── Inline\_kernel/  
│   ├── Matmul\_naive\_barebones.py      \# isolated naive kernel  
│   ├── Matmul\_tiled\_barebones.py      \# isolated square-tile kernel (stage 1)  
│   ├── Matmul\_tiled\_16\_8\_barebones.py \# asymmetric-tile experiment (stage 1, aside)  
│   ├── Matmul\_warp\_row\_barebones.py   \# final kernel: all fixes applied (stage 4/5)  
│   ├── Matmul\_naive\_dev.py            \# side-by-side benchmark harness, all 3 kernels  
│   └── scratchpad.cu                  \# standalone .cu scratch file for editor tooling  
└── triton\_kernel/                     \# placeholder — no Triton implementation yet
```

`Matmul\_naive\_dev.py` is where the original timing numbers came from (see [Results at a glance](#results-at-a-glance) below), and it's also been the file iterated on directly during debugging — so at any given point its `matmul\_warp\_per\_row\_kernel` or any other kernel for that fact, may lag behind the fully-fixed version that lives in `barebones.py` of the same implementation.

## Iteration 1: Naive (baseline)

1. This is the main starting point with just a simple implementation where we constantly fetch each value from global memory which proved to be twice as slow when compared to the pytorch variant.
2. This was also around the time where I was implicitly comparing against pytorch cpu with the built in time library which also isnt the best when it comes to measuring the GPU time it takes.
3. Later down the line, I had there be a warmup loop before the actual time benchamrks and that was indeed close to the ground truth.
4. Along with the warmup, I turned away from pytorch cpu and rather had benchmarked with pytorch GPU. The results can be seen below:

```
python Matmul_naive_dev.py
[1/2] /opt/cuda/bin/nvcc -MD -MF cuda.cuda.o.d -DTORCH_EXTENSION_NAME=matmul_naive_vs_tiled_vs_warp_per_row -DTORCH_API_INCLUDE_EXTENSION_H -isystem /home/robin/Projects/GPU-mode/Kernels/.venv/lib/python3.14/site-packages/torch/include -isystem /home/robin/Projects/GPU-mode/Kernels/.venv/lib/python3.14/site-packages/torch/include/torch/csrc/api/include -isystem /opt/cuda/include -isystem /usr/include/python3.14 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_89,code=sm_89 --compiler-options '-fPIC' -O2 -std=c++20 -c /home/robin/.cache/torch_extensions/py314_cu130/matmul_naive_vs_tiled_vs_warp_per_row/cuda.cu -o cuda.cuda.o 
[2/2] c++ main.o cuda.cuda.o -shared -L/home/robin/Projects/GPU-mode/Kernels/.venv/lib/python3.14/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/opt/cuda/lib64 -lcudart -o matmul_naive_vs_tiled_vs_warp_per_row.so
================================ No Warmup ======================================
Extension build/load: 24.838s
Pytorch Gpu with no warmup: 45.484ms
Custom kernel with no warmup: 8.488ms

============== Value difference b/w pytorch-cpu and custom kernel ==================

Is the value calculated by pytorch cpu close to the value calculated by my custom kernel:  True
How much is the difference between the two values:  3.4332275390625e-05

================================= After warmup =====================================

Pytorch avg after warmup and multiple iterations: 0.411ms
Custom kernel avg after warmup and multiple iterations: 0.763ms
```

## Iteration 2: Generic tiled GEMM

1. The naive implementation proved a solid base to work from, so as a natural upgrade I chose tiling and usage of shared memory next.

2. (`matmul_k_tiled`): square tiles of TILE=16,
each block cooperatively loading tiles of both input matrices into shared
memory and looping over the K dimension.

**Profiler results (Nsight Compute):**

| Finding | Est. Speedup | Detail |
| --- | --- | --- |
| L1TEX Global Load Access Pattern | 52.96% | Only 5.8 of 32 bytes/sector utilized |
| L1TEX Global Store Access Pattern | 10.76% | Only 26.7 of 32 bytes/sector utilized |
| Low Utilization | 91.04% (local) | SM Busy 23.35%, Issue Slot Busy 20.72% |
| Long Scoreboard Stalls | 35.41% | ~27.6 cycles/instruction waiting on memory |
| Barrier Stalls | 31.99% | ~17.8 cycles/instruction waiting at `__syncthreads()` |

**Root causes identified:**

1. **Sub-warp block width.** `TILE_X = 16` meant `blockDim.x = 16`, a half of a warp. Since threads are linearized with `x` fastest-varying, a single warp spanned **2 different output rows**. Loading the M matrix then required jumping by a stride of `K = 784` elements between sub-groups within the same warp — instead of one coalesced 128-byte transaction, this produced several small, scattered ones. This is the direct cause of the load-pattern finding, and it cascades into the stall findings too:

warps sit idle waiting on those inefficient loads (long scoreboard stalls), and occupancy alone couldn't hide it (already at 96% achieved occupancy, so there was no more parallelism to throw at the problem).

1. `tile= 16` does also mean silently writing past the end of each shared-memory row, which seemed to be slow but later experimentation with different tiles such as multiple dimensions defined as `tile_k == tile_x == 8` and `tile_y = 16` surprisingly showed worse time results. Further tile size experimentations might prove otherwise but the current findings with other sizes proved to be no better either.

2. **imbalanced cooperative load before a barrier.** the shared-memory tile
load for the weight matrix was guarded by `if (threadidx.y < tile)`,
so only some threads did work before `__syncthreads()`. warps that
finished early sat waiting at the barrier for the ones that didn't —
matching the barrier-stall finding directly.

---

## Iteration 2: Warp-per-row redesign

Rather than patching the tiled kernel's block-shape mismatch, the second
kernel (`matmul_warp_per_row`) abandoned N-tiling entirely:

- **One warp computes one full output row.** All 32 lanes stride through
  the row's 784-length K dimension together — `m[row*k + kk]` for
  `kk = lane, lane+32, lane+64, ...` — which is a fully coalesced, contiguous
  access by construction.
- Each lane accumulates partial dot products for all 10 output columns in
  registers as it sweeps K.
- A warp-shuffle reduction (`__shfl_down_sync`) combines the 32 lanes'
  partial sums per column.
- The weight matrix (`~31KB`) was read via `__ldg`, relying on it staying
  resident in L1/L2 given its small size — no explicit shared-memory
  staging yet.

This fixed the M-side coalescing problem outright, since row assignment is
now warp-aligned by construction rather than dependent on `blockDim.x`.

---

## Results at a glance

```
================ CUDA events for pytorch gpu and custom kernels ==================

pytorch GPU time measured with CUDA events: 0.390ms
Custom kernel time measured with CUDA events: 0.719ms
Tiled custom: 0.596ms
Speedup difference: 1.21x
warp_per_row_kernel time measured with CUDA events: 0.409ms
Speedup difference b/w tiled and warp_per_row: 1.46x
```
