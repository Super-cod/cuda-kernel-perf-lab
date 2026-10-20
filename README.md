# cuda-kernel-perf-lab

Hand-written **PTX** kernels, loaded and launched through the **CUDA Driver API** (via Python `ctypes`, no CUDA Toolkit or `nvcc`), with microbenchmarks for three classic GPU performance effects: shared-memory tiling, memory coalescing and warp divergence.

All numbers below were measured on the machine described next. They are small educational experiments, not a production benchmark suite.

## Test machine

| | |
| :--- | :--- |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (Ampere, compute capability 8.6, 16 SMs) |
| Driver | 592.27 (Windows 11) |
| Software | Python 3.11, NumPy 2.4, PyTorch 2.11+cu128 (only used for the cuBLAS reference) |
| Precision | FP32 |

## Layout

```
ptx/        gemm_naive.ptx, gemm_tiled_shared.ptx, memory_coalescing.ptx, warp_divergence.ptx
src/        cuda_driver.py (ctypes wrapper over nvcuda.dll), benchmark_runner.py
tests/      test_kernels.py (correctness checks against NumPy)
run_experiments.ps1   runs the tests, then the benchmarks
```

## Results

Each result is the range over 3 full runs on the machine above (run-to-run variation is visible, so single figures would overstate precision). Warm-up is 3 launches before every measurement; times come from CUDA events.

| Experiment | Setup | Result (3 runs) |
| :--- | :--- | :--- |
| **Naive GEMM** | 512x512x512 FP32 (0.27 GFLOP), 16x16 thread blocks, median of 10 launches | 342-346 GFLOPS |
| **Tiled shared-memory GEMM** | same shape, 16x16 shared-memory tiles with `bar.sync` | 455-486 GFLOPS, **1.33-1.41x** faster than naive |
| cuBLAS via PyTorch (reference) | same shape, timed with `perf_counter` + synchronize | 1630-2370 GFLOPS, i.e. **roughly 3.4-5x faster than my tiled kernel** |
| **Coalesced copy** | 16,777,216 floats (64 MB), read + write counted, mean of 10 launches | 152-173 GB/s |
| **Strided copy** (stride 32) | same data | 14.5-15.6 GB/s, so coalesced is **10.4-11.6x** faster |
| **Vectorized copy** (`ld/st.global.v4.f32`) | same data | 152-177 GB/s, **about the same as the scalar coalesced copy** (1.00-1.02x) |
| **Warp divergence** | 2,097,152 elements, 500 iterations per element, mean of 10 launches | divergent warps take **1.90-2.09x** as long as uniform ones |

What these show, and what they don't:
- Tiling helps, but a 16x16 tile with no register blocking reaches only a fraction of cuBLAS. Closing that gap (larger tiles, register tiling, `cp.async`) is not implemented.
- Uncoalesced access costs about 10x on this GPU. Once access is coalesced, the copy is limited by memory bandwidth, so the 128-bit vector version adds nothing here.
- The divergence result is consistent with a two-way branch executing both paths serially (about 2x).
- I have not verified a theoretical peak memory bandwidth for this laptop SKU, so the results are not expressed as a percentage of peak.

## Correctness tests

`tests/test_kernels.py` checks three things on the GPU: `gemm_naive` and `gemm_tiled_shared` against `numpy.matmul` (128x128, `allclose` with `atol=rtol=1e-3`) and exact reproduction of an array by `copy_vectorized_v4`. The strided-copy and divergence kernels are benchmarked but not checked for correctness.

```
[PASS] Test 1: gemm_naive matches NumPy matmul reference (allclose atol=1e-3)
[PASS] Test 2: gemm_tiled_shared matches NumPy matmul reference (allclose atol=1e-3)
[PASS] Test 3: copy_vectorized_v4 exact bit-level array reproduction
```

## Running it

Requirements: an NVIDIA GPU with a recent driver (provides `nvcuda.dll` / `libcuda.so`), Python 3.8+, `numpy` and `torch` (CUDA build). The benchmark runner imports PyTorch for the cuBLAS reference.

```powershell
git clone https://github.com/Super-cod/cuda-kernel-perf-lab.git
cd cuda-kernel-perf-lab
pip install numpy torch
.\run_experiments.ps1          # tests, then benchmarks
```

or run the pieces directly from the repository root:

```bash
python tests/test_kernels.py
python src/benchmark_runner.py
```

Paths to the PTX files are relative, so run from the repository root.

## How it works

1. `cuda_driver.py` loads the driver library and wraps context, module, buffer and event handling.
2. Each `.ptx` file is read as text and JIT-compiled by the driver with `cuModuleLoadData`.
3. Kernels are launched with `cuLaunchKernel`; timing uses `cuEventRecord` / `cuEventElapsedTime`.

## Limitations

- FP32 only; no Tensor Core (WMMA/MMA) kernels.
- Fixed 16x16 tiling, no register blocking, no `cp.async` pipelining.
- GEMM benchmark runs a single 512x512 size; there is no size sweep.
- Results come from one laptop GPU on Windows and will differ on other hardware.
- GEMM is checked at 128x128 by the tests but timed at 512x512.

## License

MIT. See `LICENSE`.
