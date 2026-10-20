# NVIDIA CUDA & PTX Kernel Performance Laboratory (`cuda-kernel-perf-lab`)

[![Target](https://img.shields.io/badge/GPU-NVIDIA%20GeForce%20RTX%203050-76B900.svg?logo=nvidia)](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3050/)
[![Architecture](https://img.shields.io/badge/Microarchitecture-Ampere%20SM%208.6-76B900.svg)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
[![Driver API](https://img.shields.io/badge/Interface-nvcuda.dll%20(Pure%20Driver%20API)-blue.svg)]()
[![Tests](https://img.shields.io/badge/Correctness-Verified%20(%CE%94%20%3C%201e--5)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A systems and GPU microarchitecture engineering laboratory dissecting low-level **NVIDIA PTX (Parallel Thread Execution) Virtual ISA** programming, memory coalescing, shared memory bank reuse, warp divergence serialization, and hardware event profiling. All experiments are executed and profiled directly on physical GPU silicon: the **NVIDIA GeForce RTX 3050 Laptop GPU (Ampere SM 8.6)**.

---

## Hardware Target Profile

Empirically queried from the physical device via the NVIDIA CUDA Driver API:

```
=================================================================
  NVIDIA CUDA & PTX KERNEL PERFORMANCE LABORATORY (RTX 3050)     
=================================================================
Device Name           : NVIDIA GeForce RTX 3050 Laptop GPU
Microarchitecture     : NVIDIA Ampere (GA107, Compute Capability 8.6)
Streaming Multiprocs  : 16 SMs (2048 CUDA Cores)
Warp Size             : 32 threads
Max Threads per Block : 1024
Memory Bus Width      : 128-bit GDDR6 (Peak Theoretical: ~192 GB/s)
Driver API Runtime    : nvcuda.dll (JIT PTX Virtual ISA Compiler)
=================================================================
```

---

## System Architecture: Direct Driver API Interface

Unlike typical CUDA applications relying on high-level `nvcc` runtime wrappers and bloated CUDA Toolkits, this laboratory interfaces directly with the NVIDIA Driver (`nvcuda.dll` on Windows / `libcuda.so` on Linux) via Python `ctypes`:

```
+-------------------------------------------------------------------+
|               Hand-Authored NVIDIA PTX Virtual ISA               |
|   (gemm_naive.ptx, gemm_tiled_shared.ptx, memory_coalescing.ptx)  |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|         NVIDIA CUDA Driver API (nvcuda.dll cuModuleLoadData)      |
|               On-the-fly JIT Compilation to SASS                 |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|           NVIDIA GeForce RTX 3050 Physical GPU Silicon            |
|       (16 Streaming Multiprocessors, Tensor Cores, GDDR6)         |
+-------------------------------------------------------------------+
```

---

## Experimental Suites & Empirical Benchmarks

### Experiment 1 & 2: Naive vs Tiled Shared-Memory Matrix Multiplication (GEMM)
* **Concept:** Multiplies two $512 \times 512$ single-precision floating-point matrices ($C = A \times B$, 0.27 GFLOP).
* **Microarchitectural Mechanics:**
  * **Naive Global GEMM (`gemm_naive.ptx`):** Each thread calculates one element $C[i, j]$ by streaming $2K$ operands directly from high-latency global DRAM memory. DRAM bandwidth becomes the primary execution bottleneck.
  * **Tiled Shared-Memory GEMM (`gemm_tiled_shared.ptx`):** Thread blocks ($16 \times 16$) collaboratively stage sub-matrices into on-chip `.shared` SRAM (1024 bytes per tile). Operands loaded from DRAM are reused 16 times across the thread block, reducing global memory requests by **16x**.
* **Empirical Results (Physical RTX 3050 GPU):**
  * Naive Global Memory GEMM: **0.774 ms** | **347.03 GFLOPS**
  * Tiled Shared-Memory GEMM: **0.552 ms** | **486.03 GFLOPS**
  * **Shared Memory Throughput Boost: 1.40x faster** than naive DRAM streaming.

```
Global Memory vs Shared Memory Tiling:

Naive GEMM (Every Multiply Hits High-Latency DRAM):
Thread (i,j) ---> [ DRAM Read A[i,k] ] + [ DRAM Read B[k,j] ] (Repeated K times)

Tiled GEMM (Cooperative DRAM Staging to Low-Latency L1/Shared SRAM):
Step 1: 256 Threads load 16x16 tile into on-chip s_A and s_B
Step 2: bar.sync 0 (Hardware warp barrier)
Step 3: 256 Threads execute 16 partial dot products from fast SRAM
Step 4: bar.sync 0 (Prepare for next tile)
```

---

### Experiment 3: Memory Coalescing & Vectorized 128-bit Loads
* **Concept:** Quantifies memory bus efficiency across 16,777,216 floating-point values (64 MB payload, 128 MB total read/write transfer traffic).
* **Microarchitectural Mechanics:**
  * **Coalesced 32-bit (`copy_coalesced.ptx`):** Adjacent threads in a 32-thread warp access adjacent 4-byte memory addresses. The memory controller merges these 32 individual requests into a single 128-byte cache line burst.
  * **Strided Access (`copy_strided.ptx`, Stride-32):** Each thread accesses every 32nd float (jumping 128 bytes apart). The memory controller cannot coalesce, issuing 32 separate, partially utilized DRAM transactions for each warp instruction.
  * **Vectorized 128-bit (`copy_vectorized_v4.ptx`):** Uses `.v4.f32` instructions (`ld.global.v4.f32` and `st.global.v4.f32`) to process 128 bits (16 bytes) per instruction, maximizing memory controller pipeline saturation.
* **Empirical Results:**
  * Coalesced 32-bit Access: **0.784 ms** | **171.29 GB/s**
  * Strided Access (Stride-32): **8.524 ms** | **15.75 GB/s**
  * Vectorized 128-bit (`v4.f32`): **0.766 ms** | **175.14 GB/s**
  * **Coalescing Throughput Advantage: 10.88x faster** (saturating 91% of physical 192 GB/s peak bus bandwidth!).

---

### Experiment 4: Warp Divergence & SIMT Execution Serialization
* **Concept:** Quantifies the hardware execution penalty when threads within a 32-thread warp execute divergent execution paths across 2,097,152 elements.
* **Microarchitectural Mechanics:**
  * GPU cores execute instructions at warp granularity (SIMT). If half the threads in a warp evaluate a conditional branch as `true` and the other half as `false`, the hardware **cannot execute both paths simultaneously**.
  * The Warp Scheduler serializes execution: it executes Path A while setting the thread active mask to zero for dissenting threads, then flips the active mask and executes Path B.
* **Empirical Results:**
  * Uniform Warp Branching (all 32 threads take identical path): **0.660 ms**
  * Divergent Warp Branching (intra-warp even/odd thread branching): **1.267 ms**
  * **Warp Divergence Penalty: 1.92x slower** (verifying the theoretical 2x execution serialization limit of a two-way branch!).

---

## Setup & Execution Instructions

### Prerequisites
* **Operating System:** Windows 10/11 x64 or Linux x86_64.
* **NVIDIA Driver:** Standard GeForce or Studio driver installed (provides `nvcuda.dll` / `libcuda.so`).
* **Python:** Python 3.8+ with `numpy`.
* *Note: No CUDA Toolkit (`nvcc`) installation is required!*

### Running the Complete Suite

Clone and run the complete laboratory in PowerShell:

```powershell
Set-Location "C:\Users\swaya\Desktop\TIMEPASS\claude\git\cuda-kernel-perf-lab"
.\run_experiments.ps1
```

Or execute directly via Python:

```bash
# 1. Run numerical correctness tests (validates PTX output against NumPy reference)
python tests/test_kernels.py

# 2. Run hardware microbenchmark suite with microsecond event timing
python src/benchmark_runner.py
```

### Verified Test Outputs

```
=================================================================
  NVIDIA CUDA & PTX KERNEL PERFORMANCE LABORATORY (RTX 3050)     
=================================================================
Device Name           : NVIDIA GeForce RTX 3050 Laptop GPU
Microarchitecture     : NVIDIA Ampere (GA107, Compute Capability 8.6)
Streaming Multiprocs  : 16 SMs (2048 CUDA Cores)
Warp Size             : 32 threads
Max Threads per Block : 1024
Memory Bus Width      : 128-bit GDDR6 (Peak Theoretical: ~192 GB/s)
Driver API Runtime    : nvcuda.dll (JIT PTX Virtual ISA Compiler)
=================================================================

[*] Running Numerical Validation Tests:
  - Naive GEMM Correctness               : PASS (max diff: 4.768e-07)
  - Tiled Shared Memory GEMM Correctness : PASS (max diff: 4.768e-07)
  - Coalesced Copy Correctness           : PASS (exact bit-level match)
  - Vectorized v4.f32 Copy Correctness   : PASS (exact bit-level match)
  - Warp Divergence Correctness          : PASS (exact bit-level match)

[*] Running Hardware Microbenchmarks:
  [Exp 1] Naive Global GEMM (512x512)     : 0.774 ms | 347.03 GFLOPS
  [Exp 2] Tiled Shared GEMM (512x512)     : 0.552 ms | 486.03 GFLOPS (1.40x boost)
  [Exp 3] Coalesced 32-bit Memory Copy   : 0.784 ms | 171.29 GB/s
  [Exp 3] Strided Memory Copy (Stride-32): 8.524 ms |  15.75 GB/s
  [Exp 3] Vectorized v4.f32 Memory Copy  : 0.766 ms | 175.14 GB/s (10.88x vs strided)
  [Exp 4] Uniform Warp Execution         : 0.660 ms
  [Exp 4] Divergent Warp Execution       : 1.267 ms (1.92x serialization penalty)
=================================================================
```

---

## Honest Limitations & Engineering Boundaries

| Dimension | Implemented Scope | Out of Current Scope / Future Enhancements |
| :--- | :--- | :--- |
| **Precision** | Single-precision FP32 | FP16/BF16 Tensor Core WMMA/MMA instructions |
| **Tile Dimensions** | Static $16 \times 16$ threads/tile for GEMM | Dynamically tuned hierarchical multi-level tiles ($128 \times 128$) |
| **Pipelining** | Synchronous cooperative SRAM copy (`bar.sync`) | Ampere asynchronous copy (`cp.async`) multi-stage software pipelining |
| **Target Architecture** | NVIDIA Ampere (SM 8.6, GA107) | Architecture portability to pre-Volta (SM < 7.0) without PTX target adjustments |
| **Scope** | Controlled microarchitectural isolation | Production BLAS library competition (e.g. cuBLAS / CUTLASS) |

---

## Relevance to NVIDIA PTX Compiler Engineering

This laboratory models the exact optimization challenges faced by NVIDIA PTX and SASS compiler passes:
1. **Instruction Selection:** Emitting wide 128-bit vector instructions (`ld.param.v4.f32`, `ld.global.v4.f32`) to saturate the memory subsystem.
2. **Register Allocation vs Shared Memory:** Balancing register pressure against shared memory occupancy to maximize active warps per SM.
3. **Branch Predication:** Converting divergent branches into predicated instructions (`@%p bra` vs predicated arithmetic) to eliminate warp divergence serialization.
4. **Synchronization Barriers:** Placing `bar.sync` directives correctly to avoid race conditions without introducing unnecessary pipeline stalls.

---

## License

MIT License. Copyright (c) 2026 Swayam.
