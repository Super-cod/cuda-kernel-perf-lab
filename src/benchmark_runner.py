"""
Comprehensive Microbenchmark Suite for NVIDIA PTX GPU Kernels
Benchmarks Custom PTX vs PyTorch/cuBLAS on the physical NVIDIA GeForce RTX 3050.
"""

import os
import sys
import time
import ctypes
from ctypes import c_uint, c_void_p, byref
import numpy as np
import torch

from cuda_driver import (
    CUDADriver, CUDADevice, CUDAContext, CUDAModule,
    CUDABuffer, CUDAEvent
)

def benchmark_gemm(driver, ctx, size=1024, warmup=3, iterations=10):
    print(f"\n=================================================================")
    print(f"  EXPERIMENT 1 & 2: Naive vs Tiled Shared Memory GEMM ({size}x{size})")
    print(f"=================================================================")

    M, N, K = size, size, size
    flops = 2.0 * M * N * K

    h_A = np.random.randn(M, K).astype(np.float32)
    h_B = np.random.randn(K, N).astype(np.float32)

    buf_A = CUDABuffer(driver, h_A.nbytes)
    buf_B = CUDABuffer(driver, h_B.nbytes)
    buf_C = CUDABuffer(driver, h_A.nbytes)

    buf_A.copy_from_host(h_A)
    buf_B.copy_from_host(h_B)

    # 1. Naive GEMM
    with open('ptx/gemm_naive.ptx', 'r') as f:
        mod_naive = CUDAModule(driver, f.read())
    k_naive = mod_naive.get_function("gemm_naive")

    block_dim = (16, 16, 1)
    grid_dim = ((N + 15) // 16, (M + 15) // 16, 1)
    c_M, c_N, c_K = c_uint(M), c_uint(N), c_uint(K)

    args = (c_void_p * 6)(
        ctypes.cast(byref(buf_A.device_ptr), c_void_p),
        ctypes.cast(byref(buf_B.device_ptr), c_void_p),
        ctypes.cast(byref(buf_C.device_ptr), c_void_p),
        ctypes.cast(byref(c_M), c_void_p),
        ctypes.cast(byref(c_N), c_void_p),
        ctypes.cast(byref(c_K), c_void_p)
    )

    ev_start = CUDAEvent(driver)
    ev_stop = CUDAEvent(driver)

    # Warmup
    for _ in range(warmup):
        driver.lib.cuLaunchKernel(
            k_naive, grid_dim[0], grid_dim[1], grid_dim[2],
            block_dim[0], block_dim[1], block_dim[2],
            0, None, args, None
        )
    ctx.synchronize()

    naive_times = []
    for _ in range(iterations):
        ev_start.record()
        driver.lib.cuLaunchKernel(
            k_naive, grid_dim[0], grid_dim[1], grid_dim[2],
            block_dim[0], block_dim[1], block_dim[2],
            0, None, args, None
        )
        ev_stop.record()
        naive_times.append(CUDAEvent.elapsed_time_ms(ev_start, ev_stop))

    med_naive_ms = float(np.median(naive_times))
    gflops_naive = (flops / (med_naive_ms * 1e-3)) / 1e9

    # 2. Tiled Shared-Memory GEMM
    with open('ptx/gemm_tiled_shared.ptx', 'r') as f:
        mod_tiled = CUDAModule(driver, f.read())
    k_tiled = mod_tiled.get_function("gemm_tiled_shared")

    # Warmup
    for _ in range(warmup):
        driver.lib.cuLaunchKernel(
            k_tiled, grid_dim[0], grid_dim[1], grid_dim[2],
            block_dim[0], block_dim[1], block_dim[2],
            0, None, args, None
        )
    ctx.synchronize()

    tiled_times = []
    for _ in range(iterations):
        ev_start.record()
        driver.lib.cuLaunchKernel(
            k_tiled, grid_dim[0], grid_dim[1], grid_dim[2],
            block_dim[0], block_dim[1], block_dim[2],
            0, None, args, None
        )
        ev_stop.record()
        tiled_times.append(CUDAEvent.elapsed_time_ms(ev_start, ev_stop))

    med_tiled_ms = float(np.median(tiled_times))
    gflops_tiled = (flops / (med_tiled_ms * 1e-3)) / 1e9

    # 3. PyTorch cuBLAS Baseline
    t_A = torch.from_numpy(h_A).cuda()
    t_B = torch.from_numpy(h_B).cuda()
    for _ in range(warmup):
        _ = torch.matmul(t_A, t_B)
    torch.cuda.synchronize()

    pt_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = torch.matmul(t_A, t_B)
        torch.cuda.synchronize()
        pt_times.append((time.perf_counter() - t0) * 1000.0)

    med_pt_ms = float(np.median(pt_times))
    gflops_pt = (flops / (med_pt_ms * 1e-3)) / 1e9

    speedup_shared = med_naive_ms / med_tiled_ms

    print(f"1. Naive Global Memory GEMM   : {med_naive_ms:8.3f} ms | Throughput: {gflops_naive:8.2f} GFLOPS")
    print(f"2. Tiled Shared-Memory GEMM   : {med_tiled_ms:8.3f} ms | Throughput: {gflops_tiled:8.2f} GFLOPS")
    print(f"3. NVIDIA cuBLAS (PyTorch)    : {med_pt_ms:8.3f} ms | Throughput: {gflops_pt:8.2f} GFLOPS")
    print(f"  -> Shared Memory Speedup    : {speedup_shared:8.2f}x faster than Naive Global GEMM")
    print(f"  -> Microarchitectural Takeaway: Tiling matrices into 16x16 shared memory tiles reduces DRAM")
    print(f"     traffic by 16x by reusing data across threads via low-latency on-chip SRAM.")

    buf_A.free()
    buf_B.free()
    buf_C.free()
    mod_naive.unload()
    mod_tiled.unload()


def benchmark_memory_coalescing(driver, ctx, num_elements=16777216): # 16M floats = 64 MB
    print(f"\n=================================================================")
    print(f"  EXPERIMENT 3: Memory Coalescing & Vectorized 128-bit Loads     ")
    print(f"=================================================================")

    total_bytes = num_elements * 4
    # Read + Write traffic = 2 * total_bytes
    transfer_bytes = 2.0 * total_bytes

    h_in = np.ones(num_elements, dtype=np.float32)
    buf_in = CUDABuffer(driver, total_bytes)
    buf_out = CUDABuffer(driver, total_bytes)
    buf_in.copy_from_host(h_in)

    with open('ptx/memory_coalescing.ptx', 'r') as f:
        mod = CUDAModule(driver, f.read())

    k_coalesced = mod.get_function("copy_coalesced")
    k_strided = mod.get_function("copy_strided")
    k_vec = mod.get_function("copy_vectorized_v4")

    ev_start = CUDAEvent(driver)
    ev_stop = CUDAEvent(driver)

    # 1. Coalesced
    c_n = c_uint(num_elements)
    args_coal = (c_void_p * 3)(
        ctypes.cast(byref(buf_in.device_ptr), c_void_p),
        ctypes.cast(byref(buf_out.device_ptr), c_void_p),
        ctypes.cast(byref(c_n), c_void_p)
    )

    block_size = 256
    grid_size = (num_elements + block_size - 1) // block_size

    for _ in range(3):
        driver.lib.cuLaunchKernel(k_coalesced, grid_size, 1, 1, block_size, 1, 1, 0, None, args_coal, None)
    ctx.synchronize()

    ev_start.record()
    for _ in range(10):
        driver.lib.cuLaunchKernel(k_coalesced, grid_size, 1, 1, block_size, 1, 1, 0, None, args_coal, None)
    ev_stop.record()
    time_coal_ms = CUDAEvent.elapsed_time_ms(ev_start, ev_stop) / 10.0
    bw_coal_gb = (transfer_bytes / (time_coal_ms * 1e-3)) / 1e9

    # 2. Strided (stride = 32, complete warp divergence / serialization)
    c_stride = c_uint(32)
    args_strided = (c_void_p * 4)(
        ctypes.cast(byref(buf_in.device_ptr), c_void_p),
        ctypes.cast(byref(buf_out.device_ptr), c_void_p),
        ctypes.cast(byref(c_n), c_void_p),
        ctypes.cast(byref(c_stride), c_void_p)
    )

    for _ in range(3):
        driver.lib.cuLaunchKernel(k_strided, grid_size, 1, 1, block_size, 1, 1, 0, None, args_strided, None)
    ctx.synchronize()

    ev_start.record()
    for _ in range(10):
        driver.lib.cuLaunchKernel(k_strided, grid_size, 1, 1, block_size, 1, 1, 0, None, args_strided, None)
    ev_stop.record()
    time_strided_ms = CUDAEvent.elapsed_time_ms(ev_start, ev_stop) / 10.0
    bw_strided_gb = (transfer_bytes / (time_strided_ms * 1e-3)) / 1e9

    # 3. Vectorized v4 (128-bit bus transfers)
    num_vecs = num_elements // 4
    c_n_vec = c_uint(num_vecs)
    args_vec = (c_void_p * 3)(
        ctypes.cast(byref(buf_in.device_ptr), c_void_p),
        ctypes.cast(byref(buf_out.device_ptr), c_void_p),
        ctypes.cast(byref(c_n_vec), c_void_p)
    )
    grid_vec = (num_vecs + block_size - 1) // block_size

    for _ in range(3):
        driver.lib.cuLaunchKernel(k_vec, grid_vec, 1, 1, block_size, 1, 1, 0, None, args_vec, None)
    ctx.synchronize()

    ev_start.record()
    for _ in range(10):
        driver.lib.cuLaunchKernel(k_vec, grid_vec, 1, 1, block_size, 1, 1, 0, None, args_vec, None)
    ev_stop.record()
    time_vec_ms = CUDAEvent.elapsed_time_ms(ev_start, ev_stop) / 10.0
    bw_vec_gb = (transfer_bytes / (time_vec_ms * 1e-3)) / 1e9

    speedup_coalesce = time_strided_ms / time_coal_ms
    speedup_vec = time_coal_ms / time_vec_ms

    print(f"1. Coalesced 32-bit Access    : {time_coal_ms:8.3f} ms | Bandwidth: {bw_coal_gb:8.2f} GB/s")
    print(f"2. Strided Stride-32 Access   : {time_strided_ms:8.3f} ms | Bandwidth: {bw_strided_gb:8.2f} GB/s")
    print(f"3. Vectorized 128-bit (v4.f32): {time_vec_ms:8.3f} ms | Bandwidth: {bw_vec_gb:8.2f} GB/s")
    print(f"  -> Coalescing Speedup       : {speedup_coalesce:8.2f}x faster than Strided-32")
    print(f"  -> Vectorized 128-bit Boost : {speedup_vec:8.2f}x throughput increase over 32-bit loads")
    print(f"  -> Microarchitectural Takeaway: Coalesced memory transactions combine 32 warp memory requests")
    print(f"     into a single 128-byte cache line request. ld.global.v4.f32 maximizes bus transaction efficiency.")

    buf_in.free()
    buf_out.free()
    mod.unload()


def benchmark_warp_divergence(driver, ctx, num_elements=2097152, iters=500):
    print(f"\n=================================================================")
    print(f"  EXPERIMENT 4: Warp Divergence & SIMT Execution Serialization   ")
    print(f"=================================================================")

    h_data = np.ones(num_elements, dtype=np.float32)
    buf_data = CUDABuffer(driver, h_data.nbytes)
    buf_data.copy_from_host(h_data)

    with open('ptx/warp_divergence.ptx', 'r') as f:
        mod = CUDAModule(driver, f.read())

    k_divergent = mod.get_function("warp_divergent")
    k_uniform = mod.get_function("warp_uniform")

    block_size = 256
    grid_size = (num_elements + block_size - 1) // block_size
    c_n = c_uint(num_elements)
    c_iters = c_uint(iters)

    args = (c_void_p * 3)(
        ctypes.cast(byref(buf_data.device_ptr), c_void_p),
        ctypes.cast(byref(c_n), c_void_p),
        ctypes.cast(byref(c_iters), c_void_p)
    )

    ev_start = CUDAEvent(driver)
    ev_stop = CUDAEvent(driver)

    # 1. Warp Uniform (All threads in each 32-thread warp take the same branch)
    for _ in range(3):
        driver.lib.cuLaunchKernel(k_uniform, grid_size, 1, 1, block_size, 1, 1, 0, None, args, None)
    ctx.synchronize()

    ev_start.record()
    for _ in range(10):
        driver.lib.cuLaunchKernel(k_uniform, grid_size, 1, 1, block_size, 1, 1, 0, None, args, None)
    ev_stop.record()
    time_uniform_ms = CUDAEvent.elapsed_time_ms(ev_start, ev_stop) / 10.0

    # 2. Warp Divergent (Even and Odd threads in each 32-thread warp take opposite branches)
    for _ in range(3):
        driver.lib.cuLaunchKernel(k_divergent, grid_size, 1, 1, block_size, 1, 1, 0, None, args, None)
    ctx.synchronize()

    ev_start.record()
    for _ in range(10):
        driver.lib.cuLaunchKernel(k_divergent, grid_size, 1, 1, block_size, 1, 1, 0, None, args, None)
    ev_stop.record()
    time_divergent_ms = CUDAEvent.elapsed_time_ms(ev_start, ev_stop) / 10.0

    divergence_penalty = time_divergent_ms / time_uniform_ms

    print(f"1. Uniform Warp Branching     : {time_uniform_ms:8.3f} ms")
    print(f"2. Divergent Warp Branching   : {time_divergent_ms:8.3f} ms")
    print(f"  -> Warp Divergence Penalty  : {divergence_penalty:8.2f}x slower due to SIMT serialization")
    print(f"  -> Microarchitectural Takeaway: When threads within a warp diverge, the warp scheduler")
    print(f"     serializes execution of each divergent path while disabling non-active threads via")
    print(f"     hardware active masks, effectively halving instruction throughput.")

    buf_data.free()
    mod.unload()


def main():
    print("=================================================================")
    print("  NVIDIA CUDA & PTX KERNEL PERFORMANCE LABORATORY (RTX 3050)     ")
    print("=================================================================")

    driver = CUDADriver()
    dev = CUDADevice(driver, 0)
    ctx = CUDAContext(driver, dev)

    print(f"Device Name           : {dev.name}")
    print(f"Streaming Multiprocs  : {dev.sm_count} SMs")
    print(f"Warp Size             : {dev.warp_size} threads")
    print(f"Max Threads per Block : {dev.max_threads_per_block}")

    benchmark_gemm(driver, ctx, size=512)
    benchmark_memory_coalescing(driver, ctx)
    benchmark_warp_divergence(driver, ctx)

    ctx.destroy()
    print("\n=================================================================")
    print("  All GPU Microarchitecture Benchmarks Completed Successfully!   ")
    print("=================================================================")

if __name__ == "__main__":
    main()
