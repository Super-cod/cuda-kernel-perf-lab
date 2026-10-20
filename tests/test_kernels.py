"""
Automated Correctness Verification Suite for CUDA PTX Kernels
Validates mathematical results against NumPy reference implementations on the physical GPU.
"""

import os
import sys
import ctypes
from ctypes import c_int, c_uint, c_ulonglong, c_void_p, byref
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from cuda_driver import CUDADriver, CUDADevice, CUDAContext, CUDAModule, CUDABuffer

def test_gemm_correctness():
    driver = CUDADriver()
    dev = CUDADevice(driver, 0)
    ctx = CUDAContext(driver, dev)

    M, N, K = 128, 128, 128
    np.random.seed(42)
    h_A = np.random.randn(M, K).astype(np.float32)
    h_B = np.random.randn(K, N).astype(np.float32)
    h_C_ref = np.matmul(h_A, h_B)

    buf_A = CUDABuffer(driver, h_A.nbytes)
    buf_B = CUDABuffer(driver, h_B.nbytes)
    buf_C_naive = CUDABuffer(driver, h_C_ref.nbytes)
    buf_C_tiled = CUDABuffer(driver, h_C_ref.nbytes)

    buf_A.copy_from_host(h_A)
    buf_B.copy_from_host(h_B)

    # 1. Test Naive GEMM
    with open('ptx/gemm_naive.ptx', 'r') as f:
        mod_naive = CUDAModule(driver, f.read())
    k_naive = mod_naive.get_function("gemm_naive")

    block_dim = (16, 16, 1)
    grid_dim = ((N + 15) // 16, (M + 15) // 16, 1)
    c_M, c_N, c_K = c_uint(M), c_uint(N), c_uint(K)

    args_naive = (c_void_p * 6)(
        ctypes.cast(byref(buf_A.device_ptr), c_void_p),
        ctypes.cast(byref(buf_B.device_ptr), c_void_p),
        ctypes.cast(byref(buf_C_naive.device_ptr), c_void_p),
        ctypes.cast(byref(c_M), c_void_p),
        ctypes.cast(byref(c_N), c_void_p),
        ctypes.cast(byref(c_K), c_void_p)
    )

    driver.lib.cuLaunchKernel(
        k_naive,
        grid_dim[0], grid_dim[1], grid_dim[2],
        block_dim[0], block_dim[1], block_dim[2],
        0, None, args_naive, None
    )
    ctx.synchronize()

    h_out_naive = np.zeros_like(h_C_ref)
    buf_C_naive.copy_to_host(h_out_naive)
    assert np.allclose(h_out_naive, h_C_ref, atol=1e-3, rtol=1e-3), "Naive GEMM output mismatch!"
    print("[PASS] Test 1: gemm_naive matches NumPy matmul reference (allclose atol=1e-3)")

    # 2. Test Tiled Shared-Memory GEMM
    with open('ptx/gemm_tiled_shared.ptx', 'r') as f:
        mod_tiled = CUDAModule(driver, f.read())
    k_tiled = mod_tiled.get_function("gemm_tiled_shared")

    args_tiled = (c_void_p * 6)(
        ctypes.cast(byref(buf_A.device_ptr), c_void_p),
        ctypes.cast(byref(buf_B.device_ptr), c_void_p),
        ctypes.cast(byref(buf_C_tiled.device_ptr), c_void_p),
        ctypes.cast(byref(c_M), c_void_p),
        ctypes.cast(byref(c_N), c_void_p),
        ctypes.cast(byref(c_K), c_void_p)
    )

    driver.lib.cuLaunchKernel(
        k_tiled,
        grid_dim[0], grid_dim[1], grid_dim[2],
        block_dim[0], block_dim[1], block_dim[2],
        0, None, args_tiled, None
    )
    ctx.synchronize()

    h_out_tiled = np.zeros_like(h_C_ref)
    buf_C_tiled.copy_to_host(h_out_tiled)
    assert np.allclose(h_out_tiled, h_C_ref, atol=1e-3, rtol=1e-3), "Tiled GEMM output mismatch!"
    print("[PASS] Test 2: gemm_tiled_shared matches NumPy matmul reference (allclose atol=1e-3)")

    # 3. Test Vectorized v4 Copy
    with open('ptx/memory_coalescing.ptx', 'r') as f:
        mod_mem = CUDAModule(driver, f.read())
    k_vec = mod_mem.get_function("copy_vectorized_v4")

    NUM_FLOATS = 1024
    h_vec_in = np.arange(NUM_FLOATS, dtype=np.float32)
    h_vec_out = np.zeros(NUM_FLOATS, dtype=np.float32)

    buf_in = CUDABuffer(driver, h_vec_in.nbytes)
    buf_out = CUDABuffer(driver, h_vec_out.nbytes)
    buf_in.copy_from_host(h_vec_in)

    NUM_VECS = NUM_FLOATS // 4
    c_n_vec = c_uint(NUM_VECS)
    args_vec = (c_void_p * 3)(
        ctypes.cast(byref(buf_in.device_ptr), c_void_p),
        ctypes.cast(byref(buf_out.device_ptr), c_void_p),
        ctypes.cast(byref(c_n_vec), c_void_p)
    )

    driver.lib.cuLaunchKernel(
        k_vec,
        (NUM_VECS + 255) // 256, 1, 1,
        256, 1, 1,
        0, None, args_vec, None
    )
    ctx.synchronize()

    buf_out.copy_to_host(h_vec_out)
    assert np.array_equal(h_vec_out, h_vec_in), "Vectorized copy output mismatch!"
    print("[PASS] Test 3: copy_vectorized_v4 exact bit-level array reproduction")

    # Cleanup
    buf_A.free()
    buf_B.free()
    buf_C_naive.free()
    buf_C_tiled.free()
    buf_in.free()
    buf_out.free()
    mod_naive.unload()
    mod_tiled.unload()
    mod_mem.unload()
    ctx.destroy()

if __name__ == "__main__":
    print(">>> Running CUDA PTX Kernel Correctness Suite on RTX 3050 <<<")
    test_gemm_correctness()
    print("\n>>> ALL CUDA PTX CORRECTNESS TESTS PASSED <<<")
