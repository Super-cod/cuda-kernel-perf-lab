"""
NVIDIA CUDA Driver API Low-Level ctypes Interface
Provides direct interaction with nvcuda.dll for loading, JIT-compiling,
and executing PTX virtual ISA kernels on NVIDIA GPUs.
"""

import ctypes
from ctypes import (
    c_int, c_uint, c_void_p, c_size_t, c_ulonglong,
    c_float, c_char_p, POINTER, byref, cast
)
import numpy as np

class CUDADriverError(Exception):
    def __init__(self, code, message=""):
        super().__init__(f"CUDA Driver API Error {code}: {message}")
        self.code = code

class CUDADriver:
    _instance = None

    def __init__(self):
        try:
            self.lib = ctypes.WinDLL("nvcuda.dll")
        except Exception as e:
            raise RuntimeError(f"Could not load nvcuda.dll: {e}")

        self._bind_functions()
        res = self.lib.cuInit(0)
        if res != 0:
            raise CUDADriverError(res, "cuInit failed")

    def _bind_functions(self):
        # Memory functions
        self.lib.cuMemAlloc_v2.argtypes = [POINTER(c_ulonglong), c_size_t]
        self.lib.cuMemAlloc_v2.restype = c_int

        self.lib.cuMemFree_v2.argtypes = [c_ulonglong]
        self.lib.cuMemFree_v2.restype = c_int

        self.lib.cuMemcpyHtoD_v2.argtypes = [c_ulonglong, c_void_p, c_size_t]
        self.lib.cuMemcpyHtoD_v2.restype = c_int

        self.lib.cuMemcpyDtoH_v2.argtypes = [c_void_p, c_ulonglong, c_size_t]
        self.lib.cuMemcpyDtoH_v2.restype = c_int

        # Event timing functions
        self.lib.cuEventCreate.argtypes = [POINTER(c_void_p), c_uint]
        self.lib.cuEventCreate.restype = c_int

        self.lib.cuEventDestroy_v2.argtypes = [c_void_p]
        self.lib.cuEventDestroy_v2.restype = c_int

        self.lib.cuEventRecord.argtypes = [c_void_p, c_void_p]
        self.lib.cuEventRecord.restype = c_int

        self.lib.cuEventSynchronize.argtypes = [c_void_p]
        self.lib.cuEventSynchronize.restype = c_int

        self.lib.cuEventElapsedTime.argtypes = [POINTER(c_float), c_void_p, c_void_p]
        self.lib.cuEventElapsedTime.restype = c_int

        # Kernel launch
        self.lib.cuLaunchKernel.argtypes = [
            c_void_p, # f
            c_uint, c_uint, c_uint, # gridDim
            c_uint, c_uint, c_uint, # blockDim
            c_uint, # sharedMemBytes
            c_void_p, # hStream
            POINTER(c_void_p), # kernelParams
            POINTER(c_void_p)  # extra
        ]
        self.lib.cuLaunchKernel.restype = c_int

class CUDADevice:
    def __init__(self, driver: CUDADriver, ordinal: int = 0):
        self.driver = driver
        self.ordinal = ordinal
        self.handle = c_int()
        res = driver.lib.cuDeviceGet(byref(self.handle), ordinal)
        if res != 0:
            raise CUDADriverError(res, "cuDeviceGet failed")

    @property
    def name(self) -> str:
        buf = ctypes.create_string_buffer(256)
        self.driver.lib.cuDeviceGetName(buf, 256, self.handle)
        return buf.value.decode("utf-8")

    def get_attribute(self, attrib_id: int) -> int:
        val = c_int()
        res = self.driver.lib.cuDeviceGetAttribute(byref(val), attrib_id, self.handle)
        if res != 0:
            raise CUDADriverError(res, f"cuDeviceGetAttribute({attrib_id}) failed")
        return val.value

    @property
    def sm_count(self) -> int:
        return self.get_attribute(16) # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT

    @property
    def max_threads_per_block(self) -> int:
        return self.get_attribute(1) # CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK

    @property
    def warp_size(self) -> int:
        return self.get_attribute(10) # CU_DEVICE_ATTRIBUTE_WARP_SIZE

class CUDAContext:
    def __init__(self, driver: CUDADriver, device: CUDADevice):
        self.driver = driver
        self.device = device
        self.handle = c_void_p()
        res = driver.lib.cuCtxCreate_v2(byref(self.handle), 0, device.handle)
        if res != 0:
            raise CUDADriverError(res, "cuCtxCreate failed")

    def destroy(self):
        if self.handle:
            self.driver.lib.cuCtxDestroy_v2(self.handle)
            self.handle = None

    def synchronize(self):
        res = self.driver.lib.cuCtxSynchronize()
        if res != 0:
            raise CUDADriverError(res, "cuCtxSynchronize failed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

class CUDABuffer:
    def __init__(self, driver: CUDADriver, size_bytes: int):
        self.driver = driver
        self.size_bytes = size_bytes
        self.device_ptr = c_ulonglong()
        res = driver.lib.cuMemAlloc_v2(byref(self.device_ptr), c_size_t(size_bytes))
        if res != 0:
            raise CUDADriverError(res, f"cuMemAlloc({size_bytes}) failed")

    def copy_from_host(self, host_array: np.ndarray):
        assert host_array.nbytes == self.size_bytes, "Array size mismatch"
        res = self.driver.lib.cuMemcpyHtoD_v2(
            self.device_ptr,
            host_array.ctypes.data,
            c_size_t(self.size_bytes)
        )
        if res != 0:
            raise CUDADriverError(res, "cuMemcpyHtoD failed")

    def copy_to_host(self, host_array: np.ndarray):
        assert host_array.nbytes == self.size_bytes, "Array size mismatch"
        res = self.driver.lib.cuMemcpyDtoH_v2(
            host_array.ctypes.data,
            self.device_ptr,
            c_size_t(self.size_bytes)
        )
        if res != 0:
            raise CUDADriverError(res, "cuMemcpyDtoH failed")

    def free(self):
        if self.device_ptr.value != 0:
            self.driver.lib.cuMemFree_v2(self.device_ptr)
            self.device_ptr.value = 0

    def __del__(self):
        self.free()

class CUDAModule:
    def __init__(self, driver: CUDADriver, ptx_source: str):
        self.driver = driver
        self.handle = c_void_p()
        ptx_bytes = ptx_source.encode("utf-8") if isinstance(ptx_source, str) else ptx_source
        res = driver.lib.cuModuleLoadData(byref(self.handle), ptx_bytes)
        if res != 0:
            raise CUDADriverError(res, "cuModuleLoadData failed (PTX JIT compilation error)")

    def get_function(self, name: str) -> c_void_p:
        func = c_void_p()
        res = self.driver.lib.cuModuleGetFunction(byref(func), self.handle, name.encode("utf-8"))
        if res != 0:
            raise CUDADriverError(res, f"cuModuleGetFunction({name}) failed")
        return func

    def unload(self):
        if self.handle:
            self.driver.lib.cuModuleUnload(self.handle)
            self.handle = None

    def __del__(self):
        self.unload()

class CUDAEvent:
    def __init__(self, driver: CUDADriver):
        self.driver = driver
        self.handle = c_void_p()
        res = driver.lib.cuEventCreate(byref(self.handle), 0)
        if res != 0:
            raise CUDADriverError(res, "cuEventCreate failed")

    def record(self, stream=None):
        res = self.driver.lib.cuEventRecord(self.handle, stream)
        if res != 0:
            raise CUDADriverError(res, "cuEventRecord failed")

    def synchronize(self):
        res = self.driver.lib.cuEventSynchronize(self.handle)
        if res != 0:
            raise CUDADriverError(res, "cuEventSynchronize failed")

    @staticmethod
    def elapsed_time_ms(start: "CUDAEvent", stop: "CUDAEvent") -> float:
        stop.synchronize()
        ms = c_float()
        res = start.driver.lib.cuEventElapsedTime(byref(ms), start.handle, stop.handle)
        if res != 0:
            raise CUDADriverError(res, "cuEventElapsedTime failed")
        return float(ms.value)

    def destroy(self):
        if self.handle:
            self.driver.lib.cuEventDestroy_v2(self.handle)
            self.handle = None

    def __del__(self):
        self.destroy()
