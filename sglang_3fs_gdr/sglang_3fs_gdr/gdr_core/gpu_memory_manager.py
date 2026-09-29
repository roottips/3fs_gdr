"""
GPU 内存管理与 RDMA 注册模块

负责分配可用于 GPUDirect RDMA 的 GPU 设备内存，
并通过 ibv_reg_mr 注册为 RDMA 可访问的内存区域。

核心概念:
  - GDRBuffer:  封装 GPU 显存指针 + RDMA MR (Memory Region) 句柄
  - GPUMemoryManager: 管理预分配的 GPU 内存池，避免频繁分配/释放

数据流:
  1. cudaMalloc 分配 GPU 设备内存
  2. 获取 GPU 内存的物理地址 (通过 cuMemGetAddressRange)
  3. ibv_reg_mr 注册 GPU 内存区域到 RDMA Protection Domain
  4. ibv_post_send/recv 使用 GPU 内存进行零拷贝传输
"""

from __future__ import annotations

import ctypes
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import IntEnum

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------- #
#  ctypes 结构体与函数签名
# ----------------------------------------------------------------------- #

# ibverbs 结构体（简化版，仅包含必要字段）
class ibv_pd(ctypes.Structure):
    pass

class ibv_context(ctypes.Structure):
    pass

class ibv_mr(ctypes.Structure):
    pass

class ibv_device(ctypes.Structure):
    pass

# CUDA Driver API 类型
cu_device = ctypes.c_int
cu_context = ctypes.c_void_p


@dataclass
class GDRBuffer:
    """
    GDR 缓冲区: GPU 显存 + RDMA 内存注册

    属性:
        gpu_ptr:       GPU 设备内存指针 (ctypes.c_void_p)
        size:          缓冲区大小 (字节)
        gpu_addr:      GPU 物理地址 (用于 RDMA)
        mr_handle:     ibv_mr 句柄 (RDMA Memory Region)
        mr_lkey:       本地访问 Key
        mr_rkey:       远程访问 Key
        stream:        CUDA Stream (可选)
    """
    gpu_ptr: int
    size: int
    gpu_addr: int
    mr_handle: int           # ibv_mr* 指针值
    mr_lkey: int
    mr_rkey: int
    stream: int = 0
    in_use: bool = False
    ref_count: int = 0

    def __repr__(self) -> str:
        return (
            f"GDRBuffer(ptr=0x{self.gpu_ptr:x}, size={self.size}, "
            f"lkey={self.mr_lkey}, rkey={self.mr_rkey})"
        )


class MemoryPoolStrategy(IntEnum):
    """内存池分配策略"""
    STATIC = 1       # 预分配固定大小池
    DYNAMIC = 2      # 动态扩展
    BUDDY = 3        # 伙伴系统


class GPUMemoryManager:
    """
    GPU 内存管理器

    管理 GPU 设备内存的分配、RDMA 注册和回收。
    支持预分配内存池以减少运行时分配开销。

    用法:
        mgr = GPUMemoryManager(gpu_id=0, ib_dev="mlx5_0")
        buf = mgr.alloc_buffer(1024 * 1024 * 1024)  # 1GB
        # ... 使用 buf 进行 GDR 传输 ...
        mgr.free_buffer(buf)
    """

    # IBV 访问权限标志
    IBV_ACCESS_LOCAL_WRITE = 1
    IBV_ACCESS_REMOTE_READ = 2
    IBV_ACCESS_REMOTE_WRITE = 4
    IBV_ACCESS_REMOTE_ATOMIC = 8
    IBV_ACCESS_RELAXED_ORDERING = 1 << 14

    # GPU 内存访问权限
    CU_MEM_ATTRIBUTE_GPU_DIRECT_RDMA = 1  # 使 GPU 内存对 RDMA 可见

    def __init__(
        self,
        gpu_id: int = 0,
        ib_dev: str = "mlx5_0",
        pool_size_gb: int = 4,
        strategy: MemoryPoolStrategy = MemoryPoolStrategy.STATIC,
    ):
        self.gpu_id = gpu_id
        self.ib_dev = ib_dev
        self.pool_size = pool_size_gb * 1024 * 1024 * 1024
        self.strategy = strategy

        self._pool: list[GDRBuffer] = []
        self._free_list: list[GDRBuffer] = []
        self._lock = threading.Lock()

        self._lib_ibverbs: Optional[ctypes.CDLL] = None
        self._lib_cuda: Optional[ctypes.CDLL] = None
        self._ib_context: Optional[ibv_context] = None
        self._pd: Optional[ibv_pd] = None
        self._initialized = False

    # ------------------------------------------------------------------ #
    #  初始化
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        """
        初始化 GPU 与 RDMA 环境连接。

        步骤:
          1. 加载 libibverbs 和 CUDA Driver 库
          2. cuInit() 初始化 CUDA
          3. 打开 IB 设备上下文
          4. 分配 Protection Domain
          5. 预分配 GPU 内存池
        """
        try:
            self._load_libs()
            self._init_cuda()
            self._init_ib_context()
            self._prealloc_pool()
            self._initialized = True
            logger.info(
                f"GPUMemoryManager 初始化成功: GPU={self.gpu_id}, "
                f"IB={self.ib_dev}, pool={len(self._pool)} buffers"
            )
            return True
        except Exception as e:
            logger.error(f"GPUMemoryManager 初始化失败: {e}")
            return False

    def _load_libs(self):
        """加载动态链接库"""
        try:
            self._lib_ibverbs = ctypes.CDLL("libibverbs.so.1")
        except OSError as e:
            raise RuntimeError(f"无法加载 libibverbs: {e}")

        try:
            self._lib_cuda = ctypes.CDLL("libcuda.so")
        except OSError as e:
            raise RuntimeError(f"无法加载 CUDA Driver: {e}")

        # 设置函数签名
        self._setup_ibv_signatures()
        self._setup_cuda_signatures()

    def _setup_ibv_signatures(self):
        lib = self._lib_ibverbs

        # ibv_open_device
        lib.ibv_open_device.restype = ctypes.POINTER(ibv_context)
        lib.ibv_open_device.argtypes = [ctypes.POINTER(ibv_device)]

        # ibv_alloc_pd
        lib.ibv_alloc_pd.restype = ctypes.POINTER(ibv_pd)
        lib.ibv_alloc_pd.argtypes = [ctypes.POINTER(ibv_context)]

        # ibv_reg_mr — 关键: 注册 GPU 内存
        lib.ibv_reg_mr.restype = ctypes.POINTER(ibv_mr)
        lib.ibv_reg_mr.argtypes = [
            ctypes.POINTER(ibv_pd),
            ctypes.c_void_p,   # buf (GPU 设备内存指针)
            ctypes.c_size_t,    # length
            ctypes.c_int,        # access flags
        ]

        # ibv_dereg_mr
        lib.ibv_dereg_mr.restype = ctypes.c_int
        lib.ibv_dereg_mr.argtypes = [ctypes.POINTER(ibv_mr)]

        # ibv_get_device_list
        lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.POINTER(ibv_device))
        lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]

        # ibv_close_device
        lib.ibv_close_device.restype = ctypes.c_int
        lib.ibv_close_device.argtypes = [ctypes.POINTER(ibv_context)]

    def _setup_cuda_signatures(self):
        lib = self._lib_cuda

        lib.cuInit.restype = ctypes.c_int
        lib.cuInit.argtypes = [ctypes.c_uint]

        lib.cuDeviceGet.restype = ctypes.c_int
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]

        # cuMemAlloc_v2 — 分配 GPU 设备内存 (用于 GDR)
        lib.cuMemAlloc_v2.restype = ctypes.c_int
        lib.cuMemAlloc_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),  # dptr
            ctypes.c_size_t,                   # bytesize
        ]

        # cuMemFree_v2
        lib.cuMemFree_v2.restype = ctypes.c_int
        lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]

        # cuMemGetAddressRange — 获取 GPU 内存的物理地址范围
        lib.cuMemGetAddressRange.restype = ctypes.c_int
        lib.cuMemGetAddressRange.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),   # pbase
            ctypes.POINTER(ctypes.c_size_t),   # psize
            ctypes.c_uint64,                    # dptr
        ]

        # cuCtxSetCurrent
        lib.cuCtxSetCurrent.restype = ctypes.c_int
        lib.cuCtxSetCurrent.argtypes = [cu_context]

    def _init_cuda(self):
        """初始化 CUDA Driver 并获取设备上下文"""
        lib = self._lib_cuda
        ret = lib.cuInit(0)
        if ret != 0:
            raise RuntimeError(f"cuInit 失败: 错误码 {ret}")

        dev = ctypes.c_int(0)
        ret = lib.cuDeviceGet(ctypes.byref(dev), self.gpu_id)
        if ret != 0:
            raise RuntimeError(f"cuDeviceGet({self.gpu_id}) 失败: {ret}")

        self._cuda_device = dev.value
        logger.info(f"CUDA 初始化成功: device {self.gpu_id}")

    def _init_ib_context(self):
        """打开 IB 设备并分配 Protection Domain"""
        lib = self._lib_ibverbs

        # 获取设备列表
        num_devices = ctypes.c_int(0)
        dev_list_ptr = lib.ibv_get_device_list(ctypes.byref(num_devices))
        if num_devices.value == 0:
            raise RuntimeError("没有找到 IB 设备")

        # 查找目标设备
        target_dev = None
        for i in range(num_devices.value):
            dev = dev_list_ptr[i]
            if dev:
                target_dev = dev
                break

        if target_dev is None:
            raise RuntimeError(f"未找到 IB 设备 {self.ib_dev}")

        # 打开设备上下文
        self._ib_context = lib.ibv_open_device(target_dev)
        if not self._ib_context:
            raise RuntimeError(f"无法打开 IB 设备 {self.ib_dev}")

        # 分配 PD
        self._pd = lib.ibv_alloc_pd(self._ib_context)
        if not self._pd:
            raise RuntimeError("ibv_alloc_pd 失败")

        logger.info(f"IB 上下文初始化成功: {self.ib_dev}")

    def _prealloc_pool(self):
        """
        预分配 GPU 内存池。

        使用伙伴式分配: 将 pool_size 分为多个固定大小块。
        """
        chunk_size = 256 * 1024 * 1024  # 256MB per chunk
        num_chunks = max(1, self.pool_size // chunk_size)

        for _ in range(num_chunks):
            buf = self._alloc_and_register(chunk_size)
            if buf:
                self._pool.append(buf)
                self._free_list.append(buf)
            else:
                logger.warning(f"预分配 chunk 失败 ({chunk_size} bytes)")

    # ------------------------------------------------------------------ #
    #  内存分配与注册
    # ------------------------------------------------------------------ #

    def _alloc_and_register(self, size: int) -> Optional[GDRBuffer]:
        """
        核心: 分配 GPU 内存并注册到 RDMA PD。

        步骤:
          1. cuMemAlloc_v2 分配 GPU 设备内存
          2. cuMemGetAddressRange 获取物理地址
          3. ibv_reg_mr 注册为 RDMA 内存区域
          4. 返回 GDRBuffer 封装
        """
        lib_cuda = self._lib_cuda
        lib_ibv = self._lib_ibverbs

        # 1. 分配 GPU 设备内存
        dptr = ctypes.c_uint64(0)
        ret = lib_cuda.cuMemAlloc_v2(ctypes.byref(dptr), size)
        if ret != 0:
            logger.error(f"cuMemAlloc_v2 失败: 错误码 {ret}, size={size}")
            return None

        gpu_ptr = dptr.value

        # 2. 获取 GPU 物理地址范围 (用于 RDMA 定位)
        phys_base = ctypes.c_uint64(0)
        phys_size = ctypes.c_size_t(0)
        ret = lib_cuda.cuMemGetAddressRange(
            ctypes.byref(phys_base),
            ctypes.byref(phys_size),
            gpu_ptr,
        )
        if ret != 0:
            logger.warning(f"cuMemGetAddressRange 失败: {ret}, 使用虚拟地址")
            phys_base.value = gpu_ptr
        gpu_addr = phys_base.value

        # 3. 注册 RDMA Memory Region
        # 关键: buf 参数为 GPU 设备内存指针
        # peermem 模块会处理 GPU<->NIC 的地址映射
        access_flags = (
            self.IBV_ACCESS_LOCAL_WRITE
            | self.IBV_ACCESS_REMOTE_READ
            | self.IBV_ACCESS_REMOTE_WRITE
            | self.IBV_ACCESS_RELAXED_ORDERING
        )

        mr = lib_ibv.ibv_reg_mr(
            self._pd,
            ctypes.c_void_p(gpu_ptr),
            size,
            access_flags,
        )

        if not mr:
            logger.error(
                f"ibv_reg_mr 失败: GPU ptr=0x{gpu_ptr:x}, size={size}"
            )
            # 清理已分配的 GPU 内存
            lib_cuda.cuMemFree_v2(gpu_ptr)
            return None

        # 4. 提取 MR 的 lkey 和 rkey
        # ibv_mr 结构体前几个字段: addr, length, lkey, rkey
        # 内存布局: void* addr (8), size_t length (8), uint32_t lkey (4), uint32_t rkey (4)
        mr_ptr = ctypes.cast(mr, ctypes.POINTER(ctypes.c_uint8))
        lkey = int.from_bytes(
            bytearray(mr_ptr[16:20]), byteorder="little"
        )
        rkey = int.from_bytes(
            bytearray(mr_ptr[20:24]), byteorder="little"
        )

        return GDRBuffer(
            gpu_ptr=gpu_ptr,
            size=size,
            gpu_addr=gpu_addr,
            mr_handle=ctypes.addressof(mr.contents),
            mr_lkey=lkey,
            mr_rkey=rkey,
        )

    def alloc_buffer(self, size: int) -> Optional[GDRBuffer]:
        """
        分配指定大小的 GDR 缓冲区。

        优先从内存池获取，不足时动态分配。

        Args:
            size: 请求的字节数

        Returns:
            GDRBuffer 或 None (分配失败)
        """
        if not self._initialized and not self.initialize():
            return None

        with self._lock:
            # 从空闲列表查找合适的块
            best_idx = -1
            best_size = float("inf")
            for i, buf in enumerate(self._free_list):
                if buf.size >= size and buf.size < best_size:
                    best_idx = i
                    best_size = buf.size

            if best_idx >= 0:
                buf = self._free_list.pop(best_idx)
                buf.in_use = True
                buf.ref_count = 1
                return buf

        # 动态分配
        buf = self._alloc_and_register(size)
        if buf:
            with self._lock:
                self._pool.append(buf)
                buf.in_use = True
                buf.ref_count = 1
        return buf

    def free_buffer(self, buf: GDRBuffer):
        """
        释放 GDR 缓冲区回内存池。

        不实际释放 GPU 内存，而是归还到空闲列表供复用。
        """
        with self._lock:
            buf.in_use = False
            buf.ref_count = 0
            if buf not in self._free_list:
                self._free_list.append(buf)

    def get_buffer_info(self, buf: GDRBuffer) -> dict:
        """获取缓冲区的远程访问信息 (用于 RDMA Write/Read)"""
        return {
            "gpu_addr": buf.gpu_addr,
            "size": buf.size,
            "rkey": buf.mr_rkey,
            "lkey": buf.mr_lkey,
        }

    # ------------------------------------------------------------------ #
    #  清理
    # ------------------------------------------------------------------ #

    def cleanup(self):
        """释放所有 GPU 内存和 RDMA MR"""
        lib_cuda = self._lib_cuda
        lib_ibv = self._lib_ibverbs

        with self._lock:
            for buf in self._pool:
                # 注销 MR
                mr_ptr = ctypes.cast(
                    buf.mr_handle, ctypes.POINTER(ibv_mr)
                )
                lib_ibv.ibv_dereg_mr(mr_ptr)
                # 释放 GPU 内存
                lib_cuda.cuMemFree_v2(buf.gpu_ptr)

            self._pool.clear()
            self._free_list.clear()

        # 关闭 PD 和 IB 上下文
        if self._pd:
            # ibv_dealloc_pd 通过 close 隐式处理
            pass

        self._initialized = False
        logger.info("GPUMemoryManager 清理完成")

    def __del__(self):
        if self._initialized:
            self.cleanup()
