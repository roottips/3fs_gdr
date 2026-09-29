"""
3FS USRBIO GDR 扩展模块

3FS 原生 USRBIO API 使用 CPU 共享内存 (Iov) 作为数据缓冲区。
本模块扩展 USRBIO 路径，增加 GPU 设备内存作为可选缓冲区，
实现 GPU↔3FS 存储节点的零拷贝 RDMA 传输。

3FS USRBIO 原始 I/O 路径:
  User Process → Iov (CPU 共享内存) → RDMA → 3FS Storage Node SSD

GDR 扩展后的 I/O 路径:
  GPU Device Memory → RDMA (GPUDirect) → 3FS Storage Node SSD
  (跳过 CPU 共享内存，减少一次 PCIe DMA)

核心改动:
  1. GDR Iov: 使用 GPU 显存替代 CPU 共享内存作为 Iov buffer
  2. GDR Ior: 保持 CPU 共享内存不变 (控制面，数据量极小)
  3. RDMA 注册: 将 GPU 内存注册到 3FS 的 RDMA Protection Domain
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
import struct
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional, Any

from ..gdr_core.gpu_memory_manager import GPUMemoryManager, GDRBuffer

logger = logging.getLogger(__name__)


class UsrbioOpType(IntEnum):
    """USRBIO 操作类型"""
    READ = 0
    WRITE = 1
    READ_MULTI = 2    # 批量读
    WRITE_MULTI = 3   # 批量写


@dataclass
class GDRRequest:
    """GDR 读写请求"""
    op_type: UsrbioOpType
    gdr_buf: GDRBuffer              # GPU 显存缓冲区
    file_offset: int                # 文件偏移 (字节)
    length: int                     # 读写长度 (字节)
    batch_id: int = 0               # 批量 ID
    status: int = 0                 # 0=pending, 1=done, -1=error
    bytes_transferred: int = 0


class UsrbioGDRExtension:
    """
    3FS USRBIO GDR 扩展

    在 3FS 原生 USRBIO API 基础上增加 GPU Direct 路径:
      - 使用 GPUMemoryManager 分配 GPU 显存
      - 将 GPU 显存注册到 3FS 的 Iov 共享内存机制
      - 3FS 存储节点通过 RDMA 直接访问 GPU 显存

    架构:
      ┌─────────────────────────────────────────────────┐
      │                  SGLang GPU Process              │
      │  ┌───────────┐    ┌──────────────┐              │
      │  │ GPU 显存  │ ←→ │ GDR Iov Mgr  │              │
      │  │ (KV Cache) │    │ (GPU MR)     │              │
      │  └───────────┘    └──────┬───────┘              │
      │                           │ GDR Ior (CPU SHM)    │
      │                    ┌──────┴───────┐              │
      │                    │ USRBIO Daemon│              │
      │                    └──────┬───────┘              │
      └───────────────────────────┼──────────────────────┘
                                  │ RDMA (GPU→NIC)
                                  ▼
      ┌─────────────────────────────────────────────────┐
      │              3FS Storage Node                    │
      │  ┌───────────┐    ┌──────────────┐              │
      │  │   SSD     │ ←→ │ RDMA Engine  │              │
      │  │ (Chunk)   │    │ (ibv_write)  │              │
      │  └───────────┘    └──────────────┘              │
      └─────────────────────────────────────────────────┘

    用法:
        ext = UsrbioGDRExtension(
            iov_path="/3fs/iov_gpu0",
            ior_path="/3fs/ior_gpu0",
            gpu_id=0,
            ib_dev="mlx5_0",
        )
        ext.initialize()
        buf = ext.alloc_gpu_iov(1024 * 1024 * 1024)  # 1GB
        req = ext.submit_read(file_offset=0, length=4096, gdr_buf=buf)
        ext.wait_completion(req)
    """

    IOV_HEADER_SIZE = 4096          # Iov 文件头部元数据
    IOR_ENTRY_SIZE = 64              # 每个 Ior 请求条目大小
    IOR_MAX_DEPTH = 256              # 最大 IO 深度
    IOR_MAGIC = 0x4846334653000000   # "HF3FS\0\0\0"

    def __init__(
        self,
        iov_path: str,
        ior_path: str,
        gpu_id: int = 0,
        ib_dev: str = "mlx5_0",
        pool_size_gb: int = 4,
    ):
        self.iov_path = iov_path
        self.ior_path = ior_path
        self.gpu_id = gpu_id
        self.ib_dev = ib_dev

        self.mem_mgr = GPUMemoryManager(gpu_id, ib_dev, pool_size_gb)

        self._iov_fd: int = -1
        self._ior_fd: int = -1
        self._iov_mmap: Optional[mmap.mmap] = None
        self._ior_mmap: Optional[mmap.mmap] = None

        self._iov_capacity: int = 0
        self._ior_depth: int = self.IOR_MAX_DEPTH

        self._pending_requests: dict[int, GDRRequest] = {}
        self._completed_requests: list[GDRRequest] = []
        self._lock = threading.Lock()
        self._batch_counter = 0

        self._initialized = False

    # ------------------------------------------------------------------ #
    #  初始化
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        """
        初始化 GDR USRBIO 扩展。

        步骤:
          1. 初始化 GPU 内存管理器
          2. 打开/创建 GDR Iov 文件 (GPU 内存映射)
          3. 打开/创建 Ior 文件 (CPU 控制环)
          4. 写入 Iov 头部 (包含 GPU MR 信息)
          5. 启动完成轮询线程
        """
        try:
            # 1. GPU 内存管理器
            if not self.mem_mgr.initialize():
                raise RuntimeError("GPU 内存管理器初始化失败")

            # 2. 创建 Iov 文件 (GPU 版本)
            self._create_iov_file()
            self._create_ior_file()

            # 3. 写入 Iov 头部
            self._write_iov_header()

            self._initialized = True
            logger.info(
                f"UsrbioGDRExtension 初始化成功: "
                f"iov={self.iov_path}, ior={self.ior_path}"
            )
            return True
        except Exception as e:
            logger.error(f"UsrbioGDRExtension 初始化失败: {e}")
            return False

    def _create_iov_file(self):
        """
        创建 GDR Iov 文件。

        原始 3FS Iov 是 CPU 共享内存文件。
        GDR 版本: 文件包含头部元数据 (GPU MR 信息)，
        数据区指向 GPU 显存 (不实际 mmap GPU 内存)。

        USRBIO daemon 读取头部获取 GPU MR 的 rkey 和 addr，
        然后 RDMA Write 到 GPU 显存。
        """
        iov_dir = os.path.dirname(self.iov_path)
        if iov_dir and not os.path.exists(iov_dir):
            os.makedirs(iov_dir, exist_ok=True)

        # 创建或打开 Iov 文件
        if os.path.exists(self.iov_path):
            self._iov_fd = os.open(self.iov_path, os.O_RDWR)
        else:
            self._iov_fd = os.open(
                self.iov_path,
                os.O_CREAT | os.O_RDWR,
                0o644,
            )

        self._iov_capacity = self.mem_mgr.pool_size
        # 预截断文件大小
        os.ftruncate(self._iov_fd, self.IOV_HEADER_SIZE)

    def _create_ior_file(self):
        """
        创建 Ior 文件 (共享内存环形缓冲区)。

        Ior 用于用户进程与 USRBIO daemon 之间传递请求:
        每个条目 64 字节，包含操作类型、偏移、长度、batch_id。
        """
        ior_dir = os.path.dirname(self.ior_path)
        if ior_dir and not os.path.exists(ior_dir):
            os.makedirs(ior_dir, exist_ok=True)

        ior_size = self.IOR_ENTRY_SIZE * self._ior_depth * 2  # 双倍深度
        if os.path.exists(self.ior_path):
            self._ior_fd = os.open(self.ior_path, os.O_RDWR)
        else:
            self._ior_fd = os.open(
                self.ior_path,
                os.O_CREAT | os.O_RDWR,
                0o644,
            )
        os.ftruncate(self._ior_fd, ior_size)

        self._ior_mmap = mmap.mmap(
            self._ior_fd, ior_size,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )

    def _write_iov_header(self):
        """
        写入 Iov 头部元数据。

        头部结构 (4096 字节):
          [0:8]     magic (uint64): 0x4846334653000000
          [8:16]    version (uint64): 2 (GDR 版本)
          [16:24]   capacity (uint64): 总容量
          [24:32]   gpu_ptr (uint64): GPU 虚拟地址
          [32:40]   gpu_phys_addr (uint64): GPU 物理地址
          [40:44]   lkey (uint32): RDMA Local Key
          [44:48]   rkey (uint32): RDMA Remote Key
          [48:56]   mr_handle (uint64): MR 句柄
          [56:64]   ib_dev_name (8 chars): IB 设备名
          [64:72]   gpu_id (uint32): GPU ID
          [72:4096] reserved
        """
        # 获取内存池中第一个缓冲区的信息
        if not self.mem_mgr._pool:
            raise RuntimeError("GPU 内存池为空")

        buf = self.mem_mgr._pool[0]

        header = bytearray(self.IOV_HEADER_SIZE)
        struct.pack_into("Q", header, 0, self.IOR_MAGIC)
        struct.pack_into("Q", header, 8, 2)                  # GDR version
        struct.pack_into("Q", header, 16, buf.size)          # capacity
        struct.pack_into("Q", header, 24, buf.gpu_ptr)       # gpu_ptr
        struct.pack_into("Q", header, 32, buf.gpu_addr)      # gpu_phys_addr
        struct.pack_into("I", header, 40, buf.mr_lkey)
        struct.pack_into("I", header, 44, buf.mr_rkey)
        struct.pack_into("Q", header, 48, buf.mr_handle)

        ib_name = self.ib_dev.encode("ascii")[:8].ljust(8, b"\x00")
        header[56:64] = ib_name
        struct.pack_into("I", header, 64, self.gpu_id)

        os.lseek(self._iov_fd, 0, os.SEEK_SET)
        os.write(self._iov_fd, bytes(header))

    # ------------------------------------------------------------------ #
    #  GPU Iov 缓冲区分配
    # ------------------------------------------------------------------ #

    def alloc_gpu_iov(self, size: int) -> Optional[GDRBuffer]:
        """
        从 GPU 内存池分配 GDR 缓冲区。

        返回的缓冲区可直接用于 3FS RDMA 传输。
        3FS 存储节点通过读取 Iov 头部的 rkey 和 gpu_addr，
        执行 RDMA Write/Read 到 GPU 显存。
        """
        buf = self.mem_mgr.alloc_buffer(size)
        if buf:
            logger.debug(
                f"分配 GPU Iov: ptr=0x{buf.gpu_ptr:x}, "
                f"size={buf.size}, rkey={buf.mr_rkey}"
            )
        return buf

    def free_gpu_iov(self, buf: GDRBuffer):
        """释放 GPU Iov 缓冲区"""
        self.mem_mgr.free_buffer(buf)

    # ------------------------------------------------------------------ #
    #  I/O 请求提交
    # ------------------------------------------------------------------ #

    def submit_read(
        self,
        file_offset: int,
        length: int,
        gdr_buf: GDRBuffer,
    ) -> GDRRequest:
        """
        提交 GDR 读请求。

        3FS 存储节点将通过 RDMA Write 将数据写入 GPU 显存。
        用户在 GPU 侧直接使用数据，无需 CPU 中转。

        Args:
            file_offset: 3FS 文件中的偏移量
            length:       读取长度
            gdr_buf:      目标 GPU 缓冲区

        Returns:
            GDRRequest (status=pending)
        """
        if not self._initialized:
            raise RuntimeError("未初始化")

        with self._lock:
            batch_id = self._batch_counter
            self._batch_counter += 1

        req = GDRRequest(
            op_type=UsrbioOpType.READ,
            gdr_buf=gdr_buf,
            file_offset=file_offset,
            length=length,
            batch_id=batch_id,
        )

        # 写入 Ior 环形缓冲区
        self._enqueue_ior_entry(req)

        with self._lock:
            self._pending_requests[batch_id] = req

        return req

    def submit_write(
        self,
        file_offset: int,
        length: int,
        gdr_buf: GDRBuffer,
    ) -> GDRRequest:
        """
        提交 GDR 写请求。

        3FS 存储节点将通过 RDMA Read 从 GPU 显存读取数据。
        用户 GPU 侧准备好数据后提交即可。
        """
        if not self._initialized:
            raise RuntimeError("未初始化")

        with self._lock:
            batch_id = self._batch_counter
            self._batch_counter += 1

        req = GDRRequest(
            op_type=UsrbioOpType.WRITE,
            gdr_buf=gdr_buf,
            file_offset=file_offset,
            length=length,
            batch_id=batch_id,
        )

        self._enqueue_ior_entry(req)

        with self._lock:
            self._pending_requests[batch_id] = req

        return req

    def submit_batch_read(
        self,
        requests: list[tuple[int, int, GDRBuffer]],
    ) -> list[GDRRequest]:
        """
        批量提交 GDR 读请求。

        利用 3FS USRBIO 的批量 I/O 能力，一次提交多个请求。

        Args:
            requests: [(file_offset, length, gdr_buf), ...]

        Returns:
            GDRRequest 列表
        """
        if not self._initialized:
            raise RuntimeError("未初始化")

        results: list[GDRRequest] = []
        for file_offset, length, gdr_buf in requests:
            req = self.submit_read(file_offset, length, gdr_buf)
            results.append(req)

        return results

    def _enqueue_ior_entry(self, req: GDRRequest):
        """
        向 Ior 环形缓冲区写入请求条目。

        Ior Entry 结构 (64 字节):
          [0:4]    op_type (uint32)
          [4:8]    batch_id (uint32)
          [8:16]   file_offset (uint64)
          [16:24]  length (uint64)
          [24:32]  gpu_ptr (uint64)
          [32:36]  lkey (uint32)
          [36:40]  rkey (uint32)
          [40:48]  gpu_addr (uint64)
          [48:64]  reserved
        """
        entry = bytearray(self.IOR_ENTRY_SIZE)
        struct.pack_into("I", entry, 0, int(req.op_type))
        struct.pack_into("I", entry, 4, req.batch_id)
        struct.pack_into("Q", entry, 8, req.file_offset)
        struct.pack_into("Q", entry, 16, req.length)
        struct.pack_into("Q", entry, 24, req.gdr_buf.gpu_ptr)
        struct.pack_into("I", entry, 32, req.gdr_buf.mr_lkey)
        struct.pack_into("I", entry, 36, req.gdr_buf.mr_rkey)
        struct.pack_into("Q", entry, 40, req.gdr_buf.gpu_addr)

        # 计算写入位置 (环形)
        write_pos = (req.batch_id % self._ior_depth) * self.IOR_ENTRY_SIZE

        if self._ior_mmap:
            self._ior_mmap[write_pos:write_pos + self.IOR_ENTRY_SIZE] = bytes(entry)

    # ------------------------------------------------------------------ #
    #  完成处理
    # ------------------------------------------------------------------ #

    def wait_completion(
        self,
        req: GDRRequest,
        timeout_ms: int = 5000,
    ) -> bool:
        """
        等待指定请求完成。

        轮询 Ior 完成区获取状态。
        """
        import time
        deadline = time.time() + timeout_ms / 1000

        while time.time() < deadline:
            with self._lock:
                if req.status != 0:
                    return req.status == 1
            # 轮询 Ior 完成区
            self._poll_ior_completion()
            time.sleep(1e-5)

        return False

    def wait_all(
        self,
        timeout_ms: int = 30000,
    ) -> int:
        """
        等待所有 pending 请求完成。

        Returns:
            成功完成的请求数
        """
        import time
        deadline = time.time() + timeout_ms / 1000

        while time.time() < deadline:
            self._poll_ior_completion()
            with self._lock:
                if not self._pending_requests:
                    break
            time.sleep(1e-4)

        with self._lock:
            return sum(1 for r in self._completed_requests if r.status == 1)

    def _poll_ior_completion(self):
        """
        轮询 Ior 完成区。

        Ior 完成区位于环形缓冲区后半部分:
          offset = IOR_ENTRY_SIZE * IOR_MAX_DEPTH + batch_id * IOR_ENTRY_SIZE

        完成条目结构与请求条目相同，status 字段被 daemon 写入。
        """
        if not self._ior_mmap:
            return

        completion_base = self.IOR_ENTRY_SIZE * self._ior_depth

        with self._lock:
            pending = list(self._pending_requests.items())

        for batch_id, req in pending:
            comp_pos = completion_base + (
                batch_id % self._ior_depth
            ) * self.IOR_ENTRY_SIZE

            entry_bytes = self._ior_mmap[
                comp_pos:comp_pos + self.IOR_ENTRY_SIZE
            ]
            if len(entry_bytes) < 4:
                continue

            status = struct.unpack_from("I", entry_bytes, 0)[0]
            if status != 0:
                bytes_done = struct.unpack_from("Q", entry_bytes, 16)[0]
                req.status = 1 if status == 1 else -1
                req.bytes_transferred = bytes_done

                with self._lock:
                    self._pending_requests.pop(batch_id, None)
                    self._completed_requests.append(req)

    # ------------------------------------------------------------------ #
    #  清理
    # ------------------------------------------------------------------ #

    def cleanup(self):
        """清理所有资源"""
        if self._ior_mmap:
            self._ior_mmap.close()
            self._ior_mmap = None
        if self._ior_fd >= 0:
            os.close(self._ior_fd)
            self._ior_fd = -1
        if self._iov_fd >= 0:
            os.close(self._iov_fd)
            self._iov_fd = -1
        if self.mem_mgr:
            self.mem_mgr.cleanup()

        self._initialized = False
        logger.info("UsrbioGDRExtension 清理完成")

    def __del__(self):
        if self._initialized:
            self.cleanup()
