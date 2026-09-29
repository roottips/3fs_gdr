"""
SGLang HiCache GDR I/O 后端

实现 SGLang HiCache 的 IOBackend 接口，
控制 GDR 数据传输的 I/O 策略和调度。

SGLang HiCache I/O 后端类型:
  - direct:   直接 I/O (绕过内核 page cache)
  - buffered: 缓冲 I/O (使用 page cache)
  - aio:      异步 I/O (Linux AIO)

GDR I/O 后端新增类型:
  - gdr:      GPUDirect RDMA I/O (直接 GPU↔3FS, 零拷贝)
  - gdr_async: GDR 异步 I/O (批量提交, 延迟轮询)

适配 SGLang 启动参数:
  --hicache-io-backend gdr
  --hicache-io-backend gdr_async
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Any

from ..gdr_core.gpu_memory_manager import GDRBuffer
from ..hf3fs_bridge.usrbio_gdr import UsrbioGDRExtension, GDRRequest

logger = logging.getLogger(__name__)


class IOBackendType(IntEnum):
    """I/O 后端类型"""
    DIRECT = 1
    BUFFERED = 2
    AIO = 3
    GDR = 4          # GPUDirect RDMA (同步)
    GDR_ASYNC = 5    # GPUDirect RDMA (异步批量)


@dataclass
class GDRIOConfig:
    """GDR I/O 后端配置"""
    backend_type: IOBackendType = IOBackendType.GDR
    io_depth: int = 64             # 并发 I/O 深度
    batch_size: int = 16           # 批量大小
    poll_interval_us: int = 1       # CQ 轮询间隔
    enable_relaxed_ordering: bool = True
    mem_layout: str = "page_first_direct"


class GDRIOBackend:
    """
    SGLang HiCache GDR I/O 后端

    负责管理 GDR 数据传输的 I/O 调度:
      1. I/O 深度控制: 限制并发 GDR 传输数量
      2. 批量提交: 聚合多个请求减少协议开销
      3. 完成轮询: 高效轮询 CQ 获取完成事件
      4. 内存布局: GPU 内存池布局策略

    I/O 路径对比:
      direct:   3FS → page bypass → CPU → cudaMemcpy → GPU
      gdr:      3FS → RDMA → GPU (同步等待)
      gdr_async: 3FS → RDMA → GPU (批量提交, 异步等待)

    用法:
        io_backend = GDRIOBackend(config, usrbio_ext)
        io_backend.initialize()

        # 异步批量读取
        futures = io_backend.submit_batch_read(
            file_offsets=[0, 4096, 8192],
            lengths=[4096, 4096, 4096],
            gdr_bufs=[buf0, buf1, buf2],
        )
        results = io_backend.wait_batch(futures)
    """

    def __init__(
        self,
        config: Optional[GDRIOConfig] = None,
        usrbio_ext: Optional[UsrbioGDRExtension] = None,
    ):
        self.config = config or GDRIOConfig()
        self.usrbio = usrbio_ext

        self._initialized = False
        self._pending_depth = 0
        self._max_depth = self.config.io_depth

        # 异步批量队列
        self._batch_queue: list[GDRRequest] = []
        self._batch_results: dict[int, GDRRequest] = {}

        # 统计
        self._stats = {
            "total_ios": 0,
            "batch_ios": 0,
            "total_batches": 0,
            "bytes_transferred": 0,
            "avg_batch_size": 0.0,
            "avg_latency_us": 0.0,
            "max_depth_reached": 0,
        }

    # ------------------------------------------------------------------ #
    #  生命周期
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        if self.usrbio is None:
            logger.error("USRBIO 扩展未提供")
            return False
        self._initialized = True
        logger.info(
            f"GDRIOBackend 初始化: type={self.config.backend_type.name}, "
            f"depth={self.config.io_depth}, batch={self.config.batch_size}"
        )
        return True

    # ------------------------------------------------------------------ #
    #  同步 I/O
    # ------------------------------------------------------------------ #

    def read(
        self,
        file_offset: int,
        length: int,
        gdr_buf: GDRBuffer,
    ) -> bool:
        """
        同步 GDR 读取。

        提交请求并等待完成。
        """
        if not self._initialized:
            raise RuntimeError("未初始化")

        start = time.perf_counter()

        req = self.usrbio.submit_read(file_offset, length, gdr_buf)
        success = self.usrbio.wait_completion(req, timeout_ms=30000)

        elapsed = (time.perf_counter() - start) * 1e6
        self._update_stats(1, length, elapsed)

        return success

    def write(
        self,
        file_offset: int,
        length: int,
        gdr_buf: GDRBuffer,
    ) -> bool:
        """同步 GDR 写入"""
        if not self._initialized:
            raise RuntimeError("未初始化")

        start = time.perf_counter()

        req = self.usrbio.submit_write(file_offset, length, gdr_buf)
        success = self.usrbio.wait_completion(req, timeout_ms=30000)

        elapsed = (time.perf_counter() - start) * 1e6
        self._update_stats(1, length, elapsed)

        return success

    # ------------------------------------------------------------------ #
    #  异步批量 I/O
    # ------------------------------------------------------------------ #

    def submit_batch_read(
        self,
        file_offsets: list[int],
        lengths: list[int],
        gdr_bufs: list[GDRBuffer],
    ) -> list[int]:
        """
        异步批量提交 GDR 读请求。

        聚合多个请求一次性提交，利用 3FS 的批量 I/O 能力。
        不等待完成，返回 batch_id 列表。

        Args:
            file_offsets: 文件偏移列表
            lengths:       长度列表
            gdr_bufs:      GPU 缓冲区列表

        Returns:
            batch_id 列表
        """
        if not self._initialized:
            raise RuntimeError("未初始化")

        batch_ids: list[int] = []

        for file_offset, length, gdr_buf in zip(
            file_offsets, lengths, gdr_bufs
        ):
            req = self.usrbio.submit_read(file_offset, length, gdr_buf)
            batch_ids.append(req.batch_id)
            self._pending_depth += 1

            self._stats["max_depth_reached"] = max(
                self._stats["max_depth_reached"], self._pending_depth
            )

        self._stats["batch_ios"] += len(batch_ids)
        self._stats["total_batches"] += 1
        self._stats["avg_batch_size"] = (
            self._stats["batch_ios"] / self._stats["total_batches"]
        )

        return batch_ids

    def wait_batch(
        self,
        batch_ids: list[int],
        timeout_ms: int = 30000,
    ) -> list[bool]:
        """
        等待一批异步请求完成。

        一次轮询所有请求的完成状态，而非逐个等待。
        """
        start = time.perf_counter()
        deadline = time.time() + timeout_ms / 1000

        results: list[bool] = [False] * len(batch_ids)
        completed = 0

        while completed < len(batch_ids) and time.time() < deadline:
            self.usrbio._poll_ior_completion()

            for i, batch_id in enumerate(batch_ids):
                if results[i]:
                    continue
                req = self.usrbio._pending_requests.get(batch_id)
                if req and req.status != 0:
                    results[i] = (req.status == 1)
                    completed += 1
                    self._pending_depth -= 1

            time.sleep(self.config.poll_interval_us * 1e-6)

        elapsed = (time.perf_counter() - start) * 1e6
        total_bytes = sum(
            self.usrbio._completed_requests[-1].bytes_transferred
            for _ in batch_ids
            if self.usrbio._completed_requests
        )
        self._update_stats(len(batch_ids), total_bytes, elapsed)

        return results

    # ------------------------------------------------------------------ #
    #  预取支持
    # ------------------------------------------------------------------ #

    def prefetch(
        self,
        file_offset: int,
        length: int,
        gdr_buf: GDRBuffer,
    ):
        """
        异步预取 (不等待完成)。

        SGLang 调度器预测后续会读取此区域时提前预取。
        """
        req = self.usrbio.submit_read(file_offset, length, gdr_buf)
        return req.batch_id

    def check_prefetch(self, batch_id: int) -> bool:
        """检查预取是否完成"""
        req = self.usrbio._pending_requests.get(batch_id)
        if req:
            return req.status == 1
        # 可能已完成
        for r in self.usrbio._completed_requests:
            if r.batch_id == batch_id:
                return r.status == 1
        return False

    # ------------------------------------------------------------------ #
    #  统计
    # ------------------------------------------------------------------ #

    def _update_stats(
        self,
        num_ios: int,
        bytes_transferred: int,
        latency_us: float,
    ):
        self._stats["total_ios"] += num_ios
        self._stats["bytes_transferred"] += bytes_transferred

        alpha = 0.1
        if self._stats["avg_latency_us"] == 0:
            self._stats["avg_latency_us"] = latency_us
        else:
            self._stats["avg_latency_us"] = (
                alpha * latency_us + (1 - alpha) * self._stats["avg_latency_us"]
            )

    def get_stats(self) -> dict:
        return dict(self._stats)

    def get_bandwidth_gbps(self) -> float:
        """获取当前带宽 (GB/s)"""
        if self._stats["avg_latency_us"] > 0:
            avg_size = (
                self._stats["bytes_transferred"]
                / max(1, self._stats["total_ios"])
            )
            return avg_size / (self._stats["avg_latency_us"] * 1e-6) / 1e9
        return 0.0
