"""
SGLang HiCache GDR 存储后端

实现 SGLang HiCache 的 StorageBackend 接口，
使用 GDR (GPUDirect RDMA) 通过 3FS 进行 KV Cache 的存储和加载。

SGLang HiCache 架构:
  ┌─────────────────────────────────────────┐
  │              GPU Memory                   │
  │  ┌──────────────────────────────────┐    │
  │  │     L1: GPU KV Cache (VRAM)      │    │
  │  └───────────┬──────────────────────┘    │
  │              │ evict / promote            │
  │  ┌───────────▼──────────────────────┐    │
  │  │  L2: Host KV Cache (CPU DRAM)     │    │
  │  └───────────┬──────────────────────┘    │
  │              │ evict / promote            │
  │  ┌───────────▼──────────────────────┐    │
  │  │  L3: GDR Storage Backend (3FS)    │    │  ← 本模块
  │  │     via GPUDirect RDMA            │    │
  │  └──────────────────────────────────┘    │
  └─────────────────────────────────────────┘

对比 SGLang 原有 hf3fs 后端:
  原有: GPU → CPU Memory → FUSE/USRBIO(CPU) → 3FS  (2次拷贝)
  GDR:  GPU → RDMA → 3FS                                  (0次拷贝)

适配 SGLang 接口:
  实现 get(key) → bytes/None
  实现 set(key, value) → bool
  实现 exists(key) → bool
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

from ..hf3fs_bridge.hf3fs_gdr_client import HF3FSGDRClient, HF3FSGDRConfig
from ..gdr_core.gpu_memory_manager import GDRBuffer

logger = logging.getLogger(__name__)


@dataclass
class GDRStorageConfig:
    """GDR 存储后端配置"""
    storage_prefix: str = "/3fs/stage/kvcache"
    enable_gdr: bool = True
    gpu_id: int = 0
    ib_dev: str = "mlx5_0"
    pool_size_gb: int = 4
    write_policy: str = "write_through"   # write_through | write_back
    prefetch_policy: str = "wait_complete"
    compression: bool = False
    encryption: bool = False


class GDRStorageBackend:
    """
    SGLang HiCache GDR 存储后端

    实现 HiCache 的三级存储接口:
      - get(key):    从 3FS 读取 KV Cache 到 GPU 显存
      - set(key, value): 将 GPU KV Cache 写入 3FS
      - exists(key): 检查 KV Cache 是否存在于 3FS
      - delete(key): 删除 3FS 中的 KV Cache
      - batch_get:   批量读取 (利用 GDR 批量传输)
      - prefetch:    预取 (异步 GDR 传输)

    用法 (与 SGLang 集成):
        # SGLang 启动时
        backend = GDRStorageBackend(config)
        backend.initialize()

        # 在 SGLang HiCache 中注册
        hicache.register_storage_backend("gdr_3fs", backend)

        # SGLang HiCache 自动调用:
        #   - KV Cache 淘汰时: backend.set(key, gpu_tensor)
        #   - KV Cache 缺失时: backend.get(key) → GPU tensor
    """

    def __init__(self, config: Optional[GDRStorageConfig] = None):
        self.config = config or GDRStorageConfig()

        # 3FS GDR 客户端
        hf3fs_config = HF3FSGDRConfig(
            cluster_id="stage",
            mountpoint="/3fs/stage",
            mgmtd_address="RDMA://192.168.1.1:8000",
            enable_gdr=self.config.enable_gdr,
            gpu_id=self.config.gpu_id,
            ib_dev=self.config.ib_dev,
            pool_size_gb=self.config.pool_size_gb,
        )
        self.client = HF3FSGDRClient(hf3fs_config)

        self._initialized = False
        self._lock = asyncio.Lock()

        # 预取队列
        self._prefetch_queue: asyncio.Queue = asyncio.Queue()
        self._prefetch_task: Optional[asyncio.Task] = None

        # 统计
        self._stats = {
            "gets": 0, "sets": 0, "deletes": 0, "exists": 0,
            "bytes_get": 0, "bytes_set": 0,
            "gdr_hits": 0, "fallback_hits": 0,
            "prefetch_hits": 0, "prefetch_misses": 0,
            "avg_get_latency_us": 0.0,
            "avg_set_latency_us": 0.0,
        }

    # ------------------------------------------------------------------ #
    #  生命周期
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        """初始化存储后端"""
        if not self.client.initialize():
            logger.error("3FS GDR 客户端初始化失败")
            return False

        # 确保存储目录存在
        storage_dir = Path(self.config.storage_prefix)
        storage_dir.mkdir(parents=True, exist_ok=True)

        # 启动预取协程
        loop = asyncio.get_event_loop()
        if loop.is_running():
            self._prefetch_task = loop.create_task(self._prefetch_worker())
        else:
            logger.warning("事件循环未运行，预取功能将在首次调用时启动")

        self._initialized = True
        logger.info(
            f"GDRStorageBackend 初始化: prefix={self.config.storage_prefix}"
        )
        return True

    def shutdown(self):
        """关闭存储后端"""
        if self._prefetch_task:
            self._prefetch_task.cancel()
        self.client.cleanup()
        self._initialized = False

    # ------------------------------------------------------------------ #
    #  SGLang HiCache 接口实现
    # ------------------------------------------------------------------ #

    def get(self, key: str) -> Optional[bytes]:
        """
        从 3FS 读取 KV Cache 数据到 CPU (兼容接口)。

        SGLang HiCache 调用此方法获取存储的 KV Cache。
        GDR 优化: 数据通过 GDR 路径传输到 GPU，再拷贝回 CPU。
        更好的做法是使用 get_to_gpu() 直接获取 GPU tensor。
        """
        start = time.perf_counter()
        self._stats["gets"] += 1

        file_path = self._key_to_path(key)

        if not os.path.exists(file_path):
            return None

        # 读取到 GPU (GDR 优先)
        buf = self.client.read_to_gpu(file_path)
        if buf is None:
            return None

        # GPU → CPU (兼容接口需要返回 bytes)
        try:
            import torch
            tensor = torch.from_blob(
                buf.gpu_ptr,
                (buf.size,),
                dtype=torch.uint8,
            )
            data = bytes(tensor.cpu().numpy())
            self._stats["gdr_hits"] += 1
            self._stats["bytes_get"] += len(data)

            elapsed = (time.perf_counter() - start) * 1e6
            self._update_latency("get", elapsed)
            return data
        except Exception as e:
            logger.error(f"GPU→CPU 转换失败: {e}")
            return None
        finally:
            self.client._usrbio.free_gpu_iov(buf) if self.client._usrbio else None

    async def get_to_gpu(
        self,
        key: str,
        gpu_tensor: Any,
    ) -> bool:
        """
        直接从 3FS 读取 KV Cache 到 GPU tensor (GDR 零拷贝)。

        这是 GDR 后端的核心优化接口:
        数据从 3FS 通过 RDMA 直接写入 GPU tensor 的显存。

        Args:
            key:        KV Cache 键 (如 session_id)
            gpu_tensor: 目标 GPU 张量 (PyTorch)

        Returns:
            True 成功, False 失败
        """
        start = time.perf_counter()
        self._stats["gets"] += 1

        file_path = self._key_to_path(key)
        if not os.path.exists(file_path):
            return False

        success = await self.client.load_kv_cache(
            session_id=key,
            gpu_tensor=gpu_tensor,
            kv_path_prefix=self.config.storage_prefix,
        )

        if success:
            self._stats["gdr_hits"] += 1
            self._stats["bytes_get"] += gpu_tensor.nbytes
        else:
            self._stats["fallback_hits"] += 1

        elapsed = (time.perf_counter() - start) * 1e6
        self._update_latency("get", elapsed)
        return success

    def set(self, key: str, value: bytes) -> bool:
        """
        将 KV Cache 数据存储到 3FS。

        SGLang HiCache 在 L2 淘汰时调用此方法。
        """
        start = time.perf_counter()
        self._stats["sets"] += 1

        file_path = self._key_to_path(key)

        try:
            # CPU → GPU → GDR → 3FS
            buf = self.client._cpu_to_gpu(value, len(value))
            if buf is None:
                # 回退: 直接写 FUSE
                with open(file_path, "wb") as f:
                    f.write(value)
                self._stats["bytes_set"] += len(value)
                return True

            success = self.client.write_from_gpu(file_path, buf, length=len(value))

            if success:
                self._stats["bytes_set"] += len(value)

            elapsed = (time.perf_counter() - start) * 1e6
            self._update_latency("set", elapsed)
            return success
        except Exception as e:
            logger.error(f"set 失败: {key}, {e}")
            return False

    async def set_from_gpu(
        self,
        key: str,
        gpu_tensor: Any,
    ) -> bool:
        """
        直接从 GPU tensor 存储 KV Cache 到 3FS (GDR 零拷贝)。
        """
        start = time.perf_counter()
        self._stats["sets"] += 1

        file_path = self._key_to_path(key)

        success = await self.client.store_kv_cache(
            session_id=key,
            gpu_tensor=gpu_tensor,
            kv_path_prefix=self.config.storage_prefix,
        )

        if success:
            self._stats["bytes_set"] += gpu_tensor.nbytes

        elapsed = (time.perf_counter() - start) * 1e6
        self._update_latency("set", elapsed)
        return success

    def exists(self, key: str) -> bool:
        """检查 KV Cache 是否存在于 3FS"""
        self._stats["exists"] += 1
        file_path = self._key_to_path(key)
        return os.path.exists(file_path)

    def delete(self, key: str) -> bool:
        """删除 3FS 中的 KV Cache"""
        self._stats["deletes"] += 1
        file_path = self._key_to_path(key)
        try:
            os.remove(file_path)
            return True
        except FileNotFoundError:
            return False

    # ------------------------------------------------------------------ #
    #  批量操作与预取
    # ------------------------------------------------------------------ #

    async def batch_get_to_gpu(
        self,
        keys: list[str],
        gpu_tensors: list[Any],
    ) -> list[bool]:
        """
        批量从 3FS 读取 KV Cache 到 GPU。

        利用 GDR 的批量 RDMA 能力，减少单次传输的协议开销。
        """
        if len(keys) != len(gpu_tensors):
            raise ValueError("keys 和 gpu_tensors 长度不匹配")

        tasks = [
            self.get_to_gpu(key, tensor)
            for key, tensor in zip(keys, gpu_tensors)
        ]
        results = await asyncio.gather(*tasks)
        return results

    def prefetch(self, key: str):
        """
        异步预取 KV Cache 到 GPU。

        SGLang 调度器预测下一段 prompt 的 KV Cache 可能被命中时，
       提前通过 GDR 预取到 GPU。
        """
        self._prefetch_queue.put_nowait(key)
        logger.debug(f"预取请求: {key}")

    async def _prefetch_worker(self):
        """预取工作协程"""
        while True:
            try:
                key = await self._prefetch_queue.get()
                file_path = self._key_to_path(key)

                if not os.path.exists(file_path):
                    self._stats["prefetch_misses"] += 1
                    continue

                # 预取到 GPU (等待完成)
                buf = self.client.read_to_gpu(file_path)
                if buf:
                    self._stats["prefetch_hits"] += 1
                    # 保留在 GPU 内存池中 (不立即释放)
                    logger.debug(f"预取完成: {key}")
                else:
                    self._stats["prefetch_misses"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"预取错误: {e}")

    # ------------------------------------------------------------------ #
    #  辅助方法
    # ------------------------------------------------------------------ #

    def _key_to_path(self, key: str) -> str:
        """将 KV Cache key 转换为 3FS 文件路径"""
        # 使用 hash 分片存储，避免单目录文件过多
        key_hash = hashlib.md5(key.encode()).hexdigest()
        shard_dir = key_hash[:2]
        return os.path.join(
            self.config.storage_prefix,
            shard_dir,
            f"{key_hash}.kv",
        )

    def _update_latency(self, op: str, latency_us: float):
        """更新延迟统计"""
        key = f"avg_{op}_latency_us"
        alpha = 0.1
        if self._stats[key] == 0:
            self._stats[key] = latency_us
        else:
            self._stats[key] = (
                alpha * latency_us + (1 - alpha) * self._stats[key]
            )

    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = dict(self._stats)
        stats["gdr_ratio"] = (
            stats["gdr_hits"] / max(1, stats["gets"])
        )
        return stats
