"""
3FS GDR 客户端

提供 SGLang 使用的 3FS 高层接口，封装 GDR USRBIO 扩展。
支持:
  1. 挂载 3FS 文件系统 (FUSE 或直接 API)
  2. 通过 GDR 路径读写 3FS 文件
  3. KV Cache 的异步加载/卸载
  4. 模型权重的 GDR 加载

3FS 部署参考:
  - 集群管理: Mgmtd 节点 (RDMA://host:port)
  - 元数据服务: FoundationDB 后端
  - 存储服务: CRAQ 复制协议
  - 客户端: FUSE 或 USRBIO API
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

from ..gdr_core.gpu_memory_manager import GPUMemoryManager, GDRBuffer
from ..gdr_core.gdr_detector import GDRDetector
from .usrbio_gdr import UsrbioGDRExtension, GDRRequest, UsrbioOpType

logger = logging.getLogger(__name__)


@dataclass
class HF3FSGDRConfig:
    """3FS GDR 配置"""
    # 集群配置
    cluster_id: str = "stage"
    mountpoint: str = "/3fs/stage"
    mgmtd_address: str = "RDMA://192.168.1.1:8000"
    token_file: str = "/opt/3fs/etc/token.txt"

    # GDR 配置
    enable_gdr: bool = True
    gpu_id: int = 0
    ib_dev: str = "mlx5_0"
    pool_size_gb: int = 4

    # USRBIO 配置
    iov_path: str = "/3fs/stage/.iov_gpu"
    ior_path: str = "/3fs/stage/.ior_gpu"
    io_depth: int = 64
    batch_size: int = 16

    # 性能调优
    prefetch_depth: int = 4
    enable_batch: bool = True
    relaxed_ordering: bool = True

    # FUSE 回退
    enable_fuse_fallback: bool = True
    fuse_mount_options: str = "rdma_port=1,rdma_timeout=30000"


class HF3FSGDRClient:
    """
    3FS GDR 客户端

    提供 SGLang 的高层 3FS 访问接口:
      - read_to_gpu():  从 3FS 读取数据直接到 GPU 显存
      - write_from_gpu(): 从 GPU 显存写入 3FS
      - load_kv_cache():  异步加载 KV Cache 到 GPU
      - store_kv_cache():  异步存储 KV Cache 到 3FS

    架构:
      SGLang HiCache → HF3FSGDRClient → UsrbioGDRExtension → RDMA → 3FS

    用法:
        client = HF3FSGDRClient(config)
        client.initialize()

        # 读取模型权重到 GPU
        buf = client.read_to_gpu("/models/deepseek.bin", offset=0, length=1024)

        # 加载 KV Cache
        await client.load_kv_cache(
            session_id="sess_001",
            gpu_tensor=kv_cache_tensor,
        )
    """

    def __init__(self, config: Optional[HF3FSGDRConfig] = None):
        self.config = config or HF3FSGDRConfig()
        self._usrbio: Optional[UsrbioGDRExtension] = None
        self._mem_mgr: Optional[GPUMemoryManager] = None
        self._mounted = False
        self._initialized = False

        # 异步 I/O 事件循环
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._io_thread: Optional[threading.Thread] = None

        # 统计
        self._stats = {
            "total_reads": 0,
            "total_writes": 0,
            "total_bytes_read": 0,
            "total_bytes_written": 0,
            "gdr_reads": 0,
            "fallback_reads": 0,
            "avg_latency_us": 0.0,
        }

    # ------------------------------------------------------------------ #
    #  初始化
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        """
        初始化 3FS GDR 客户端。

        步骤:
          1. 检测 GDR 环境
          2. 挂载 3FS (如未挂载)
          3. 初始化 USRBIO GDR 扩展
          4. 启动异步 I/O 线程
        """
        # 1. GDR 环境检测
        if self.config.enable_gdr:
            detector = GDRDetector()
            report = detector.detect()
            if not report["gdr_available"]:
                logger.warning(
                    f"GDR 不可用: {report['warnings']}. "
                    f"回退到 FUSE 路径"
                )
                self.config.enable_gdr = False

        # 2. 挂载 3FS
        self._mount_3fs()

        # 3. 初始化 USRBIO GDR 或 FUSE 回退
        if self.config.enable_gdr:
            self._usrbio = UsrbioGDRExtension(
                iov_path=self.config.iov_path,
                ior_path=self.config.ior_path,
                gpu_id=self.config.gpu_id,
                ib_dev=self.config.ib_dev,
                pool_size_gb=self.config.pool_size_gb,
            )
            if not self._usrbio.initialize():
                logger.warning("USRBIO GDR 初始化失败，回退到 FUSE")
                self.config.enable_gdr = False
                self._usrbio = None

        # 4. 启动异步 I/O 线程
        self._start_io_thread()

        self._initialized = True
        logger.info(
            f"HF3FSGDRClient 初始化成功: "
            f"GDR={'enabled' if self.config.enable_gdr else 'disabled'}, "
            f"mount={self.config.mountpoint}"
        )
        return True

    def _mount_3fs(self):
        """挂载 3FS 文件系统"""
        mountpoint = self.config.mountpoint

        if os.path.ismount(mountpoint):
            self._mounted = True
            logger.info(f"3FS 已挂载于 {mountpoint}")
            return

        if not os.path.exists(mountpoint):
            os.makedirs(mountpoint, exist_ok=True)

        # 构造 hf3fs_fuse_main 命令
        cmd = [
            "hf3fs_fuse_main",
            "--cluster_id", self.config.cluster_id,
            "--mountpoint", mountpoint,
            "--token_file", self.config.token_file,
            "--mgmtd_server_addresses", self.config.mgmtd_address,
        ]

        if self.config.fuse_mount_options:
            cmd.extend(["-o", self.config.fuse_mount_options])

        try:
            # 后台运行 fuse 挂载
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(2)  # 等待挂载

            if os.path.ismount(mountpoint):
                self._mounted = True
                logger.info(f"3FS 挂载成功: {mountpoint}")
            else:
                logger.warning(f"3FS 挂载可能失败，请检查 {mountpoint}")
                self._mounted = False
        except FileNotFoundError:
            logger.warning("hf3fs_fuse_main 未找到，假设已挂载")
            self._mounted = os.path.ismount(mountpoint)

    def _start_io_thread(self):
        """启动异步 I/O 处理线程"""
        def _run_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._io_thread = threading.Thread(
            target=_run_loop, daemon=True, name="hf3fs-gdr-io"
        )
        self._io_thread.start()

    # ------------------------------------------------------------------ #
    #  同步读写接口
    # ------------------------------------------------------------------ #

    def read_to_gpu(
        self,
        file_path: str,
        offset: int = 0,
        length: int = 0,
    ) -> Optional[GDRBuffer]:
        """
        从 3FS 文件读取数据直接到 GPU 显存。

        GDR 路径: 3FS SSD → RDMA Write → GPU 显存 (零拷贝)
        回退路径: 3FS → CPU 内存 → cudaMemcpy → GPU 显存

        Args:
            file_path: 3FS 文件路径
            offset:     文件内偏移
            length:     读取长度 (0=读全部)

        Returns:
            GDRBuffer 包含 GPU 显存中的数据
        """
        if not self._initialized:
            raise RuntimeError("客户端未初始化")

        # 获取文件大小
        if length == 0:
            try:
                length = os.path.getsize(file_path)
            except OSError:
                logger.error(f"无法获取文件大小: {file_path}")
                return None

        start = time.perf_counter()

        if self.config.enable_gdr and self._usrbio:
            buf = self._read_via_gdr(file_path, offset, length)
            self._stats["gdr_reads"] += 1
        else:
            buf = self._read_via_fuse(file_path, offset, length)
            self._stats["fallback_reads"] += 1

        elapsed = (time.perf_counter() - start) * 1e6
        self._stats["total_reads"] += 1
        self._stats["total_bytes_read"] += length
        self._update_avg_latency(elapsed)

        if buf:
            logger.debug(
                f"read_to_gpu: {file_path} [{offset}:{offset+length}] "
                f"→ GPU 0x{buf.gpu_ptr:x} ({elapsed:.1f}us)"
            )

        return buf

    def _read_via_gdr(
        self,
        file_path: str,
        offset: int,
        length: int,
    ) -> Optional[GDRBuffer]:
        """通过 GDR USRBIO 路径读取"""
        # 分配 GPU 缓冲区
        buf = self._usrbio.alloc_gpu_iov(length)
        if buf is None:
            logger.error("GPU 内存分配失败")
            return None

        # 提交 GDR 读请求
        # file_offset 是 3FS 文件内的偏移
        # 3FS 存储节点通过 Iov 头部获取 GPU MR 信息后执行 RDMA Write
        req = self._usrbio.submit_read(
            file_offset=offset,
            length=length,
            gdr_buf=buf,
        )

        # 等待完成
        if not self._usrbio.wait_completion(req, timeout_ms=30000):
            logger.error(f"GDR 读取超时: {file_path}")
            self._usrbio.free_gpu_iov(buf)
            return None

        if req.status != 1:
            logger.error(f"GDR 读取失败: {file_path}, status={req.status}")
            self._usrbio.free_gpu_iov(buf)
            return None

        return buf

    def _read_via_fuse(
        self,
        file_path: str,
        offset: int,
        length: int,
    ) -> Optional[GDRBuffer]:
        """通过 FUSE 路径读取 (回退方案)"""
        try:
            # 分配 CPU 缓冲区
            cpu_buf = bytearray(length)

            with open(file_path, "rb") as f:
                f.seek(offset)
                data = f.read(length)
                cpu_buf[:len(data)] = data
                actual_len = len(data)

            if actual_len < length:
                length = actual_len

            # 通过 PyTorch 或 CUDA 拷贝到 GPU
            buf = self._cpu_to_gpu(bytes(cpu_buf[:length]), length)
            return buf

        except OSError as e:
            logger.error(f"FUSE 读取失败: {file_path}, {e}")
            return None

    def write_from_gpu(
        self,
        file_path: str,
        gdr_buf: GDRBuffer,
        offset: int = 0,
        length: int = 0,
    ) -> bool:
        """
        将 GPU 显存数据写入 3FS 文件。

        GDR 路径: GPU 显存 → RDMA Read → 3FS SSD (零拷贝)
        回退路径: GPU → cudaMemcpy → CPU → FUSE → 3FS
        """
        if not self._initialized:
            raise RuntimeError("客户端未初始化")

        if length == 0:
            length = gdr_buf.size

        start = time.perf_counter()

        if self.config.enable_gdr and self._usrbio:
            req = self._usrbio.submit_write(
                file_offset=offset,
                length=length,
                gdr_buf=gdr_buf,
            )
            success = self._usrbio.wait_completion(req, timeout_ms=30000)
        else:
            success = self._write_via_fuse(file_path, gdr_buf, offset, length)

        elapsed = (time.perf_counter() - start) * 1e6
        self._stats["total_writes"] += 1
        self._stats["total_bytes_written"] += length
        self._update_avg_latency(elapsed)

        return success

    def _write_via_fuse(
        self,
        file_path: str,
        gdr_buf: GDRBuffer,
        offset: int,
        length: int,
    ) -> bool:
        """通过 FUSE 路径写入 (回退方案)"""
        try:
            # GPU → CPU
            data = self._gpu_to_cpu(gdr_buf, length)

            with open(file_path, "wb") as f:
                f.seek(offset)
                f.write(data)

            return True
        except (OSError, RuntimeError) as e:
            logger.error(f"FUSE 写入失败: {file_path}, {e}")
            return False

    def _cpu_to_gpu(self, data: bytes, length: int) -> Optional[GDRBuffer]:
        """将 CPU 数据拷贝到 GPU (使用 PyTorch)"""
        try:
            import torch
            tensor = torch.frombuffer(data, dtype=torch.uint8)
            tensor = tensor.cuda()  # 拷贝到 GPU

            # 获取 GPU 指针
            gpu_ptr = tensor.data_ptr()

            # 创建 GDRBuffer (非池管理)
            buf = GDRBuffer(
                gpu_ptr=gpu_ptr,
                size=length,
                gpu_addr=gpu_ptr,
                mr_handle=0,
                mr_lkey=0,
                mr_rkey=0,
            )
            buf._tensor = tensor  # 防止 GC 回收
            return buf
        except ImportError:
            logger.error("PyTorch 未安装")
            return None

    def _gpu_to_cpu(self, buf: GDRBuffer, length: int) -> bytes:
        """从 GPU 拷贝数据到 CPU"""
        try:
            import torch
            tensor = torch.from_blob(
                buf.gpu_ptr,
                (length,),
                dtype=torch.uint8,
            ).cuda()
            return bytes(tensor.cpu().numpy())
        except ImportError:
            raise RuntimeError("PyTorch 未安装")

    # ------------------------------------------------------------------ #
    #  异步接口 (KV Cache 加载/存储)
    # ------------------------------------------------------------------ #

    async def load_kv_cache(
        self,
        session_id: str,
        gpu_tensor: Any,
        kv_path_prefix: str = "/3fs/stage/kvcache",
    ) -> bool:
        """
        异步加载 KV Cache 到 GPU。

        Args:
            session_id:    会话 ID
            gpu_tensor:    目标 GPU 张量 (PyTorch tensor)
            kv_path_prefix: 3FS 中的 KV Cache 路径前缀
        """
        file_path = f"{kv_path_prefix}/{session_id}.kv"

        # 获取 GPU 张量信息
        try:
            gpu_ptr = gpu_tensor.data_ptr()
            size = gpu_tensor.nbytes
        except AttributeError:
            logger.error("gpu_tensor 不是有效的 PyTorch tensor")
            return False

        if self.config.enable_gdr and self._usrbio:
            return await self._load_kv_gdr(file_path, gpu_ptr, size)
        else:
            return await self._load_kv_fuse(file_path, gpu_tensor)

    async def _load_kv_gdr(
        self, file_path: str, gpu_ptr: int, size: int,
    ) -> bool:
        """通过 GDR 路径加载 KV Cache"""
        loop = asyncio.get_event_loop()

        def _read():
            buf = self._usrbio.alloc_gpu_iov(size)
            if buf is None:
                return False
            req = self._usrbio.submit_read(
                file_offset=0, length=size, gdr_buf=buf,
            )
            return self._usrbio.wait_completion(req, timeout_ms=10000)

        return await loop.run_in_executor(None, _read)

    async def _load_kv_fuse(
        self, file_path: str, gpu_tensor: Any,
    ) -> bool:
        """通过 FUSE 路径加载 KV Cache"""
        loop = asyncio.get_event_loop()

        def _read():
            try:
                import torch
                with open(file_path, "rb") as f:
                    data = f.read()
                tensor = torch.frombuffer(data, dtype=torch.uint8)
                gpu_tensor.copy_(tensor.cuda())
                return True
            except (OSError, RuntimeError) as e:
                logger.error(f"FUSE KV 加载失败: {e}")
                return False

        return await loop.run_in_executor(None, _read)

    async def store_kv_cache(
        self,
        session_id: str,
        gpu_tensor: Any,
        kv_path_prefix: str = "/3fs/stage/kvcache",
    ) -> bool:
        """
        异步存储 KV Cache 到 3FS。

        GDR 路径: GPU 显存 → RDMA Read → 3FS SSD
        """
        file_path = f"{kv_path_prefix}/{session_id}.kv"

        try:
            gpu_ptr = gpu_tensor.data_ptr()
            size = gpu_tensor.nbytes
        except AttributeError:
            logger.error("gpu_tensor 不是有效的 PyTorch tensor")
            return False

        if self.config.enable_gdr and self._usrbio:
            buf = GDRBuffer(
                gpu_ptr=gpu_ptr,
                size=size,
                gpu_addr=gpu_ptr,
                mr_handle=0,
                mr_lkey=0,
                mr_rkey=0,
            )
            req = self._usrbio.submit_write(
                file_offset=0, length=size, gdr_buf=buf,
            )
            return self._usrbio.wait_completion(req, timeout_ms=30000)
        else:
            loop = asyncio.get_event_loop()

            def _write():
                try:
                    import torch
                    cpu_tensor = gpu_tensor.cpu()
                    with open(file_path, "wb") as f:
                        f.write(bytes(cpu_tensor.numpy()))
                    return True
                except (OSError, RuntimeError) as e:
                    logger.error(f"FUSE KV 存储失败: {e}")
                    return False

            return await loop.run_in_executor(None, _write)

    # ------------------------------------------------------------------ #
    #  模型权重加载
    # ------------------------------------------------------------------ #

    def load_model_weights(
        self,
        model_path: str,
        layer_names: Optional[list[str]] = None,
    ) -> dict[str, GDRBuffer]:
        """
        通过 GDR 加载模型权重到 GPU。

        Args:
            model_path:  模型目录 (如 /3fs/stage/models/DeepSeek-R1/)
            layer_names: 要加载的层名列表 (None=全部)

        Returns:
            {layer_name: GDRBuffer} 映射
        """
        weights: dict[str, GDRBuffer] = {}

        model_dir = Path(model_path)
        if not model_dir.exists():
            logger.error(f"模型目录不存在: {model_path}")
            return weights

        # 查找权重文件 (safetensors 或 bin)
        weight_files = sorted(
            list(model_dir.glob("*.safetensors"))
            + list(model_dir.glob("*.bin"))
        )

        for wf in weight_files:
            if self.config.enable_gdr and self._usrbio:
                # 批量加载
                file_size = wf.stat().st_size
                buf = self.read_to_gpu(str(wf), offset=0, length=file_size)
                if buf:
                    weights[wf.name] = buf
            else:
                # FUSE 回退
                file_size = wf.stat().st_size
                buf = self._read_via_fuse(str(wf), 0, file_size)
                if buf:
                    weights[wf.name] = buf

        logger.info(
            f"模型权重加载完成: {len(weights)} 个文件, "
            f"GDR={'yes' if self.config.enable_gdr else 'no'}"
        )
        return weights

    # ------------------------------------------------------------------ #
    #  统计与诊断
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        """获取 I/O 统计"""
        return dict(self._stats)

    def get_bandwidth_gbps(self) -> float:
        """获取当前读带宽 (GB/s)"""
        if self._stats["avg_latency_us"] > 0 and self._stats["total_reads"] > 0:
            avg_size = (
                self._stats["total_bytes_read"]
                / max(1, self._stats["total_reads"])
            )
            return avg_size / (self._stats["avg_latency_us"] * 1e-6) / 1e9
        return 0.0

    def _update_avg_latency(self, latency_us: float):
        """更新平均延迟 (指数移动平均)"""
        alpha = 0.1
        if self._stats["avg_latency_us"] == 0:
            self._stats["avg_latency_us"] = latency_us
        else:
            self._stats["avg_latency_us"] = (
                alpha * latency_us
                + (1 - alpha) * self._stats["avg_latency_us"]
            )

    # ------------------------------------------------------------------ #
    #  清理
    # ------------------------------------------------------------------ #

    def cleanup(self):
        """清理所有资源"""
        if self._usrbio:
            self._usrbio.cleanup()
            self._usrbio = None

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._io_thread:
            self._io_thread.join(timeout=5)

        self._initialized = False
        logger.info("HF3FSGDRClient 清理完成")
