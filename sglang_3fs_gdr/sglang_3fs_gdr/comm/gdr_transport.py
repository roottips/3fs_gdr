"""
SGLang 通信层 GDR 传输适配

实现 SGLang relay 后端接口，使用 GPUDirect RDMA 进行跨节点 GPU 数据传输。

SGLang 通信架构:
  Relay Backend Interface (sglang_omni/relay/base.py)
    ├── CudaIPCRelay     (同节点 GPU 传输)
    ├── SHMRelay          (同节点 CPU 共享内存)
    ├── MooncakeRelay     (跨节点 RDMA)
    └── GDRRelayBackend   (跨节点 GPUDirect RDMA)  ← 本模块

GDRRelayBackend 对比 MooncakeRelay:
  Mooncake: GPU → Mooncake → CPU → RDMA → CPU → GPU (可能经过 CPU)
  GDR:      GPU → RDMA → GPU (完全绕过 CPU)

适配接口:
  - put_async(data_ref):  异步发送 GPU 数据到远端
  - get_async(data_ref):  异步从远端获取 GPU 数据
  - close():              关闭连接

Router 集成:
  当 GDR 检测可用时，Router 优先选择 GDR 传输:
    if gdr_available and cross_node:
        transport = GDRTransportType.GDR
    elif cross_node:
        transport = GDRTransportType.MOONCAKE
    else:
        transport = GDRTransportType.CUDA_IPC
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Any

from ..gdr_core.gpu_memory_manager import GPUMemoryManager, GDRBuffer
from ..gdr_core.rdma_ops import RDMATransport, RDMAOps, RDMAEndpoint, TransferResult
from ..gdr_core.gdr_detector import GDRDetector
from .gdr_data_ref import GDRDataRef, GDRTransportType

logger = logging.getLogger(__name__)


@dataclass
class GDRTransportConfig:
    """GDR 传输配置"""
    gpu_id: int = 0
    ib_dev: str = "mlx5_0"
    pool_size_gb: int = 4
    cq_depth: int = 256
    qp_depth: int = 128
    enable_batch: bool = True
    max_retries: int = 3
    timeout_ms: int = 30000


class GDRTransport:
    """
    GDR 传输底层封装

    管理 RDMA 连接和 GPU 内存传输。
    每个 GDRTransport 实例管理一个 GPU-NIC 配对。
    """

    def __init__(self, config: Optional[GDRTransportConfig] = None):
        self.config = config or GDRTransportConfig()

        self.mem_mgr = GPUMemoryManager(
            gpu_id=self.config.gpu_id,
            ib_dev=self.config.ib_dev,
            pool_size_gb=self.config.pool_size_gb,
        )
        self.rdma_ops = RDMAOps(
            gpu_id=self.config.gpu_id,
            ib_dev=self.config.ib_dev,
            pool_size_gb=self.config.pool_size_gb,
        )

        self._initialized = False
        self._endpoints: dict[int, RDMAEndpoint] = {}  # rank -> endpoint

    def initialize(self) -> bool:
        """初始化传输层"""
        if not self.rdma_ops.initialize():
            logger.error("RDMA 操作初始化失败")
            return False
        self._initialized = True
        logger.info(
            f"GDRTransport 初始化: GPU={self.config.gpu_id}, "
            f"IB={self.config.ib_dev}"
        )
        return True

    def register_endpoint(
        self,
        rank: int,
        endpoint: RDMAEndpoint,
    ):
        """注册远端端点"""
        self._endpoints[rank] = endpoint
        self.rdma_ops.transport.connect(endpoint)
        logger.info(f"端点注册: rank={rank}, qpn={endpoint.qpn}")

    def send_gpu_buffer(
        self,
        dst_rank: int,
        gdr_buf: GDRBuffer,
        length: int,
    ) -> TransferResult:
        """
        发送 GPU 缓冲区到远端 (RDMA Write)。

        数据流: 本地 GPU 显存 → RDMA → 远端 GPU 显存
        """
        endpoint = self._endpoints.get(dst_rank)
        if endpoint is None:
            return TransferResult(False, 0, 0, f"未找到 rank={dst_rank} 的端点")

        return self.rdma_ops.transport.rdma_write(
            local_buf=gdr_buf,
            remote_addr=endpoint.remote_addr,
            rkey=endpoint.rkey,
            length=length,
        )

    def recv_gpu_buffer(
        self,
        src_rank: int,
        gdr_buf: GDRBuffer,
        length: int,
    ) -> TransferResult:
        """
        从远端接收数据到 GPU 缓冲区 (RDMA Read)。

        数据流: 远端 GPU/内存 → RDMA → 本地 GPU 显存
        """
        endpoint = self._endpoints.get(src_rank)
        if endpoint is None:
            return TransferResult(False, 0, 0, f"未找到 rank={src_rank} 的端点")

        return self.rdma_ops.transport.rdma_read(
            local_buf=gdr_buf,
            remote_addr=endpoint.remote_addr,
            rkey=endpoint.rkey,
            length=length,
        )

    def cleanup(self):
        """清理传输层"""
        if self.rdma_ops:
            self.rdma_ops.cleanup()
        self._initialized = False


class GDRRelayBackend:
    """
    SGLang Relay GDR 后端

    实现 SGLang relay 后端接口，使用 GDR 进行跨节点 GPU 数据传输。

    与 SGLang 的集成点:
      1. 在 sglang_omni/relay/base.py 的后端注册表中注册 "gdr"
      2. Router 在检测到 GDR 可用时自动选择此后端
      3. Stage 之间的 DataRef 通过 GDR 传输

    用法 (SGLang 集成):
        # 在 SGLang 初始化时
        from sglang_3fs_gdr.comm import GDRRelayBackend

        backend = GDRRelayBackend(config)
        backend.initialize()

        # 注册到 SGLang relay
        sglang_omni.relay.register_backend("gdr", backend)

        # Router 自动选择:
        # cross_node + gdr_available → GDR
        # cross_node + no gdr        → Mooncake
        # same_node                  → CUDA IPC
    """

    BACKEND_NAME = "gdr"

    def __init__(self, config: Optional[GDRTransportConfig] = None):
        self.config = config or GDRTransportConfig()
        self.transport = GDRTransport(self.config)

        self._initialized = False
        self._gdr_available = False

        # 异步操作追踪
        self._pending_ops: dict[str, asyncio.Future] = {}

    # ------------------------------------------------------------------ #
    #  生命周期
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        """
        初始化 GDR Relay 后端。

        步骤:
          1. 检测 GDR 环境
          2. 初始化传输层
          3. 建立 RDMA 连接
        """
        # 1. GDR 检测
        detector = GDRDetector()
        report = detector.detect()
        self._gdr_available = report["gdr_available"]

        if not self._gdr_available:
            logger.warning(
                f"GDR 不可用: {report['warnings']}. "
                f"请配置 GDR 环境后重试"
            )
            return False

        # 2. 传输层初始化
        if not self.transport.initialize():
            logger.error("GDR 传输层初始化失败")
            return False

        self._initialized = True
        logger.info("GDRRelayBackend 初始化成功")
        return True

    def is_available(self) -> bool:
        """检查 GDR 后端是否可用"""
        return self._gdr_available and self._initialized

    # ------------------------------------------------------------------ #
    #  SGLang Relay 接口
    # ------------------------------------------------------------------ #

    async def put_async(self, data_ref: GDRDataRef) -> bool:
        """
        异步发送 GPU 数据到远端。

        实现 SGLang relay 的 put_async 接口:
        将本地 GPU 内存通过 RDMA Write 发送到远端。

        Args:
            data_ref: GDR 数据引用 (包含 GPU 内存地址和 MR 信息)

        Returns:
            True 成功, False 失败
        """
        if not self._initialized:
            raise RuntimeError("GDR 后端未初始化")

        if not data_ref.is_gdr():
            logger.warning(f"非 GDR DataRef, 跳过: {data_ref.transport}")
            return False

        # 从内存池获取缓冲区
        buf = self.transport.mem_mgr.alloc_buffer(data_ref.size)
        if buf is None:
            logger.error(f"内存分配失败: {data_ref.size} bytes")
            return False

        try:
            result = self.transport.send_gpu_buffer(
                dst_rank=data_ref.dst_rank,
                gdr_buf=buf,
                length=data_ref.size,
            )

            if not result.success:
                logger.error(
                    f"GDR put 失败: {data_ref.object_id}, "
                    f"error={result.error}"
                )
                return False

            logger.debug(
                f"GDR put: {data_ref.object_id} → rank={data_ref.dst_rank}, "
                f"{result.bytes_transferred} bytes, {result.latency_us:.1f}us"
            )
            return True

        finally:
            self.transport.mem_mgr.free_buffer(buf)

    async def get_async(self, data_ref: GDRDataRef) -> Optional[GDRBuffer]:
        """
        异步从远端获取 GPU 数据。

        实现 SGLang relay 的 get_async 接口:
        通过 RDMA Read 从远端读取数据到本地 GPU 显存。

        Args:
            data_ref: GDR 数据引用

        Returns:
            GDRBuffer (包含 GPU 数据) 或 None
        """
        if not self._initialized:
            raise RuntimeError("GDR 后端未初始化")

        if not data_ref.is_gdr():
            logger.warning(f"非 GDR DataRef, 跳过: {data_ref.transport}")
            return None

        # 分配 GPU 接收缓冲区
        buf = self.transport.mem_mgr.alloc_buffer(data_ref.size)
        if buf is None:
            logger.error(f"内存分配失败: {data_ref.size} bytes")
            return None

        result = self.transport.recv_gpu_buffer(
            src_rank=data_ref.src_rank,
            gdr_buf=buf,
            length=data_ref.size,
        )

        if not result.success:
            logger.error(
                f"GDR get 失败: {data_ref.object_id}, "
                f"error={result.error}"
            )
            self.transport.mem_mgr.free_buffer(buf)
            return None

        logger.debug(
            f"GDR get: {data_ref.object_id} ← rank={data_ref.src_rank}, "
            f"{result.bytes_transferred} bytes, {result.latency_us:.1f}us"
        )

        return buf

    async def batch_get(
        self,
        data_refs: list[GDRDataRef],
    ) -> list[Optional[GDRBuffer]]:
        """
        批量异步获取 GPU 数据。

        利用 RDMA 批量传输能力，减少单次传输的协议开销。
        """
        if not self._initialized:
            raise RuntimeError("GDR 后端未初始化")

        tasks = [self.get_async(ref) for ref in data_refs]
        results = await asyncio.gather(*tasks)
        return results

    # ------------------------------------------------------------------ #
    #  端点管理
    # ------------------------------------------------------------------ #

    def register_remote(
        self,
        rank: int,
        endpoint_info: dict,
    ):
        """
        注册远端 GDR 端点。

        在 SGLang 分布式初始化时调用:
        所有 rank 通过 TCP 控制面交换 GDR 端点信息。

        Args:
            rank:          远端 rank
            endpoint_info: {lid, qpn, psn, rkey, remote_addr}
        """
        endpoint = RDMAEndpoint(
            lid=endpoint_info["lid"],
            qpn=endpoint_info["qpn"],
            psn=endpoint_info.get("psn", 0),
            rkey=endpoint_info["rkey"],
            remote_addr=endpoint_info["remote_addr"],
        )
        self.transport.register_endpoint(rank, endpoint)

    def get_local_endpoint_info(self) -> dict:
        """
        获取本地 GDR 端点信息 (用于广播给其他 rank)。

        Returns:
            {lid, qpn, psn, rkey, remote_addr, gpu_id, ib_dev}
        """
        transport = self.transport.rdma_ops.transport
        return {
            "lid": 0,  # 实际从 ibv_query_port 获取
            "qpn": 0,  # 实际从 QP 获取
            "psn": 0,
            "rkey": (
                self.transport.mem_mgr._pool[0].mr_rkey
                if self.transport.mem_mgr._pool
                else 0
            ),
            "remote_addr": (
                self.transport.mem_mgr._pool[0].gpu_addr
                if self.transport.mem_mgr._pool
                else 0
            ),
            "gpu_id": self.config.gpu_id,
            "ib_dev": self.config.ib_dev,
        }

    # ------------------------------------------------------------------ #
    #  清理
    # ------------------------------------------------------------------ #

    def close(self):
        """关闭 GDR 后端"""
        self.transport.cleanup()
        self._initialized = False
        logger.info("GDRRelayBackend 已关闭")

    def get_stats(self) -> dict:
        """获取传输统计"""
        return {
            "gdr_available": self._gdr_available,
            "initialized": self._initialized,
            "endpoints": len(self.transport._endpoints),
            "pool_buffers": len(self.transport.mem_mgr._pool),
            "pool_free": len(self.transport.mem_mgr._free_list),
        }


# ----------------------------------------------------------------------- #
#  SGLang Router 集成
# ----------------------------------------------------------------------- #


class GDRTransportRouter:
    """
    SGLang Router GDR 扩展

    在 SGLang 的传输路由逻辑中增加 GDR 选项:
    1. 检测 GDR 可用性
    2. 跨节点传输时优先 GDR (如果可用)
    3. 同节点回退 CUDA IPC / SHM

    SGLang 原有 Router (sglang_omni/comm/router.py) 逻辑:
      if same_node:
          if cuda_tensor: transport = cuda_ipc
          else: transport = shm
      else:
          transport = mooncake

    GDR 扩展后:
      if same_node:
          if cuda_tensor: transport = cuda_ipc
          else: transport = shm
      else:
          if gdr_available and cuda_tensor:
              transport = gdr           # ← 优先 GDR
          elif gdr_available and via_3fs:
              transport = gdr_3fs       # ← 通过 3FS 中转
          else:
              transport = mooncake      # ← 回退
    """

    def __init__(self):
        self._gdr_available = False
        self._gdr_backend: Optional[GDRRelayBackend] = None
        self._mooncake_available = False
        self._3fs_available = False

    def initialize(self) -> bool:
        """检测可用的传输后端"""
        detector = GDRDetector()
        report = detector.detect()
        self._gdr_available = report["gdr_available"]

        if self._gdr_available:
            self._gdr_backend = GDRRelayBackend()
            self._gdr_available = self._gdr_backend.initialize()

        # 检测 3FS 是否可用 (通过 FUSE 挂载点)
        self._3fs_available = os.path.ismount("/3fs/stage")

        # Mooncake 检测 (通过 sglang_omni 导入)
        try:
            import sglang_omni  # noqa: F401
            self._mooncake_available = True
        except ImportError:
            self._mooncake_available = False

        logger.info(
            f"Transport Router 初始化: "
            f"GDR={'yes' if self._gdr_available else 'no'}, "
            f"3FS={'yes' if self._3fs_available else 'no'}, "
            f"Mooncake={'yes' if self._mooncake_available else 'no'}"
        )
        return True

    def select_transport(
        self,
        src_rank: int,
        dst_rank: int,
        is_cuda_tensor: bool = True,
        via_3fs: bool = False,
    ) -> GDRTransportType:
        """
        选择最佳传输方式。

        Args:
            src_rank:       源 rank
            dst_rank:       目标 rank
            is_cuda_tensor: 数据是否在 GPU 上
            via_3fs:        是否通过 3FS 中转

        Returns:
            GDRTransportType
        """
        same_node = (src_rank == dst_rank) or self._is_same_node(src_rank, dst_rank)

        if same_node:
            if is_cuda_tensor:
                return GDRTransportType.CUDA_IPC
            else:
                return GDRTransportType.SHM
        else:
            # 跨节点
            if self._gdr_available and is_cuda_tensor and not via_3fs:
                return GDRTransportType.GDR
            elif self._gdr_available and via_3fs and self._3fs_available:
                return GDRTransportType.GDR_3FS
            elif self._mooncake_available:
                return GDRTransportType.MOONCAKE
            else:
                logger.warning("无可用跨节点传输后端")
                return GDRTransportType.GDR  # 尝试 GDR

    def _is_same_node(self, rank_a: int, rank_b: int) -> bool:
        """判断两个 rank 是否在同一节点 (通过 hostname 或 IP)"""
        # 实际实现需要访问 SGLang 的 rank → host 映射
        # 简化: 假设 rank 0-7 在节点 0, rank 8-15 在节点 1
        return (rank_a // 8) == (rank_b // 8)

    def get_backend(
        self,
        transport_type: GDRTransportType,
    ) -> Optional[Any]:
        """获取指定类型的传输后端"""
        if transport_type == GDRTransportType.GDR:
            return self._gdr_backend
        elif transport_type == GDRTransportType.GDR_3FS:
            # 通过 3FS 中转的 GDR 后端
            return self._gdr_backend  # 复用 GDR 后端
        return None


# 模块级单例
_router: Optional[GDRTransportRouter] = None


def get_router() -> GDRTransportRouter:
    """获取全局 Router 实例"""
    global _router
    if _router is None:
        _router = GDRTransportRouter()
        _router.initialize()
    return _router
