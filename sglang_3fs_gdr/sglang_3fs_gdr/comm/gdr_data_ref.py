"""
GDR DataRef 扩展

扩展 SGLang 的 DataRef 以支持 GDR 传输类型。
DataRef 是 SGLang 通信层中跨阶段数据传递的核心结构。

SGLang 通信架构:
  sglang_omni/comm/data_ref.py:  DataRef 结构体
  sglang_omni/comm/router.py:    传输路由选择
  sglang_omni/relay/base.py:     后端接口
  sglang_omni/relay/mooncake.py: Mooncake RDMA 后端

GDR 扩展:
  - 新增 GDRTransportType 传输类型
  - GDRDataRef 包含 GPU MR 信息 (rkey, gpu_addr)
  - Router 在 GDR 可用时优先选择 GDR 传输
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Any


class GDRTransportType(enum.Enum):
    """
    GDR 传输类型枚举。

    SGLang 原有传输类型:
      - local_object: 同进程
      - cuda_ipc:     CUDA IPC (同节点)
      - shm:          共享内存 (同节点)
      - mooncake:     Mooncake RDMA (跨节点)

    新增:
      - gdr:          GPUDirect RDMA (跨节点, 零拷贝 GPU 直通)
      - gdr_3fs:      GDR via 3FS (通过 3FS 中转的 GDR)
    """
    LOCAL = "local_object"
    CUDA_IPC = "cuda_ipc"
    SHM = "shm"
    MOONCAKE = "mooncake"
    GDR = "gdr"
    GDR_3FS = "gdr_3fs"


@dataclass
class GDRDataRef:
    """
    GDR 数据引用

    封装通过 GDR 传输的数据引用信息，
    包含 GPU 内存的 RDMA 访问信息。

    当 SGLang 的 Router 选择 GDR 传输时，
    会创建此 DataRef 传递给接收端。

    接收端使用 rkey 和 gpu_addr 直接通过 RDMA 访问发送端的 GPU 内存。

    字段:
      object_id:       全局唯一对象 ID
      data_kind:       数据类型 (kv_cache, activation, weight)
      transport:        传输类型
      gpu_addr:        GPU 显存物理地址 (远端可访问)
      size:             数据大小 (字节)
      rkey:             RDMA Remote Key
      lkey:             RDMA Local Key (仅本地使用)
      gpu_id:           GPU 设备 ID
      ib_dev:           IB 设备名
      src_rank:         发送端 rank
      dst_rank:         接收端 rank
      tensor_meta:      张量元信息 (shape, dtype)
    """
    object_id: str
    data_kind: str                      # kv_cache | activation | weight
    transport: GDRTransportType = GDRTransportType.GDR
    gpu_addr: int = 0                   # 远端 GPU 内存地址
    size: int = 0                       # 数据大小
    rkey: int = 0                       # RDMA Remote Key
    lkey: int = 0                       # RDMA Local Key (仅本地)
    gpu_id: int = 0                     # GPU 设备 ID
    ib_dev: str = ""                    # IB 设备名
    src_rank: int = -1                  # 发送端
    dst_rank: int = -1                  # 接收端
    tensor_meta: dict = field(default_factory=dict)  # shape, dtype, etc.
    ready: bool = False                # 数据是否就绪

    def to_dict(self) -> dict:
        """序列化为字典 (用于网络传输)"""
        return {
            "object_id": self.object_id,
            "data_kind": self.data_kind,
            "transport": self.transport.value,
            "gpu_addr": self.gpu_addr,
            "size": self.size,
            "rkey": self.rkey,
            "gpu_id": self.gpu_id,
            "ib_dev": self.ib_dev,
            "src_rank": self.src_rank,
            "dst_rank": self.dst_rank,
            "tensor_meta": self.tensor_meta,
            "ready": self.ready,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GDRDataRef":
        """从字典反序列化"""
        return cls(
            object_id=d["object_id"],
            data_kind=d["data_kind"],
            transport=GDRTransportType(d.get("transport", "gdr")),
            gpu_addr=d.get("gpu_addr", 0),
            size=d.get("size", 0),
            rkey=d.get("rkey", 0),
            gpu_id=d.get("gpu_id", 0),
            ib_dev=d.get("ib_dev", ""),
            src_rank=d.get("src_rank", -1),
            dst_rank=d.get("dst_rank", -1),
            tensor_meta=d.get("tensor_meta", {}),
            ready=d.get("ready", False),
        )

    def is_gdr(self) -> bool:
        """是否使用 GDR 传输"""
        return self.transport in (
            GDRTransportType.GDR,
            GDRTransportType.GDR_3FS,
        )
