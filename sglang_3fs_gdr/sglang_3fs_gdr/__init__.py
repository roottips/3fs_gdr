"""
SGLang + 3FS + GPUDirect RDMA (GDR) 集成模块

本模块提供 SGLang 推理框架基于 3FS 分布式文件系统
并通过 GPUDirect RDMA 实现零拷贝 GPU-NIC 数据传输的能力。

核心组件:
  - gdr_core:    GDR 检测、GPU 内存管理与 RDMA 注册
  - hf3fs_bridge: 3FS USRBIO 扩展与 GDR 桥接
  - hicache:     SGLang HiCache GDR 存储后端
  - comm:        SGLang 通信层 GDR 传输适配
  - utils:       配置与基准测试工具
"""

__version__ = "0.1.0"
__all__ = ["gdr_core", "hf3fs_bridge", "hicache", "comm", "utils"]
