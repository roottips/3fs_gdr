"""
GDR 配置模块

集中管理 SGLang + 3FS + GDR 的所有配置参数。
支持 YAML/TOML 配置文件和命令行参数。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any


@dataclass
class GDRConfig:
    """SGLang + 3FS + GDR 全局配置"""

    # GDR 核心
    enable_gdr: bool = True
    gpu_id: int = 0
    ib_dev: str = "mlx5_0"
    pool_size_gb: int = 4

    # 3FS
    cluster_id: str = "stage"
    mountpoint: str = "/3fs/stage"
    mgmtd_address: str = "RDMA://192.168.1.1:8000"
    token_file: str = "/opt/3fs/etc/token.txt"
    iov_path: str = "/3fs/stage/.iov_gpu"
    ior_path: str = "/3fs/stage/.ior_gpu"
    io_depth: int = 64
    batch_size: int = 16

    # HiCache
    hicache_storage_prefix: str = "/3fs/stage/kvcache"
    hicache_io_backend: str = "gdr"         # gdr | gdr_async | direct
    hicache_write_policy: str = "write_through"
    hicache_prefetch_depth: int = 4

    # 通信层
    comm_enable_gdr_relay: bool = True
    comm_cq_depth: int = 256
    comm_qp_depth: int = 128

    # SGLang 集成
    sglang_model_path: str = "/3fs/stage/models/DeepSeek-R1"
    sglang_tp: int = 8
    sglang_host: str = "0.0.0.0"
    sglang_port: int = 10000

    # 性能调优
    enable_relaxed_ordering: bool = True
    enable_batch: bool = True
    poll_interval_us: int = 1

    # 日志
    log_level: str = "INFO"
    log_file: str = "/var/log/sglang_gdr.log"

    @classmethod
    def from_dict(cls, d: dict) -> "GDRConfig":
        """从字典创建配置"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "GDRConfig":
        """从 YAML 文件加载配置"""
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except ImportError:
            raise RuntimeError("PyYAML 未安装")
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件不存在: {path}")

    def to_sglang_args(self) -> list[str]:
        """转换为 SGLang 启动命令行参数"""
        args = [
            "python3", "-m", "sglang.launch_server",
            "--model-path", self.sglang_model_path,
            "--tp", str(self.sglang_tp),
            "--host", self.sglang_host,
            "--port", str(self.sglang_port),
            "--enable-hierarchical-cache",
            "--hicache-ratio", "2",
            "--hicache-mem-layout", "page_first_direct",
            "--hicache-io-backend", self.hicache_io_backend,
            "--hicache-storage-backend", "gdr_3fs",
            "--hicache-write-policy", self.hicache_write_policy,
            "--hicache-storage-prefetch-policy", "wait_complete",
        ]
        if self.hicache_prefetch_depth > 0:
            args.extend([
                "--hicache-storage-prefetch-depth",
                str(self.hicache_prefetch_depth),
            ])
        return args

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


def load_config(config_path: Optional[str] = None) -> GDRConfig:
    """
    加载 GDR 配置。

    优先级: 命令行参数 > 环境变量 > 配置文件 > 默认值
    """
    config = GDRConfig()

    # 环境变量覆盖
    if os.environ.get("GDR_ENABLE"):
        config.enable_gdr = os.environ["GDR_ENABLE"].lower() in ("1", "true", "yes")
    if os.environ.get("GDR_GPU_ID"):
        config.gpu_id = int(os.environ["GDR_GPU_ID"])
    if os.environ.get("GDR_IB_DEV"):
        config.ib_dev = os.environ["GDR_IB_DEV"]
    if os.environ.get("GDR_POOL_GB"):
        config.pool_size_gb = int(os.environ["GDR_POOL_GB"])
    if os.environ.get("HF3FS_MOUNTPOINT"):
        config.mountpoint = os.environ["HF3FS_MOUNTPOINT"]
    if os.environ.get("HF3FS_MGMTD"):
        config.mgmtd_address = os.environ["HF3FS_MGMTD"]

    # 配置文件覆盖
    if config_path:
        path = Path(config_path)
        if path.suffix in (".yaml", ".yml"):
            file_config = GDRConfig.from_yaml(str(path))
        elif path.suffix == ".toml":
            try:
                import tomllib
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                file_config = GDRConfig.from_dict(data)
            except ImportError:
                raise RuntimeError("Python 3.11+ 或 tomli 库未安装")
        else:
            raise ValueError(f"不支持的配置文件格式: {path.suffix}")

        # 合并 (文件配置覆盖环境变量)
        for key, val in file_config.to_dict().items():
            if val is not None:
                setattr(config, key, val)

    return config


# 默认配置文件模板
DEFAULT_CONFIG_YAML = """\
# SGLang + 3FS + GDR 配置文件

# GDR 核心
enable_gdr: true
gpu_id: 0
ib_dev: mlx5_0
pool_size_gb: 4

# 3FS
cluster_id: stage
mountpoint: /3fs/stage
mgmtd_address: RDMA://192.168.1.1:8000
token_file: /opt/3fs/etc/token.txt
iov_path: /3fs/stage/.iov_gpu
ior_path: /3fs/stage/.ior_gpu
io_depth: 64
batch_size: 16

# HiCache
hicache_storage_prefix: /3fs/stage/kvcache
hicache_io_backend: gdr
hicache_write_policy: write_through
hicache_prefetch_depth: 4

# 通信层
comm_enable_gdr_relay: true
comm_cq_depth: 256
comm_qp_depth: 128

# SGLang
sglang_model_path: /3fs/stage/models/DeepSeek-R1
sglang_tp: 8
sglang_host: 0.0.0.0
sglang_port: 10000

# 性能
enable_relaxed_ordering: true
enable_batch: true
poll_interval_us: 1

# 日志
log_level: INFO
log_file: /var/log/sglang_gdr.log
"""
