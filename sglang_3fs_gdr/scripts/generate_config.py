#!/usr/bin/env python3
"""
3FS + SGLang + GDR 配置生成器

交互式生成所有配置文件:
  1. GDR 全局配置 (gdr_config.yaml)
  2. 3FS FUSE 客户端配置 (hf3fs_fuse.toml)
  3. SGLang systemd 服务文件 (sglang-gdr.service)
  4. SGLang 启动脚本 (launch_sglang_gdr.sh)
  5. 环境变量配置 (.env.gdr)

用法:
    python3 generate_config.py [选项]

选项:
    --gpu <id>           GPU 设备 ID
    --ib <dev>           IB 设备名
    --mgmtd <addr>       3FS mgmtd 地址
    --model <path>       模型路径
    --tp <n>             Tensor 并行度
    --port <n>           SGLang 端口
    --output-dir <path>  输出目录
    --non-interactive    非交互模式 (使用默认值)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ConfigParams:
    """配置参数"""
    gpu_id: int = 0
    ib_dev: str = "mlx5_0"
    pool_size_gb: int = 4
    cluster_id: str = "stage"
    mountpoint: str = "/3fs/stage"
    mgmtd_addr: str = "RDMA://192.168.1.1:8000"
    token_file: str = "/opt/3fs/etc/token.txt"
    model_path: str = "/3fs/stage/models/DeepSeek-R1"
    tp: int = 8
    host: str = "0.0.0.0"
    port: int = 10000
    hicache_ratio: float = 2.0
    hicache_io_backend: str = "gdr"
    hicache_write_policy: str = "write_through"
    io_depth: int = 64
    batch_size: int = 16
    output_dir: str = "/etc/sglang-gdr"


def prompt_input(prompt: str, default: str) -> str:
    """交互式输入"""
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def prompt_int(prompt: str, default: int) -> int:
    val = prompt_input(prompt, str(default))
    try:
        return int(val)
    except ValueError:
        print(f"  无效输入，使用默认值: {default}")
        return default


def collect_params(args: argparse.Namespace) -> ConfigParams:
    """收集配置参数"""
    if args.non_interactive:
        return ConfigParams(
            gpu_id=args.gpu,
            ib_dev=args.ib,
            mgmtd_addr=args.mgmtd,
            model_path=args.model,
            tp=args.tp,
            port=args.port,
            output_dir=args.output_dir,
        )

    print("\n=== GDR 配置参数收集 ===\n")

    params = ConfigParams()
    params.gpu_id = prompt_int("GPU 设备 ID", args.gpu)
    params.ib_dev = prompt_input("IB 设备名", args.ib)
    params.pool_size_gb = prompt_int("GPU 内存池大小 (GB)", 4)
    params.cluster_id = prompt_input("3FS 集群 ID", "stage")
    params.mountpoint = prompt_input("3FS 挂载点", "/3fs/stage")
    params.mgmtd_addr = prompt_input("3FS mgmtd 地址", args.mgmtd)
    params.token_file = prompt_input("3FS token 文件路径", "/opt/3fs/etc/token.txt")
    params.model_path = prompt_input("模型路径", args.model)
    params.tp = prompt_int("Tensor 并行度", args.tp)
    params.host = prompt_input("监听地址", "0.0.0.0")
    params.port = prompt_int("SGLang 端口", args.port)
    params.hicache_io_backend = prompt_input("HiCache I/O 后端", "gdr")
    params.hicache_write_policy = prompt_input("HiCache 写策略", "write_through")
    params.io_depth = prompt_int("I/O 深度", 64)
    params.batch_size = prompt_int("批量大小", 16)
    params.output_dir = prompt_input("配置输出目录", args.output_dir)

    return params


# ======================================================================= #
#  配置文件生成
# ======================================================================= #


def gen_gdr_config_yaml(p: ConfigParams) -> str:
    """生成 GDR 配置 YAML"""
    return f"""# SGLang + 3FS + GDR 配置文件
# 由 generate_config.py 生成
# 日期: $(date)

# ============================================================
# GDR 核心配置
# ============================================================
enable_gdr: true
gpu_id: {p.gpu_id}
ib_dev: "{p.ib_dev}"
pool_size_gb: {p.pool_size_gb}

# ============================================================
# 3FS 配置
# ============================================================
cluster_id: "{p.cluster_id}"
mountpoint: "{p.mountpoint}"
mgmtd_address: "{p.mgmtd_addr}"
token_file: "{p.token_file}"

# USRBIO GDR 配置
iov_path: "{p.mountpoint}/.iov_gpu{p.gpu_id}"
ior_path: "{p.mountpoint}/.ior_gpu{p.gpu_id}"
io_depth: {p.io_depth}
batch_size: {p.batch_size}

# FUSE 挂载选项
fuse_mount_options: "rdma_port=1,rdma_timeout=30000"

# ============================================================
# HiCache 配置
# ============================================================
hicache_storage_prefix: "{p.mountpoint}/kvcache"
hicache_io_backend: "{p.hicache_io_backend}"
hicache_write_policy: "{p.hicache_write_policy}"
hicache_prefetch_depth: 4

# ============================================================
# 通信层配置
# ============================================================
comm_enable_gdr_relay: true
comm_cq_depth: 256
comm_qp_depth: 128

# ============================================================
# SGLang 服务配置
# ============================================================
sglang_model_path: "{p.model_path}"
sglang_tp: {p.tp}
sglang_host: "{p.host}"
sglang_port: {p.port}

# ============================================================
# 性能调优
# ============================================================
enable_relaxed_ordering: true
enable_batch: true
poll_interval_us: 1

# ============================================================
# 日志
# ============================================================
log_level: INFO
log_file: /var/log/sglang_gdr.log
"""


def gen_hf3fs_fuse_toml(p: ConfigParams) -> str:
    """生成 3FS FUSE 配置 TOML"""
    return f"""# 3FS FUSE 客户端配置
# 由 generate_config.py 生成

cluster_id = "{p.cluster_id}"
mountpoint = "{p.mountpoint}"
token_file = "{p.token_file}"

[mgmtd_client]
mgmtd_server_addresses = ["{p.mgmtd_addr}"]

[fuse_client]
# RDMA 配置
rdma_timeout_ms = 30000
rdma_max_msg_size = 1048576

# I/O worker 线程数
io_worker_count = 16

# 批量 I/O 深度
batch_io_depth = {p.io_depth}

# USRBIO 配置
[usrbio]
io_depth = {p.io_depth}
batch_size = {p.batch_size}
"""


def gen_systemd_service(p: ConfigParams) -> str:
    """生成 systemd 服务文件"""
    return f"""[Unit]
Description=SGLang Inference Server with GPUDirect RDMA
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/sglang

# 环境变量
Environment=SGLANG_GDR_PLUGIN=1
Environment=GDR_CONFIG_PATH={p.output_dir}/gdr_config.yaml
Environment=CUDA_VISIBLE_DEVICES={p.gpu_id}

# 资源限制 (GDR 需要锁定内存)
LimitMEMLOCK=infinity
LimitNPROC=infinity
LimitNOFILE=65536

ExecStart=/usr/bin/python3 -m sglang.launch_server \\
    --model-path {p.model_path} \\
    --tp {p.tp} \\
    --host {p.host} \\
    --port {p.port} \\
    --enable-hierarchical-cache \\
    --hicache-ratio {p.hicache_ratio} \\
    --hicache-mem-layout page_first_direct \\
    --hicache-io-backend {p.hicache_io_backend} \\
    --hicache-storage-backend gdr_3fs \\
    --hicache-write-policy {p.hicache_write_policy} \\
    --hicache-storage-prefetch-policy wait_complete

Restart=on-failure
RestartSec=10
TimeoutStopSec=30

# 日志
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sglang-gdr

[Install]
WantedBy=multi-user.target
"""


def gen_launch_script(p: ConfigParams) -> str:
    """生成 SGLang 启动脚本"""
    return f"""#!/bin/bash
# SGLang + 3FS + GDR 启动脚本
# 由 generate_config.py 生成

set -euo pipefail

# 配置
export SGLANG_GDR_PLUGIN=1
export GDR_CONFIG_PATH="{p.output_dir}/gdr_config.yaml"
export CUDA_VISIBLE_DEVICES="{p.gpu_id}"

# 加载 nvidia-peermem (如果未加载)
if ! lsmod | grep -q "nvidia_peermem\\|nv_peer_mem"; then
    echo "加载 nvidia-peermem..."
    sudo modprobe nvidia-peermem || sudo modprobe nv_peer_mem
fi

# 确认 3FS 已挂载
if ! mountpoint -q "{p.mountpoint}" 2>/dev/null; then
    echo "3FS 未挂载，尝试挂载..."
    sudo hf3fs_fuse_main --config {p.output_dir}/hf3fs_fuse.toml &
    sleep 5
fi

# 启动 SGLang
echo "启动 SGLang with GDR..."
python3 -m sglang.launch_server \\
    --model-path {p.model_path} \\
    --tp {p.tp} \\
    --host {p.host} \\
    --port {p.port} \\
    --enable-hierarchical-cache \\
    --hicache-ratio {p.hicache_ratio} \\
    --hicache-mem-layout page_first_direct \\
    --hicache-io-backend {p.hicache_io_backend} \\
    --hicache-storage-backend gdr_3fs \\
    --hicache-write-policy {p.hicache_write_policy} \\
    --hicache-storage-prefetch-policy wait_complete
"""


def gen_env_file(p: ConfigParams) -> str:
    """生成环境变量文件"""
    return f"""# GDR 环境变量配置
# 加载方式: source {p.output_dir}/.env.gdr

export SGLANG_GDR_PLUGIN=1
export GDR_CONFIG_PATH="{p.output_dir}/gdr_config.yaml"
export CUDA_VISIBLE_DEVICES="{p.gpu_id}"

# RDMA 配置
export RDMA_DEVICE="{p.ib_dev}"
export IB_DEV="{p.ib_dev}"

# 3FS 配置
export HF3FS_CLUSTER_ID="{p.cluster_id}"
export HF3FS_MOUNTPOINT="{p.mountpoint}"
export HF3FS_MGMTD="{p.mgmtd_addr}"

# 性能调优
export RDMAV_FORK_SAFE=1
export RDMAV_HUGEPAGES_SAFE=1

# 调试 (生产环境设置为 0)
export GDR_DEBUG=0
export GDR_LOG_LEVEL=INFO
"""


# ======================================================================= #
#  主函数
# ======================================================================= #


def main():
    parser = argparse.ArgumentParser(
        description="3FS + SGLang + GDR 配置生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU 设备 ID")
    parser.add_argument("--ib", default="mlx5_0", help="IB 设备名")
    parser.add_argument("--mgmtd", default="RDMA://192.168.1.1:8000",
                        help="3FS mgmtd 地址")
    parser.add_argument("--model", default="/3fs/stage/models/DeepSeek-R1",
                        help="模型路径")
    parser.add_argument("--tp", type=int, default=8, help="Tensor 并行度")
    parser.add_argument("--port", type=int, default=10000, help="SGLang 端口")
    parser.add_argument("--output-dir", default="/etc/sglang-gdr",
                        help="配置输出目录")
    parser.add_argument("--non-interactive", action="store_true",
                        help="非交互模式")
    args = parser.parse_args()

    params = collect_params(args)

    print(f"\n=== 生成配置文件到 {params.output_dir} ===\n")

    # 创建输出目录
    Path(params.output_dir).mkdir(parents=True, exist_ok=True)

    # 生成所有配置文件
    files = {
        "gdr_config.yaml": gen_gdr_config_yaml(params),
        "hf3fs_fuse.toml": gen_hf3fs_fuse_toml(params),
        "sglang-gdr.service": gen_systemd_service(params),
        "launch_sglang_gdr.sh": gen_launch_script(params),
        ".env.gdr": gen_env_file(params),
    }

    for filename, content in files.items():
        filepath = os.path.join(params.output_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)
        if filename.endswith(".sh"):
            os.chmod(filepath, 0o755)
        print(f"  [OK] {filepath}")

    print(f"\n=== 配置文件生成完成 ===")
    print(f"\n使用方式:")
    print(f"  1. 挂载 3FS: sudo hf3fs_fuse_main --config {params.output_dir}/hf3fs_fuse.toml")
    print(f"  2. 启动 SGLang:")
    print(f"     方式 A (脚本): bash {params.output_dir}/launch_sglang_gdr.sh")
    print(f"     方式 B (systemd):")
    print(f"       sudo cp {params.output_dir}/sglang-gdr.service /etc/systemd/system/")
    print(f"       sudo systemctl daemon-reload")
    print(f"       sudo systemctl start sglang-gdr")
    print(f"  3. 环境变量: source {params.output_dir}/.env.gdr")


if __name__ == "__main__":
    main()
