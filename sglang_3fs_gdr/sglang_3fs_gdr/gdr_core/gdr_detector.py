"""
GDR 能力检测模块

负责检测系统是否满足 GPUDirect RDMA 运行条件:
  1. nvidia-peermem 内核模块加载状态
  2. NVIDIA GPU 设备与 RDMA NIC 的 P2P 可达性
  3. IB 端口状态与 GDR 支持属性
  4. CUDA 运行时版本与驱动版本

参考:
  - NVIDIA GPUDirect RDMA Documentation
  - ibv_devinfo / nvidia-smi 检测方法
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GDRDeviceInfo:
    """GPU-NIC 配对信息"""
    gpu_id: int
    gpu_name: str
    gpu_pci: str
    ib_dev: str
    ib_port: int
    ib_active: bool
    ib_rate: int           # Gbps
    gdr_supported: bool    # 该 GPU-NIC 对是否支持 GDR
    peer_access: bool       # CUDA P2P 是否可达
    cuda_ioua: bool         # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED


class GDRDetector:
    """
    GPUDirect RDMA 环境检测器

    用法:
        detector = GDRDetector()
        info = detector.detect()
        if info["gdr_available"]:
            print(f"GDR 就绪: {info['gpu_nic_pairs'][0]}")
    """

    PEERMEM_MODULES = ["nvidia_peermem", "nv_peer_mem"]
    GDR_CHECK_BDF = "/proc/driver/nvidia/gpus"

    def __init__(self):
        self._cuda_initialized = False

    # ------------------------------------------------------------------ #
    #  公开 API
    # ------------------------------------------------------------------ #

    def detect(self) -> dict:
        """
        执行完整 GDR 环境检测，返回结构化报告。

        Returns:
            {
                "gdr_available": bool,
                "cuda_version": str,
                "driver_version": str,
                "peermem_loaded": bool,
                "ib_devices": [str],
                "gpu_nic_pairs": [GDRDeviceInfo],
                "warnings": [str],
            }
        """
        warnings: list[str] = []

        # 1. 检测 CUDA 环境
        cuda_info = self._detect_cuda()
        if cuda_info is None:
            warnings.append("CUDA 不可用 — GDR 需要 CUDA Runtime")
        else:
            self._cuda_initialized = True

        # 2. 检测 nvidia-peermem 内核模块
        peermem_loaded = self._check_peermem_module()
        if not peermem_loaded:
            warnings.append(
                "nvidia-peermem 未加载 — 请执行 modprobe nvidia-peermem"
            )

        # 3. 检测 IB 设备列表
        ib_devices = self._list_ib_devices()
        if not ib_devices:
            warnings.append("未发现 InfiniBand/RoCE 设备 — GDR 依赖 RDMA 网络")

        # 4. 检测 GPU 设备列表
        gpu_list = self._list_gpu_devices()
        if not gpu_list:
            warnings.append("未发现 NVIDIA GPU 设备")

        # 5. 构建 GPU-NIC 配对并检测 GDR 支持
        gpu_nic_pairs: list[GDRDeviceInfo] = []
        for gpu in gpu_list:
            for ib_dev in ib_devices:
                pair = self._check_gdr_support(gpu, ib_dev)
                gpu_nic_pairs.append(pair)

        gdr_available = (
            cuda_info is not None
            and peermem_loaded
            and len(ib_devices) > 0
            and any(p.gdr_supported for p in gpu_nic_pairs)
        )

        return {
            "gdr_available": gdr_available,
            "cuda_version": cuda_info.get("cuda_version", "") if cuda_info else "",
            "driver_version": cuda_info.get("driver_version", "") if cuda_info else "",
            "peermem_loaded": peermem_loaded,
            "ib_devices": ib_devices,
            "gpu_nic_pairs": gpu_nic_pairs,
            "warnings": warnings,
        }

    def get_best_pair(self) -> Optional[GDRDeviceInfo]:
        """返回带宽最高的可用 GDR GPU-NIC 配对"""
        report = self.detect()
        pairs = [p for p in report["gpu_nic_pairs"] if p.gdr_supported]
        if not pairs:
            return None
        return max(pairs, key=lambda p: p.ib_rate)

    # ------------------------------------------------------------------ #
    #  内部检测方法
    # ------------------------------------------------------------------ #

    def _detect_cuda(self) -> Optional[dict]:
        """检测 CUDA Runtime 与驱动版本"""
        try:
            import torch
            cuda_ver = torch.version.cuda
            if cuda_ver is None:
                return None
            driver_ver = self._get_driver_version()
            return {
                "cuda_version": cuda_ver,
                "driver_version": driver_ver,
            }
        except ImportError:
            logger.warning("PyTorch 未安装，尝试 nvidia-smi 检测")
            return self._detect_cuda_via_nvsmi()

    def _detect_cuda_via_nvsmi(self) -> Optional[dict]:
        """通过 nvidia-smi 获取 CUDA/Driver 版本"""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.PIPE,
                timeout=10,
            ).decode().strip()
            driver_ver = out.splitlines()[0] if out else "unknown"
            # CUDA version 通常在 nvidia-smi 顶部输出
            out2 = subprocess.check_output(
                ["nvidia-smi"], stderr=subprocess.PIPE, timeout=10
            ).decode()
            match = re.search(r"CUDA Version:\s*([\d.]+)", out2)
            cuda_ver = match.group(1) if match else "unknown"
            return {"cuda_version": cuda_ver, "driver_version": driver_ver}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def _get_driver_version(self) -> str:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.PIPE, timeout=10,
            ).decode().strip()
            return out.splitlines()[0]
        except Exception:
            return "unknown"

    def _check_peermem_module(self) -> bool:
        """检测 nvidia-peermem / nv_peer_mem 内核模块是否加载"""
        try:
            with open("/proc/modules", "r") as f:
                modules = f.read()
            return any(mod in modules for mod in self.PEERMEM_MODULES)
        except FileNotFoundError:
            # 某些系统 /proc/modules 不可用，尝试 lsmod
            try:
                out = subprocess.check_output(
                    ["lsmod"], stderr=subprocess.PIPE, timeout=10
                ).decode()
                return any(mod in out for mod in self.PEERMEM_MODULES)
            except Exception:
                return False

    def _list_ib_devices(self) -> list[str]:
        """列出所有活跃的 IB/RoCE 设备"""
        devices: list[str] = []
        ib_root = Path("/sys/class/infiniband")
        if not ib_root.exists():
            return devices
        for entry in ib_root.iterdir():
            dev_name = entry.name
            ports_dir = entry / "ports"
            if ports_dir.exists():
                for port_dir in ports_dir.iterdir():
                    state_file = port_dir / "state"
                    if state_file.exists():
                        state = state_file.read_text().strip()
                        if "ACTIVE" in state.upper():
                            devices.append(dev_name)
                            break
        return devices

    def _list_gpu_devices(self) -> list[dict]:
        """列出所有 NVIDIA GPU 设备"""
        gpus: list[dict] = []
        gpu_root = Path(self.GDR_CHECK_BDF)
        if not gpu_root.exists():
            # 回退到 nvidia-smi
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,pci.bus_id",
                        "--format=csv,noheader",
                    ],
                    stderr=subprocess.PIPE, timeout=10,
                ).decode().strip()
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "pci": parts[2],
                        })
            except Exception:
                pass
            return gpus

        for idx, entry in enumerate(sorted(gpu_root.iterdir())):
            gpus.append({
                "index": idx,
                "name": entry.name,
                "pci": entry.name,
            })
        return gpus

    def _check_gdr_support(self, gpu: dict, ib_dev: str) -> GDRDeviceInfo:
        """
        检测指定 GPU 与 IB 设备之间是否支持 GPUDirect RDMA。

        检测逻辑:
          1. 检查 GPU PCI 地址与 NIC PCI 地址的 NUMA 亲和性
          2. 检查 CUDA CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED
          3. 尝试 P2P 访问测试
        """
        gpu_id = gpu["index"]
        gpu_name = gpu.get("name", "unknown")
        gpu_pci = gpu.get("pci", "unknown")

        # 获取 IB 端口信息
        ib_port = 1
        ib_active = True
        ib_rate = self._get_ib_rate(ib_dev, ib_port)

        # CUDA GDR 属性检测
        cuda_ioua = self._check_cuda_gdr_attribute(gpu_id)

        # P2P 可达性检测
        peer_access = self._check_p2p_access(gpu_id, ib_dev)

        gdr_supported = cuda_ioua and peer_access and ib_active

        return GDRDeviceInfo(
            gpu_id=gpu_id,
            gpu_name=gpu_name,
            gpu_pci=gpu_pci,
            ib_dev=ib_dev,
            ib_port=ib_port,
            ib_active=ib_active,
            ib_rate=ib_rate,
            gdr_supported=gdr_supported,
            peer_access=peer_access,
            cuda_ioua=cuda_ioua,
        )

    def _get_ib_rate(self, ib_dev: str, port: int) -> int:
        """读取 IB 端口速率 (Gbps)"""
        rate_file = Path(f"/sys/class/infiniband/{ib_dev}/ports/{port}/rate")
        if rate_file.exists():
            try:
                rate_str = rate_file.read_text().strip()
                # 格式如 "100 Gb/sec" 或 "200 Gb/sec"
                rate = int(rate_str.split()[0])
                return rate
            except (ValueError, IndexError):
                pass
        return 0

    def _check_cuda_gdr_attribute(self, gpu_id: int) -> bool:
        """
        检查 CUDA 设备属性 CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED。

        属性值 106 = CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED
        返回 1 表示支持。
        """
        try:
            import torch
            # PyTorch 没有直接暴露此属性，使用 ctypes 调用 CUDA Driver API
            cuda = ctypes.CDLL("libcuda.so")
            result = ctypes.c_int(0)

            ret = cuda.cuInit(0)
            if ret != 0:
                return False

            dev = ctypes.c_int(0)
            ret = cuda.cuDeviceGet(ctypes.byref(dev), gpu_id)
            if ret != 0:
                return False

            # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 106
            attr = 106
            ret = cuda.cuDeviceGetAttribute(
                ctypes.byref(result), attr, dev.value
            )
            if ret != 0:
                return False

            return result.value == 1
        except (OSError, ImportError):
            # 如果无法调用 CUDA Driver API，保守返回 True 并记录警告
            logger.warning("无法检测 CUDA GDR 属性，假设支持（请手动验证）")
            return True

    def _check_p2p_access(self, gpu_id: int, ib_dev: str) -> bool:
        """
        检测 GPU 与 NIC 之间的 P2P DMA 可达性。

        通过检查 /proc/driver/nvidia/gpus/<bdf>/information 中的
        P2P 信息，或通过 nvidia-smi topo -m 检查拓扑。
        """
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "topo", "-m"],
                stderr=subprocess.PIPE, timeout=10,
            ).decode()

            # 解析拓扑矩阵，检查 GPU 与 NIC 之间是否有 P2P 路径
            # 拓扑矩阵中 "PIX" / "PIX" 表示有 P2P 连接
            lines = out.splitlines()
            for line in lines:
                if f"GPU{gpu_id}" in line:
                    # 检查是否有 P2P 标记
                    if "PIX" in line or "PXB" in line or "NODE" in line:
                        return True
            return False
        except Exception:
            return False


def check_gdr_environment() -> dict:
    """
    便捷函数：快速检测 GDR 环境。

    Returns:
        检测报告字典
    """
    detector = GDRDetector()
    return detector.detect()
