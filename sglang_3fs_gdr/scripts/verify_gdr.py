#!/usr/bin/env python3
"""
GDR 环境验证工具 — 完整版

验证 GPUDirect RDMA 环境是否正确配置，输出详细检测报告。

功能:
  1. NVIDIA 驱动 / CUDA 版本检测
  2. nvidia-peermem 内核模块检测
  3. MOFED / ibverbs 库检测
  4. IB 设备端口状态检测
  5. CUDA GDR 属性检测 (CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED)
  6. GPU-NIC P2P 拓扑分析
  7. 系统资源检测 (memlock / hugepages)
  8. 3FS 挂载状态检测
  9. GDR 功能性测试 (GPU 内存注册 + RDMA 回环)
 10. JSON / HTML 报告输出

用法:
    python3 verify_gdr.py [选项]

选项:
    --gpu <id>          GPU 设备 ID (默认 0)
    --ib <dev>          IB 设备名 (默认 mlx5_0)
    --output <path>     报告输出路径 (JSON 或 HTML)
    --format <fmt>      报告格式: text (默认) / json / html
    --run-functional    执行 GDR 功能性测试 (需要 root)
    --fix-suggestions   输出修复建议
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any

# ======================================================================= #
#  数据结构
# ======================================================================= #


@dataclass
class CheckResult:
    """单项检测结果"""
    name: str
    passed: bool
    details: str = ""
    data: dict = field(default_factory=dict)
    remediation: str = ""
    severity: str = "info"  # info / warning / error


@dataclass
class GDRReport:
    """完整检测报告"""
    timestamp: str = ""
    hostname: str = ""
    gpu_id: int = 0
    ib_device: str = ""
    os_info: str = ""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "warning")

    @property
    def all_passed(self) -> bool:
        return self.failed_count == 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "hostname": self.hostname,
            "gpu_id": self.gpu_id,
            "ib_device": self.ib_device,
            "os_info": self.os_info,
            "summary": {
                "total": self.total,
                "passed": self.passed_count,
                "warnings": self.warning_count,
                "failed": self.failed_count,
                "gdr_ready": self.all_passed,
            },
            "checks": [asdict(c) for c in self.checks],
        }


# ======================================================================= #
#  工具函数
# ======================================================================= #


def run_cmd(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    """运行命令并返回 (返回码, stdout, stderr)"""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return (
            proc.returncode,
            proc.stdout.decode("utf-8", errors="replace"),
            proc.stderr.decode("utf-8", errors="replace"),
        )
    except FileNotFoundError:
        return -1, "", f"命令未找到: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -2, "", f"命令超时: {' '.join(cmd)}"
    except Exception as e:
        return -3, "", str(e)


def read_file(path: str) -> str:
    """读取文件内容"""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except Exception:
        return ""


# ======================================================================= #
#  检测函数
# ======================================================================= #


class GDRVerifier:
    """GDR 环境验证器"""

    def __init__(self, gpu_id: int = 0, ib_dev: str = "mlx5_0"):
        self.gpu_id = gpu_id
        self.ib_dev = ib_dev
        self.report = GDRReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            hostname=platform.node(),
            gpu_id=gpu_id,
            ib_device=ib_dev,
            os_info=f"{platform.system()} {platform.release()}",
        )

    def add_result(self, result: CheckResult):
        self.report.checks.append(result)
        self._print_result(result)

    def _print_result(self, r: CheckResult):
        status = "PASS" if r.passed else "FAIL" if r.severity == "error" else "WARN"
        color = "\033[92m" if r.passed else "\033[91m" if r.severity == "error" else "\033[93m"
        reset = "\033[0m"
        print(f"  {color}[{status:4}]{reset} {r.name}: {r.details}")
        if not r.passed and r.remediation:
            for line in r.remediation.split("\n"):
                print(f"         {line}")

    # ------------------------------------------------------------------ #
    #  1. NVIDIA 驱动检测
    # ------------------------------------------------------------------ #

    def check_nvidia_driver(self):
        print("\n[1] NVIDIA 驱动检测")

        if not shutil.which("nvidia-smi"):
            self.add_result(CheckResult(
                name="NVIDIA 驱动",
                passed=False,
                details="nvidia-smi 未找到",
                remediation=(
                    "安装 NVIDIA 驱动:\n"
                    "  1. 下载: https://www.nvidia.com/drivers\n"
                    "  2. 安装: sudo sh NVIDIA-Linux-x86_64-550.xx.run"
                ),
                severity="error",
            ))
            return

        rc, out, _ = run_cmd([
            "nvidia-smi", "--query-gpu=driver_version,name,memory.total,pci.bus_id",
            "--format=csv,noheader", "-i", str(self.gpu_id),
        ])

        if rc != 0:
            self.add_result(CheckResult(
                name="NVIDIA 驱动",
                passed=False,
                details=f"nvidia-smi 查询失败",
                severity="error",
            ))
            return

        parts = [p.strip() for p in out.split(",")]
        if len(parts) < 4:
            self.add_result(CheckResult(
                name="NVIDIA 驱动",
                passed=False,
                details="GPU 信息解析失败",
                severity="error",
            ))
            return

        driver_ver = parts[0]
        gpu_name = parts[1]
        mem_total = parts[2]
        pci_bus = parts[3]

        driver_major = int(driver_ver.split(".")[0]) if driver_ver[0].isdigit() else 0

        # CUDA 版本
        _, smi_out, _ = run_cmd(["nvidia-smi"])
        cuda_match = re.search(r"CUDA Version:\s*([\d.]+)", smi_out)
        cuda_ver = cuda_match.group(1) if cuda_match else "unknown"

        passed = driver_major >= 525
        severity = "error" if not passed else "info"

        self.add_result(CheckResult(
            name="NVIDIA 驱动",
            passed=passed,
            details=(
                f"{gpu_name} | Driver {driver_ver} | CUDA {cuda_ver} | "
                f"VRAM {mem_total} | PCIe {pci_bus}"
            ),
            data={
                "driver_version": driver_ver,
                "cuda_version": cuda_ver,
                "gpu_name": gpu_name,
                "memory": mem_total,
                "pci_bus_id": pci_bus,
            },
            remediation=(
                f"驱动版本 {driver_ver} < 525\n"
                "请升级到 >= 525 以获得完整 GDR 支持"
            ) if not passed else "",
            severity=severity,
        ))

    # ------------------------------------------------------------------ #
    #  2. nvidia-peermem 模块检测
    # ------------------------------------------------------------------ #

    def check_peermem(self):
        print("\n[2] nvidia-peermem 模块检测")

        # 检查 /proc/modules
        modules = read_file("/proc/modules")
        if not modules:
            # 回退到 lsmod
            rc, out, _ = run_cmd(["lsmod"])
            modules = out if rc == 0 else ""

        peermem_names = ["nvidia_peermem", "nv_peer_mem"]
        loaded = any(name in modules for name in peermem_names)

        if loaded:
            module_name = next(name for name in peermem_names if name in modules)
            self.add_result(CheckResult(
                name="nvidia-peermem",
                passed=True,
                details=f"已加载 (模块: {module_name})",
                data={"module_name": module_name},
            ))
        else:
            self.add_result(CheckResult(
                name="nvidia-peermem",
                passed=False,
                details="未加载",
                remediation=(
                    "加载 nvidia-peermem:\n"
                    "  sudo modprobe nvidia-peermem\n"
                    "  或: sudo modprobe nv_peer_mem\n"
                    "\n"
                    "如果模块不存在:\n"
                    "  1. 确认驱动 >= 525\n"
                    "  2. 从源码编译: git clone https://github.com/Mellanox/nvidia-peermem.git\n"
                    "  3. cd nvidia-peermem && make && make install"
                ),
                severity="error",
            ))

    # ------------------------------------------------------------------ #
    #  3. MOFED / ibverbs 检测
    # ------------------------------------------------------------------ #

    def check_mofed(self):
        print("\n[3] MOFED / ibverbs 检测")

        # ibv_devinfo 工具
        has_ibv_devinfo = shutil.which("ibv_devinfo") is not None

        # libibverbs 库
        lib_paths = [
            "/usr/lib/x86_64-linux-gnu/libibverbs.so.1",
            "/usr/lib64/libibverbs.so.1",
            "/usr/lib/libibverbs.so.1",
        ]
        has_lib = any(os.path.exists(p) for p in lib_paths)

        # MOFED 版本
        mofed_ver = "unknown"
        if shutil.which("ofed_info"):
            rc, out, _ = run_cmd(["ofed_info", "-s"])
            if rc == 0 and out.strip():
                mofed_ver = out.strip().split("\n")[0]

        passed = has_ibv_devinfo and has_lib

        details_parts = []
        if has_ibv_devinfo:
            details_parts.append("ibv_devinfo 可用")
        if has_lib:
            details_parts.append("libibverbs 库可用")
        if mofed_ver != "unknown":
            details_parts.append(f"MOFED {mofed_ver}")

        self.add_result(CheckResult(
            name="MOFED / ibverbs",
            passed=passed,
            details=" | ".join(details_parts) if details_parts else "未安装",
            data={
                "has_ibv_devinfo": has_ibv_devinfo,
                "has_libibverbs": has_lib,
                "mofed_version": mofed_ver,
            },
            remediation=(
                "安装 MOFED:\n"
                "  方式1 (完整 MOFED): https://network.nvidia.com/products/infiniband-drivers/linux/\n"
                "  方式2 (包管理器): sudo apt install rdma-core libibverbs-dev ibverbs-utils\n"
                "  方式3 (RHEL): sudo yum install rdma-core libibverbs libibverbs-utils"
            ) if not passed else "",
            severity="error" if not has_lib else "info",
        ))

    # ------------------------------------------------------------------ #
    #  4. IB 设备端口检测
    # ------------------------------------------------------------------ #

    def check_ib_device(self):
        print(f"\n[4] IB 设备检测 ({self.ib_dev})")

        ib_root = Path("/sys/class/infiniband")
        if not ib_root.exists():
            self.add_result(CheckResult(
                name="IB 设备",
                passed=False,
                details="/sys/class/infiniband 不存在",
                remediation="请安装 MOFED 并确保 NIC 驱动已加载",
                severity="error",
            ))
            return

        # 列出设备
        all_devs = [d.name for d in ib_root.iterdir()]

        if self.ib_dev not in all_devs:
            self.add_result(CheckResult(
                name=f"IB 设备 ({self.ib_dev})",
                passed=False,
                details=f"设备不存在 (可用: {', '.join(all_devs)})",
                remediation=f"检查 IB 设备名，可用设备: {', '.join(all_devs)}",
                severity="error",
            ))
            return

        # 设备信息
        dev_path = ib_root / self.ib_dev
        node_type = read_file(str(dev_path / "node_type"))
        board_id = read_file(str(dev_path / "board_id"))
        fw_ver = read_file(str(dev_path / "fw_ver"))
        hw_rev = read_file(str(dev_path / "hca_type"))

        # 端口信息
        ports_info = []
        all_active = True
        ports_dir = dev_path / "ports"
        if ports_dir.exists():
            for port_dir in sorted(ports_dir.iterdir()):
                port_num = port_dir.name
                state = read_file(str(port_dir / "state"))
                rate = read_file(str(port_dir / "rate"))
                lid = read_file(str(port_dir / "lid"))
                phys_state = read_file(str(port_dir / "phys_state"))

                # GID 信息
                gids_dir = port_dir / "gids"
                gid_count = len(list(gids_dir.iterdir())) if gids_dir.exists() else 0

                ports_info.append({
                    "port": port_num,
                    "state": state,
                    "phys_state": phys_state,
                    "rate": rate,
                    "lid": lid,
                    "gid_count": gid_count,
                })

                if "ACTIVE" not in state.upper():
                    all_active = False

        # ibv_devinfo 详细信息
        ibv_info = ""
        if shutil.which("ibv_devinfo"):
            rc, out, _ = run_cmd(["ibv_devinfo", "-d", self.ib_dev])
            if rc == 0:
                ibv_info = out

        rate_str = ports_info[0]["rate"] if ports_info else "unknown"
        state_str = ports_info[0]["state"] if ports_info else "unknown"

        self.add_result(CheckResult(
            name=f"IB 设备 ({self.ib_dev})",
            passed=all_active and len(ports_info) > 0,
            details=f"{state_str} @ {rate_str} ({len(ports_info)} 端口)",
            data={
                "device": self.ib_dev,
                "node_type": node_type,
                "board_id": board_id,
                "firmware": fw_ver,
                "ports": ports_info,
                "ibv_devinfo": ibv_info[:500] if ibv_info else "",
            },
            remediation=(
                "IB 端口未激活:\n"
                "  sudo ip link set <iface> up\n"
                "  sudo systemctl start opensmd  (InfiniBand)\n"
                "  或检查 RoCE 配置"
            ) if not all_active else "",
            severity="error" if not all_active else "info",
        ))

    # ------------------------------------------------------------------ #
    #  5. CUDA GDR 属性检测
    # ------------------------------------------------------------------ #

    def check_cuda_gdr_attribute(self):
        print(f"\n[5] CUDA GDR 属性检测 (GPU {self.gpu_id})")

        try:
            cuda = ctypes.CDLL("libcuda.so")
        except OSError:
            self.add_result(CheckResult(
                name="CUDA GDR 属性",
                passed=False,
                details="无法加载 libcuda.so",
                remediation="确认 CUDA Driver 已安装且 libcuda.so 在库搜索路径中",
                severity="error",
            ))
            return

        CUDA_SUCCESS = 0

        ret = cuda.cuInit(0)
        if ret != CUDA_SUCCESS:
            self.add_result(CheckResult(
                name="CUDA GDR 属性",
                passed=False,
                details=f"cuInit 失败 (错误码 {ret})",
                severity="error",
            ))
            return

        dev = ctypes.c_int(0)
        ret = cuda.cuDeviceGet(ctypes.byref(dev), self.gpu_id)
        if ret != CUDA_SUCCESS:
            self.add_result(CheckResult(
                name="CUDA GDR 属性",
                passed=False,
                details=f"cuDeviceGet({self.gpu_id}) 失败 (错误码 {ret})",
                severity="error",
            ))
            return

        # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 106
        gdr_result = ctypes.c_int(0)
        ret = cuda.cuDeviceGetAttribute(
            ctypes.byref(gdr_result), 106, dev.value
        )

        gdr_supported = (ret == CUDA_SUCCESS and gdr_result.value == 1)

        # 额外属性: GDR Flush Buffer Size (108)
        flush_size = 0
        flush_result = ctypes.c_int(0)
        ret2 = cuda.cuDeviceGetAttribute(
            ctypes.byref(flush_result), 108, dev.value
        )
        if ret2 == CUDA_SUCCESS:
            flush_size = flush_result.value

        # P2P 支持 (20)
        p2p_result = ctypes.c_int(0)
        cuda.cuDeviceGetAttribute(ctypes.byref(p2p_result), 20, dev.value)

        details_parts = []
        if gdr_supported:
            details_parts.append("GDR 支持")
        else:
            details_parts.append("GDR 不支持")
        if flush_size > 0:
            details_parts.append(f"Flush Buffer: {flush_size} bytes")
        details_parts.append(f"P2P: {'支持' if p2p_result.value == 1 else '不支持'}")

        self.add_result(CheckResult(
            name="CUDA GDR 属性",
            passed=gdr_supported,
            details=" | ".join(details_parts),
            data={
                "gdr_supported": gdr_supported,
                "flush_buffer_size": flush_size,
                "p2p_supported": p2p_result.value == 1,
            },
            remediation=(
                "GPU 不支持 GPUDirect RDMA:\n"
                "  需要 Pascal 架构 (P100) 或更新\n"
                "  检查 GPU 型号是否为: Tesla P100/V100/A100/H100\n"
                "  确认 nvidia-peermem 已加载"
            ) if not gdr_supported else "",
            severity="error" if not gdr_supported else "info",
        ))

    # ------------------------------------------------------------------ #
    #  6. GPU-NIC P2P 拓扑检测
    # ------------------------------------------------------------------ #

    def check_gpu_nic_topology(self):
        print(f"\n[6] GPU-NIC P2P 拓扑检测")

        rc, out, _ = run_cmd(["nvidia-smi", "topo", "-m"])
        if rc != 0:
            self.add_result(CheckResult(
                name="GPU-NIC 拓扑",
                passed=False,
                details="nvidia-smi topo 不可用",
                severity="warning",
            ))
            return

        lines = out.splitlines()
        header = lines[0] if lines else ""

        # 查找 GPU 行
        gpu_line = ""
        for line in lines:
            if f"GPU{self.gpu_id}" in line:
                gpu_line = line
                break

        if not gpu_line:
            self.add_result(CheckResult(
                name="GPU-NIC 拓扑",
                passed=False,
                details=f"未找到 GPU{self.gpu_id} 的拓扑行",
                severity="warning",
            ))
            return

        # 查找 NIC 列
        nic_col_idx = -1
        cols = header.split()
        for i, col in enumerate(cols):
            if "NIC" in col.upper() or "MLX" in col.upper():
                nic_col_idx = i
                break

        link_type = "unknown"
        if nic_col_idx >= 0:
            gpu_cols = gpu_line.split()
            if nic_col_idx < len(gpu_cols):
                link_type = gpu_cols[nic_col_idx]

        # 分析连接类型
        link_desc = {
            "PIX": "同 PCIe Bridge (最优)",
            "PXB": "同 PCIe Root Complex (优)",
            "NODE": "同 NUMA 节点 (可用)",
            "SYS": "跨 NUMA 节点 (不推荐)",
        }
        link_quality = {"PIX": "best", "PXB": "good", "NODE": "acceptable", "SYS": "poor"}

        desc = link_desc.get(link_type, f"未知类型: {link_type}")
        quality = link_quality.get(link_type, "unknown")
        passed = link_type in ("PIX", "PXB", "NODE")

        # 提取拓扑图例
        legend = ""
        for i, line in enumerate(lines):
            if "Legend" in line:
                legend = "\n".join(lines[i:i+10])
                break

        self.add_result(CheckResult(
            name="GPU-NIC 拓扑",
            passed=passed,
            details=f"GPU{self.gpu_id} ↔ NIC: {link_type} ({desc})",
            data={
                "link_type": link_type,
                "quality": quality,
                "gpu_topo_line": gpu_line,
                "legend": legend,
            },
            remediation=(
                "GPU 与 NIC 之间跨 NUMA (SYS)，性能会降级\n"
                "建议:\n"
                "  1. 调整 GPU 和 NIC 的 PCIe 插槽位置\n"
                "  2. 使用 numactl 绑定进程到 GPU 所在 NUMA 节点\n"
                "  3. 参考服务器手册的 PCIe 拓扑布局"
            ) if link_type == "SYS" else "",
            severity="warning" if link_type == "SYS" else "info",
        ))

    # ------------------------------------------------------------------ #
    #  7. 系统资源检测
    # ------------------------------------------------------------------ #

    def check_system_resources(self):
        print(f"\n[7] 系统资源检测")

        # Locked memory
        locked_mem = os.popen("ulimit -l").read().strip()
        locked_ok = locked_mem == "unlimited" or (
            locked_mem.isdigit() and int(locked_mem) >= 65536
        )
        self.add_result(CheckResult(
            name="Locked Memory",
            passed=locked_ok,
            details=f"ulimit -l = {locked_mem} KB",
            remediation=(
                "增大 locked memory 限制:\n"
                "  sudo bash -c 'echo \"* soft memlock unlimited\" >> /etc/security/limits.conf'\n"
                "  sudo bash -c 'echo \"* hard memlock unlimited\" >> /etc/security/limits.conf'\n"
                "  然后重新登录或重启"
            ) if not locked_ok else "",
            severity="warning" if not locked_ok else "info",
        ))

        # Hugepages
        meminfo = read_file("/proc/meminfo")
        hp_match = re.search(r"HugePages_Total:\s+(\d+)", meminfo)
        hp_total = int(hp_match.group(1)) if hp_match else 0
        hp_ok = hp_total >= 1024
        self.add_result(CheckResult(
            name="Hugepages",
            passed=hp_ok,
            details=f"HugePages_Total = {hp_total} (建议 >= 1024)",
            remediation=(
                "配置 Hugepages:\n"
                "  sudo bash -c 'echo \"vm.nr_hugepages=1024\" > /etc/sysctl.d/99-gdr-hugepages.conf'\n"
                "  sudo sysctl -p /etc/sysctl.d/99-gdr-hugepages.conf"
            ) if not hp_ok else "",
            severity="warning" if not hp_ok else "info",
        ))

        # GPU persistence
        if shutil.which("nvidia-smi"):
            rc, out, _ = run_cmd([
                "nvidia-smi", "--query-gpu=persistence_mode",
                "--format=csv,noheader", "-i", str(self.gpu_id),
            ])
            pm = out.strip() if rc == 0 else "unknown"
            pm_ok = "Enabled" in pm
            self.add_result(CheckResult(
                name="GPU Persistence",
                passed=pm_ok,
                details=f"persistence_mode = {pm}",
                remediation="sudo nvidia-smi -pm 1" if not pm_ok else "",
                severity="warning" if not pm_ok else "info",
            ))

    # ------------------------------------------------------------------ #
    #  8. 3FS 挂载检测
    # ------------------------------------------------------------------ #

    def check_3fs(self):
        print(f"\n[8] 3FS 挂载检测")

        mountpoint = "/3fs/stage"
        mounted = os.path.ismount(mountpoint)

        mount_info = ""
        if mounted:
            rc, out, _ = run_cmd(["mount"])
            if rc == 0:
                for line in out.splitlines():
                    if mountpoint in line:
                        mount_info = line.strip()
                        break

        # 检测 3FS 二进制
        hf3fs_paths = [
            "hf3fs_fuse_main",
            "/opt/3fs/bin/hf3fs_fuse_main",
            "/usr/local/bin/hf3fs_fuse_main",
        ]
        hf3fs_bin = shutil.which(hf3fs_paths[0])
        if not hf3fs_bin:
            for p in hf3fs_paths[1:]:
                if os.path.exists(p) and os.access(p, os.X_OK):
                    hf3fs_bin = p
                    break

        # 写测试
        write_ok = False
        if mounted:
            test_file = os.path.join(mountpoint, ".gdr_test")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                write_ok = True
            except Exception:
                pass

        self.add_result(CheckResult(
            name="3FS 挂载",
            passed=mounted,
            details=(
                f"挂载于 {mountpoint}" if mounted else f"未挂载 ({mountpoint})"
            ) + (f" | 读写测试: {'通过' if write_ok else '失败'}" if mounted else ""),
            data={
                "mountpoint": mountpoint,
                "mounted": mounted,
                "mount_info": mount_info,
                "hf3fs_binary": hf3fs_bin or "未找到",
                "write_test": write_ok,
            },
            remediation=(
                "挂载 3FS:\n"
                f"  sudo bash scripts/setup_gdr.sh --mount-3fs\n"
                "  或手动:\n"
                "  sudo hf3fs_fuse_main --config /etc/3fs/hf3fs_fuse.toml"
            ) if not mounted else "",
            severity="warning" if not mounted else "info",
        ))

    # ------------------------------------------------------------------ #
    #  9. GDR 功能性测试
    # ------------------------------------------------------------------ #

    def check_gdr_functional(self):
        print(f"\n[9] GDR 功能性测试")

        test_script = Path(__file__).parent / "gdr_functional_test.py"
        if not test_script.exists():
            self.add_result(CheckResult(
                name="GDR 功能性测试",
                passed=False,
                details="测试脚本不存在",
                severity="warning",
            ))
            return

        rc, out, err = run_cmd([
            sys.executable, str(test_script),
            "--gpu", str(self.gpu_id),
            "--ib", self.ib_dev,
            "--test", "basic",
        ], timeout=30)

        passed = rc == 0
        self.add_result(CheckResult(
            name="GDR 功能性测试",
            passed=passed,
            details="RDMA 回环测试通过" if passed else f"测试失败 (退出码 {rc})",
            data={
                "exit_code": rc,
                "output": out[:500],
                "error": err[:500] if err else "",
            },
            remediation=(
                "功能性测试失败:\n"
                f"  输出: {out[:200]}\n"
                f"  错误: {err[:200]}\n"
                "检查:\n"
                "  1. nvidia-peermem 是否加载\n"
                "  2. IB 端口是否 ACTIVE\n"
                "  3. CUDA GDR 属性是否支持"
            ) if not passed else "",
            severity="error" if not passed else "info",
        ))

    # ------------------------------------------------------------------ #
    #  主入口
    # ------------------------------------------------------------------ #

    def run_all(self, run_functional: bool = False):
        print("=" * 60)
        print(" GPUDirect RDMA (GDR) 环境验证")
        print(f" GPU: {self.gpu_id} | IB: {self.ib_dev}")
        print(f" Host: {self.report.hostname}")
        print(f" OS: {self.report.os_info}")
        print("=" * 60)

        self.check_nvidia_driver()
        self.check_peermem()
        self.check_mofed()
        self.check_ib_device()
        self.check_cuda_gdr_attribute()
        self.check_gpu_nic_topology()
        self.check_system_resources()
        self.check_3fs()

        if run_functional:
            self.check_gdr_functional()

        # 汇总
        print("\n" + "=" * 60)
        print(" 验证结果汇总")
        print("=" * 60)
        r = self.report
        for c in r.checks:
            status = "OK" if c.passed else "FAIL" if c.severity == "error" else "WARN"
            print(f"  [{status:4}] {c.name}")

        print(f"\n  通过: {r.passed_count} | 警告: {r.warning_count} | 失败: {r.failed_count}")

        if r.all_passed:
            print("\n  GDR 环境就绪! 可以启动 SGLang with GDR。")
        else:
            print("\n  部分检查未通过，请根据上述提示修复。")

        return r

    def output_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n  JSON 报告已保存: {path}")

    def output_html(self, path: str):
        report = self.report.to_dict()
        html = self._generate_html(report)
        with open(path, "w") as f:
            f.write(html)
        print(f"\n  HTML 报告已保存: {path}")

    def _generate_html(self, report: dict) -> str:
        checks_html = ""
        for c in report["checks"]:
            status = "pass" if c["passed"] else "fail" if c["severity"] == "error" else "warn"
            status_text = "PASS" if c["passed"] else "FAIL" if c["severity"] == "error" else "WARN"
            rem = c.get("remediation", "").replace("\n", "<br>")
            checks_html += f"""
            <tr class="{status}">
                <td>{c['name']}</td>
                <td>{status_text}</td>
                <td>{c['details']}</td>
                <td class="remediation">{rem}</td>
            </tr>"""

        s = report["summary"]
        ready = "就绪" if s["gdr_ready"] else "未就绪"
        ready_class = "ready" if s["gdr_ready"] else "not-ready"

        return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>GDR 环境验证报告</title>
<style>
body {{ font-family: sans-serif; background: #1a1a2e; color: #eee; padding: 40px; }}
h1 {{ color: #64ffda; }} h2 {{ color: #82aaff; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th {{ background: #16213e; padding: 12px; text-align: left; border-bottom: 2px solid #0f3460; }}
td {{ padding: 10px; border-bottom: 1px solid #0f3460; }}
tr.pass td {{ background: #0d2818; }} tr.fail td {{ background: #2d0a0a; }}
tr.warn td {{ background: #2d250a; }} .remediation {{ color: #aaa; font-size: 13px; }}
.summary {{ font-size: 18px; margin: 20px 0; }}
.ready {{ color: #4caf50; }} .not-ready {{ color: #f44336; }}
</style></head><body>
<h1>GDR 环境验证报告</h1>
<p>时间: {report['timestamp']} | 主机: {report['hostname']} | GPU: {report['gpu_id']} | IB: {report['ib_device']}</p>
<p>操作系统: {report['os_info']}</p>
<div class="summary">
  <b>GDR 状态: <span class="{ready_class}">{ready}</span></b><br>
  通过: {s['passed']} | 警告: {s['warnings']} | 失败: {s['failed']} | 共 {s['total']} 项
</div>
<table>
<tr><th>检查项</th><th>状态</th><th>详情</th><th>修复建议</th></tr>
{checks_html}
</table>
</body></html>"""


# ======================================================================= #
#  主函数
# ======================================================================= #


def main():
    parser = argparse.ArgumentParser(
        description="GDR 环境验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU 设备 ID")
    parser.add_argument("--ib", default="mlx5_0", help="IB 设备名")
    parser.add_argument("--output", "-o", help="报告输出路径")
    parser.add_argument("--format", choices=["text", "json", "html"], default="text",
                        help="报告格式")
    parser.add_argument("--run-functional", action="store_true",
                        help="执行 GDR 功能性测试")
    parser.add_argument("--fix-suggestions", action="store_true",
                        help="输出详细修复建议")
    args = parser.parse_args()

    verifier = GDRVerifier(gpu_id=args.gpu, ib_dev=args.ib)
    report = verifier.run_all(run_functional=args.run_functional)

    if args.output and args.format in ("json", "html"):
        if args.format == "json":
            verifier.output_json(args.output)
        else:
            verifier.output_html(args.output)

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
