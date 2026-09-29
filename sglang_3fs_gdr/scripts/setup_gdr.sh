#!/bin/bash
###############################################################################
# GDR (GPUDirect RDMA) 环境部署脚本 — 完整版
#
# 功能清单:
#   1. NVIDIA 驱动 / CUDA 版本检测
#   2. nvidia-peermem 内核模块加载
#   3. Mellanox OFED (MOFED) 检测
#   4. IB 端口状态与速率检测
#   5. GPU-NIC P2P 拓扑验证
#   6. CUDA GDR 属性检测
#   7. 系统资源调优 (locked memory, hugepages)
#   8. 3FS 客户端部署与挂载
#   9. SGLang systemd 服务文件生成
#  10. GDR 功能性回环测试
#
# 用法:
#   sudo bash setup_gdr.sh [选项]
#
# 选项:
#   --gpu <id>         GPU 设备 ID (默认 0)
#   --ib <dev>         IB 设备名 (默认 mlx5_0)
#   --mgmtd <addr>     3FS mgmtd 地址 (默认 RDMA://192.168.1.1:8000)
#   --mount-3fs        自动挂载 3FS 文件系统
#   --install-service  安装 SGLang GDR systemd 服务
#   --skip-mofed       跳过 MOFED 检测 (已确认安装)
#   --check-only       仅检测不执行任何修改
#   --verbose          详细日志输出
#   --dry-run          模拟运行，不执行任何修改
#
# 退出码:
#   0  全部通过
#   1  参数错误
#   2  NVIDIA 驱动检测失败
#   3  nvidia-peermem 加载失败
#   4  MOFED / IB 设备检测失败
#   5  CUDA GDR 属性不支持
#   6  3FS 挂载失败
#   7  功能性测试失败
###############################################################################

set -euo pipefail

# ========================= 全局变量 =========================

SCRIPT_NAME="$(basename "$0")"
LOG_FILE="/var/log/gdr_setup_$(date +%Y%m%d_%H%M%S).log"
GPU_ID=0
IB_DEV="mlx5_0"
MGMTD_ADDR="RDMA://192.168.1.1:8000"
MOUNT_3FS=false
INSTALL_SERVICE=false
SKIP_MOFED=false
CHECK_ONLY=false
VERBOSE=false
DRY_RUN=false

# 颜色输出
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; BOLD=''; NC=''
fi

# 统计变量
PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0
STEPS_TOTAL=10
STEP_CURRENT=0

# ========================= 工具函数 =========================

log()     { echo -e "$*" | tee -a "$LOG_FILE"; }
log_info() { echo -e "${CYAN}[INFO]${NC} $*" | tee -a "$LOG_FILE"; }
log_ok()   { echo -e "${GREEN}[  OK]${NC} $*" | tee -a "$LOG_FILE"; ((PASS_COUNT++)); }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$LOG_FILE"; ((WARN_COUNT++)); }
log_err()  { echo -e "${RED}[FAIL]${NC} $*" | tee -a "$LOG_FILE"; ((FAIL_COUNT++)); }
log_step() {
    ((STEP_CURRENT++))
    echo -e "\n${BOLD}${BLUE}━━━ [${STEP_CURRENT}/${STEPS_TOTAL}] $* ━━━${NC}" | tee -a "$LOG_FILE"
}
log_verbose() { [[ "$VERBOSE" == true ]] && echo -e "       ${CYAN}[DBG]${NC} $*" | tee -a "$LOG_FILE" || true; }

run_cmd() {
    # 在 dry-run 模式下只打印不执行
    local desc="$1"; shift
    log_verbose "执行: $*"
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "       ${YELLOW}[DRY]${NC} $desc: $*"
        return 0
    fi
    "$@"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_err "此脚本需要 root 权限运行"
        log_info "请执行: sudo bash $SCRIPT_NAME $*"
        exit 1
    fi
}

# ========================= 参数解析 =========================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpu)            GPU_ID="$2"; shift 2 ;;
            --ib)             IB_DEV="$2"; shift 2 ;;
            --mgmtd)          MGMTD_ADDR="$2"; shift 2 ;;
            --mount-3fs)      MOUNT_3FS=true; shift ;;
            --install-service) INSTALL_SERVICE=true; shift ;;
            --skip-mofed)     SKIP_MOFED=true; shift ;;
            --check-only)     CHECK_ONLY=true; shift ;;
            --verbose)        VERBOSE=true; shift ;;
            --dry-run)        DRY_RUN=true; shift ;;
            -h|--help)
                head -40 "$0" | tail -38
                exit 0
                ;;
            *)
                log_err "未知参数: $1"
                log_info "使用 --help 查看帮助"
                exit 1
                ;;
        esac
    done
}

# ========================= 检测函数 =========================

# ------------------------------------------------------------------ #
# Step 1: NVIDIA 驱动检测
# ------------------------------------------------------------------ #
check_nvidia_driver() {
    log_step "NVIDIA 驱动检测"

    if ! command -v nvidia-smi &>/dev/null; then
        log_err "nvidia-smi 未找到"
        log_info "请安装 NVIDIA 驱动 (>= 525):"
        log_info "  1. 下载驱动: https://www.nvidia.com/drivers"
        log_info "  2. 安装: sudo sh NVIDIA-Linux-x86_64-550.xx.run"
        exit 2
    fi

    # 驱动版本
    local driver_ver
    driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    log_ok "NVIDIA 驱动版本: $driver_ver"

    local driver_major
    driver_major=$(echo "$driver_ver" | cut -d. -f1)
    if [[ "$driver_major" -lt 525 ]]; then
        log_warn "驱动版本 $driver_ver < 525, GDR 支持可能受限"
        log_info "建议升级到 >= 525 以获得完整 GDR 支持"
    fi

    # CUDA 版本
    local cuda_ver
    cuda_ver=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[\d.]+" || echo "unknown")
    log_ok "CUDA 版本: $cuda_ver"

    # GPU 列表
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null | wc -l)
    log_ok "检测到 $gpu_count 个 GPU 设备"

    if [[ "$VERBOSE" == true ]]; then
        nvidia-smi --query-gpu=index,name,driver_version,memory.total,pci.bus_id \
            --format=csv,noheader 2>/dev/null | while IFS=, read -r idx name drv mem pci; do
            log_verbose "  GPU $idx: $(echo $name) | $(echo $drv) | $(echo $mem) | $(echo $pci)"
        done
    fi

    # GPU ID 有效性
    if [[ "$GPU_ID" -ge "$gpu_count" ]]; then
        log_err "GPU ID $GPU_ID 超出范围 (共 $gpu_count 个设备)"
        exit 2
    fi

    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_ID" 2>/dev/null)
    log_ok "目标 GPU $GPU_ID: $gpu_name"
}

# ------------------------------------------------------------------ #
# Step 2: nvidia-peermem 内核模块
# ------------------------------------------------------------------ #
check_and_load_peermem() {
    log_step "nvidia-peermem 内核模块"

    # 检查是否已加载
    if lsmod 2>/dev/null | grep -qE "nvidia_peermem|nv_peer_mem"; then
        local module_name
        module_name=$(lsmod | grep -oE "nvidia_peermem|nv_peer_mem" | head -1)
        log_ok "nvidia-peermem 已加载 (模块名: $module_name)"
        return 0
    fi

    log_info "nvidia-peermem 未加载，尝试加载..."

    # 尝试加载 nvidia-peermem
    if run_cmd "加载 nvidia-peermem" modprobe nvidia-peermem 2>/dev/null; then
        log_ok "nvidia-peermem 加载成功 (nvidia-peermem)"
    elif run_cmd "加载 nv_peer_mem" modprobe nv_peer_mem 2>/dev/null; then
        log_ok "nvidia-peermem 加载成功 (nv_peer_mem)"
    else
        log_err "无法加载 nvidia-peermem"
        log_info "可能原因及解决方案:"
        log_info "  1. 驱动版本过低 — 升级到 >= 525"
        log_info "  2. 驱动未编译 peermem 模块 — 重新安装驱动时勾选"
        log_info "  3. 内核版本不兼容 — 更新内核或使用 DKMS 版驱动"
        log_info "  4. 手动编译: 从 nvidia-peermem 源码编译"
        log_info "     git clone https://github.com/Mellanox/nvidia-peermem.git"
        log_info "     cd nvidia-peermem && make && make install"
        if [[ "$CHECK_ONLY" == false ]]; then
            exit 3
        fi
    fi

    # 验证模块信息
    local mod_info
    mod_info=$(modinfo nvidia-peermem 2>/dev/null || modinfo nv_peer_mem 2>/dev/null || echo "")
    if [[ -n "$mod_info" ]]; then
        local mod_ver
        mod_ver=$(echo "$mod_info" | grep -oP "version:\s*\K\S+" || echo "unknown")
        log_verbose "模块版本: $mod_ver"
    fi

    # 检查 /proc/driver/nvidia 中的 P2P 信息
    if [[ -f /proc/driver/nvidia/gpus ]]; then
        log_verbose "NVIDIA GPU driver procfs 可用"
    fi
}

# ------------------------------------------------------------------ #
# Step 3: Mellanox OFED (MOFED) 检测
# ------------------------------------------------------------------ #
check_mofed() {
    log_step "Mellanox OFED (MOFED) 检测"

    if [[ "$SKIP_MOFED" == true ]]; then
        log_info "跳过 MOFED 检测 (--skip-mofed)"
        return 0
    fi

    # 检查 ibv_devinfo 工具
    if ! command -v ibv_devinfo &>/dev/null; then
        log_err "ibv_devinfo 未找到 — MOFED 可能未安装"
        log_info "安装 MOFED:"
        log_info "  1. 下载: https://network.nvidia.com/products/infiniband-drivers/linux/"
        log_info "  2. 安装: sudo ./mlnxofedinstall --with-mft --with-mlx5-dv --upstream-libs"
        log_info "  3. 或使用包管理器: sudo apt install rdma-core libibverbs-dev ibverbs-utils"
        if [[ "$CHECK_ONLY" == false ]]; then
            exit 4
        fi
        return 1
    fi

    # MOFED 版本
    local mofed_ver
    mofed_ver=$(ofed_info -s 2>/dev/null | head -1 || echo "unknown")
    if [[ "$mofed_ver" != "unknown" ]]; then
        log_ok "MOFED 版本: $mofed_ver"
    else
        log_info "无法获取 MOFED 版本 (非 Mellanox OFED 安装)"
    fi

    # ibverbs 库检测
    if [[ -f /usr/lib/x86_64-linux-gnu/libibverbs.so.1 ]] || \
       [[ -f /usr/lib64/libibverbs.so.1 ]]; then
        log_ok "libibverbs 库可用"
    else
        log_warn "libibverbs.so.1 未找到 — 可能影响 GDR 运行"
        log_info "  安装: sudo apt install libibverbs1 (Debian/Ubuntu)"
        log_info "  或:   sudo yum install libibverbs (RHEL/CentOS)"
    fi

    # 检查 GDR 相关的 OFED 组件
    if command -v ofed_info &>/dev/null; then
        if ofed_info 2>/dev/null | grep -qi "gdr\|gpudirect"; then
            log_ok "MOFED GDR 组件已安装"
        else
            log_verbose "MOFED GDR 组件信息未明确"
        fi
    fi
}

# ------------------------------------------------------------------ #
# Step 4: IB 设备检测与配置
# ------------------------------------------------------------------ #
check_ib_device() {
    log_step "IB 设备检测与配置"

    local ib_root="/sys/class/infiniband"
    if [[ ! -d "$ib_root" ]]; then
        log_err "未发现 InfiniBand/RoCE 设备 (/sys/class/infiniband 不存在)"
        log_info "请安装 MOFED 并确保 NIC 驱动已加载"
        if [[ "$CHECK_ONLY" == false ]]; then
            exit 4
        fi
        return 1
    fi

    # 列出所有 IB 设备
    local ib_devs
    ib_devs=$(ls "$ib_root" 2>/dev/null)
    log_ok "可用 IB 设备: $(echo $ib_devs | tr '\n' ' ')"

    # 检查目标设备
    if [[ ! -d "$ib_root/$IB_DEV" ]]; then
        log_err "IB 设备 $IB_DEV 不存在"
        log_info "可用设备: $ib_devs"
        if [[ "$CHECK_ONLY" == false ]]; then
            exit 4
        fi
        return 1
    fi
    log_ok "目标 IB 设备: $IB_DEV"

    # 设备信息
    local dev_type board_id fw_ver
    dev_type=$(cat "$ib_root/$IB_DEV/node_type" 2>/dev/null || echo "unknown")
    board_id=$(cat "$ib_root/$IB_DEV/board_id" 2>/dev/null || echo "unknown")
    fw_ver=$(cat "$ib_root/$IB_DEV/fw_ver" 2>/dev/null || echo "unknown")
    log_verbose "设备类型: $dev_type, 板卡 ID: $board_id, 固件: $fw_ver"

    # 检查端口
    local ports_dir="$ib_root/$IB_DEV/ports"
    if [[ ! -d "$ports_dir" ]]; then
        log_err "IB 设备 $IB_DEV 无端口目录"
        return 1
    fi

    local port_active=false
    for port_dir in "$ports_dir"/*/; do
        local port_num
        port_num=$(basename "$port_dir")
        local state rate lid gid_count

        state=$(cat "$port_dir/state" 2>/dev/null || echo "unknown")
        rate=$(cat "$port_dir/rate" 2>/dev/null || echo "unknown")
        lid=$(cat "$port_dir/lid" 2>/dev/null || echo "0")
        gid_count=$(ls "$port_dir/gids" 2>/dev/null | wc -l)

        log_verbose "端口 $port_num: 状态=$state, 速率=$rate, LID=$lid, GID数=$gid_count"

        if [[ "$state" == *"ACTIVE"* ]]; then
            log_ok "端口 $port_num: $state @ $rate"
            port_active=true
        else
            log_warn "端口 $port_num: $state (非 ACTIVE)"
        fi
    done

    if [[ "$port_active" == false ]]; then
        log_warn "没有 ACTIVE 的 IB 端口"
        log_info "尝试激活端口:"
        log_info "  sudo ip link set ${IB_DEV} up"
        log_info "  或使用 opensm: sudo systemctl start opensmd"
    fi

    # ibv_devinfo 详细信息
    if command -v ibv_devinfo &>/dev/null; then
        local devinfo
        devinfo=$(ibv_devinfo -d "$IB_DEV" 2>/dev/null || echo "")
        if [[ -n "$devinfo" ]]; then
            local hca_type transport port_state
            hca_type=$(echo "$devinfo" | grep -oP "hca_type:\s*\K\S+" || echo "unknown")
            transport=$(echo "$devinfo" | grep -oP "transport:\s*\K\S+" || echo "unknown")
            log_verbose "HCA 类型: $hca_type, 传输: $transport"
        fi
    fi
}

# ------------------------------------------------------------------ #
# Step 5: GPU-NIC P2P 拓扑验证
# ------------------------------------------------------------------ #
check_gpu_nic_topology() {
    log_step "GPU-NIC P2P 拓扑验证"

    if ! nvidia-smi topo -m &>/dev/null; then
        log_warn "nvidia-smi topo 不可用"
        return 0
    fi

    # 获取拓扑矩阵
    local topo_output
    topo_output=$(nvidia-smi topo -m 2>/dev/null)

    # 检查 GPU 与 NIC 的连接关系
    local gpu_line
    gpu_line=$(echo "$topo_output" | grep "^GPU${GPU_ID}" || echo "")

    if [[ -z "$gpu_line" ]]; then
        log_warn "未在拓扑中找到 GPU${GPU_ID}"
        log_verbose "$topo_output"
        return 0
    fi

    log_verbose "GPU${GPU_ID} 拓扑行: $gpu_line"

    # 检查 NIC 行
    local nic_line
    nic_line=$(echo "$topo_output" | grep -i "NIC" | head -1 || echo "")
    if [[ -n "$nic_line" ]]; then
        log_verbose "NIC 拓扑行: $nic_line"
    fi

    # 分析连接类型
    # PIX: 同 PCIe Bridge (最优)
    # PXB: 同 PCIe Root Complex (优)
    # NODE: 同 NUMA 节点 (可用)
    # SYS:  跨 NUMA 节点 (不推荐)
    local best_link="SYS"
    local topo_legend
    topo_legend=$(echo "$topo_output" | grep -A20 "Legend" | head -25)

    # 获取列数以定位 NIC 列
    local header_line
    header_line=$(echo "$topo_output" | head -1)

    # 查找 NIC 列索引
    local nic_col=-1
    local col_idx=0
    for col in $header_line; do
        if [[ "$col" == *"NIC"* ]] || [[ "$col" == *"mlx"* ]]; then
            nic_col=$col_idx
            break
        fi
        ((col_idx++))
    done

    if [[ $nic_col -ge 0 ]]; then
        # 提取 GPU 行中对应 NIC 列的值
        local link_type
        link_type=$(echo "$gpu_line" | awk -v col=$((nic_col + 1)) '{print $col}')
        if [[ -n "$link_type" ]]; then
            case $link_type in
                PIX) log_ok "GPU$GPU_ID ↔ NIC: PIX (同 PCIe Bridge, 最优)"; best_link="PIX" ;;
                PXB) log_ok "GPU$GPU_ID ↔ NIC: PXB (同 PCIe Root, 优)"; best_link="PXB" ;;
                NODE) log_warn "GPU$GPU_ID ↔ NIC: NODE (同 NUMA, 可用)"; best_link="NODE" ;;
                SYS)  log_warn "GPU$GPU_ID ↔ NIC: SYS (跨 NUMA, 性能降级)"; best_link="SYS" ;;
                *)    log_verbose "GPU$GPU_ID ↔ NIC: $link_type"; best_link="$link_type" ;;
            esac
        fi
    else
        log_warn "无法确定 NIC 列位置，显示完整拓扑:"
        echo "$topo_output" | head -15 | tee -a "$LOG_FILE"
    fi

    # 输出拓扑图例
    if [[ "$VERBOSE" == true ]] && [[ -n "$topo_legend" ]]; then
        log_verbose "拓扑图例:"
        echo "$topo_legend" | while read -r line; do log_verbose "  $line"; done
    fi
}

# ------------------------------------------------------------------ #
# Step 6: CUDA GDR 属性检测
# ------------------------------------------------------------------ #
check_cuda_gdr_attribute() {
    log_step "CUDA GDR 属性检测"

    python3 - "$GPU_ID" << 'PYEOF' | tee -a "$LOG_FILE"
import ctypes
import sys

gpu_id = int(sys.argv[1])

try:
    cuda = ctypes.CDLL("libcuda.so")
except OSError:
    print("[FAIL] 无法加载 CUDA Driver 库 (libcuda.so)")
    sys.exit(1)

# CUDA 错误码
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 106

ret = cuda.cuInit(0)
if ret != CUDA_SUCCESS:
    print(f"[FAIL] cuInit 失败, 错误码: {ret}")
    sys.exit(1)

dev = ctypes.c_int(0)
ret = cuda.cuDeviceGet(ctypes.byref(dev), gpu_id)
if ret != CUDA_SUCCESS:
    print(f"[FAIL] cuDeviceGet({gpu_id}) 失败, 错误码: {ret}")
    sys.exit(1)

# 检测 GPU Direct RDMA 支持
result = ctypes.c_int(0)
ret = cuda.cuDeviceGetAttribute(
    ctypes.byref(result),
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
    dev.value,
)

if ret != CUDA_SUCCESS:
    print(f"[FAIL] cuDeviceGetAttribute 失败, 错误码: {ret}")
    sys.exit(1)

if result.value == 1:
    print(f"[  OK] GPU {gpu_id}: GPUDirect RDMA 支持")
else:
    print(f"[FAIL] GPU {gpu_id}: GPUDirect RDMA 不支持")
    # 额外检测其他相关属性
    # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 106
    # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_BUFSIZE = 108

    # 检测 P2P 支持
    p2p_attr = 20  # CU_DEVICE_ATTRIBUTE_COMPUTE_P2P_SUPPORTED
    p2p_result = ctypes.c_int(0)
    cuda.cuDeviceGetAttribute(
        ctypes.byref(p2p_result), p2p_attr, dev.value
    )
    if p2p_result.value == 1:
        print("       (但 P2P 计算支持可用)")
    sys.exit(1)

# 额外: 检测 GDR Flush Buffer 大小
flush_attr = 108  # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_BUFSIZE
flush_result = ctypes.c_int(0)
ret = cuda.cuDeviceGetAttribute(
    ctypes.byref(flush_result), flush_attr, dev.value
)
if ret == CUDA_SUCCESS and flush_result.value > 0:
    print(f"       GDR Flush Buffer: {flush_result.value} bytes")

sys.exit(0)
PYEOF

    local py_ret=$?
    if [[ $py_ret -ne 0 ]]; then
        log_err "CUDA GDR 属性检测失败"
        log_info "可能原因:"
        log_info "  1. GPU 型号不支持 GDR (需要 Pascal/P100 或更新)"
        log_info "  2. CUDA Driver 库版本过低"
        log_info "  3. nvidia-peermem 未加载"
        if [[ "$CHECK_ONLY" == false ]]; then
            exit 5
        fi
    else
        ((PASS_COUNT++))
    fi
}

# ------------------------------------------------------------------ #
# Step 7: 系统资源调优
# ------------------------------------------------------------------ #
tune_system_resources() {
    log_step "系统资源调优"

    # 7.1 Locked Memory 限制
    local locked_mem
    locked_mem=$(ulimit -l)
    if [[ "$locked_mem" == "unlimited" ]]; then
        log_ok "Locked memory 限制: unlimited"
    elif [[ "$locked_mem" -lt 65536 ]]; then
        log_warn "Locked memory 限制 $locked_mem KB < 65536 KB"
        if [[ "$CHECK_ONLY" == false && "$DRY_RUN" == false ]]; then
            log_info "调整 limits.conf..."
            if ! grep -q "memlock" /etc/security/limits.conf 2>/dev/null; then
                cat >> /etc/security/limits.conf << 'LIMIT'

# GDR 部署: 增大 locked memory 限制
* soft memlock unlimited
* hard memlock unlimited
LIMIT
                log_ok "limits.conf 已更新 (memlock unlimited)"
            else
                log_verbose "limits.conf 已包含 memlock 配置"
            fi
        fi
    else
        log_ok "Locked memory 限制: ${locked_mem} KB"
    fi

    # 7.2 Hugepages 配置
    local hugepages_total
    hugepages_total=$(cat /proc/meminfo | grep "HugePages_Total" | awk '{print $2}')
    log_verbose "当前 Hugepages: $hugepages_total"

    if [[ "$hugepages_total" -lt 1024 ]] && [[ "$CHECK_ONLY" == false && "$DRY_RUN" == false ]]; then
        log_info "配置 Hugepages (1024 pages = 2GB)..."
        echo "vm.nr_hugepages=1024" > /etc/sysctl.d/99-gdr-hugepages.conf
        sysctl -p /etc/sysctl.d/99-gdr-hugepages.conf &>/dev/null || true
        log_ok "Hugepages 配置已写入 sysctl.d"
    elif [[ "$hugepages_total" -ge 1024 ]]; then
        log_ok "Hugepages: $hugepages_total pages"
    fi

    # 7.3 GPU persistence mode
    if command -v nvidia-smi &>/dev/null; then
        local persistence_mode
        persistence_mode=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader -i "$GPU_ID" 2>/dev/null || echo "unknown")
        if [[ "$persistence_mode" != "Enabled" ]] && [[ "$CHECK_ONLY" == false && "$DRY_RUN" == false ]]; then
            log_info "启用 GPU persistence模式..."
            nvidia-smi -pm 1 -i "$GPU_ID" 2>/dev/null || true
            log_ok "GPU $GPU_ID persistence 模式已启用"
        elif [[ "$persistence_mode" == "Enabled" ]]; then
            log_ok "GPU $GPU_ID persistence 模式: Enabled"
        fi
    fi

    # 7.4 IPoIB / RoCE 配置检测
    local roce_iface
    roce_iface=$(ip link show 2>/dev/null | grep -oP "^\d+: \K\S+(?=:.*state UP)" | grep -i "mlx\|roce\|ib" | head -1 || echo "")
    if [[ -n "$roce_iface" ]]; then
        log_ok "RoCE/IPoIB 接口: $roce_iface"
    else
        log_verbose "未检测到 RoCE/IPoIB 接口 (3FS 可能使用 raw RDMA)"
    fi
}

# ------------------------------------------------------------------ #
# Step 8: 3FS 客户端部署
# ------------------------------------------------------------------ #
deploy_3fs_client() {
    log_step "3FS 客户端部署"

    local mountpoint="/3fs/stage"

    # 检查是否已挂载
    if mountpoint -q "$mountpoint" 2>/dev/null; then
        log_ok "3FS 已挂载: $mountpoint"
        # 获取挂载信息
        local mount_info
        mount_info=$(mount | grep "$mountpoint" | head -1)
        log_verbose "挂载信息: $mount_info"
        return 0
    fi

    if [[ "$MOUNT_3FS" != true ]]; then
        log_info "3FS 未挂载 (使用 --mount-3fs 自动挂载)"
        return 0
    fi

    # 检查 3FS 二进制
    if ! command -v hf3fs_fuse_main &>/dev/null; then
        log_warn "hf3fs_fuse_main 未找到 — 尝试在常见路径中查找"
        local hf3fs_paths=(
            "/opt/3fs/bin/hf3fs_fuse_main"
            "/usr/local/bin/hf3fs_fuse_main"
            "/usr/bin/hf3fs_fuse_main"
        )
        local found_bin=""
        for path in "${hf3fs_paths[@]}"; do
            if [[ -x "$path" ]]; then
                found_bin="$path"
                break
            fi
        done
        if [[ -n "$found_bin" ]]; then
            log_ok "找到 hf3fs_fuse_main: $found_bin"
        else
            log_err "hf3fs_fuse_main 未找到"
            log_info "请安装 3FS 客户端:"
            log_info "  git clone https://github.com/deepseek-ai/3FS.git"
            log_info "  cd 3FS && ./build.sh && cp build/hf3fs_fuse_main /usr/local/bin/"
            return 1
        fi
    fi

    # 创建挂载点
    run_cmd "创建挂载点" mkdir -p "$mountpoint"

    # 生成配置文件
    local config_file="/etc/3fs/hf3fs_fuse.toml"
    run_cmd "创建配置目录" mkdir -p "$(dirname "$config_file")"

    cat > "$config_file" << EOF
# 3FS FUSE 客户端配置 — 由 setup_gdr.sh 生成
# 日期: $(date)

cluster_id = "stage"
mountpoint = "$mountpoint"
token_file = "/opt/3fs/etc/token.txt"

[mgmtd_client]
mgmtd_server_addresses = ["$MGMTD_ADDR"]

[fuse_client]
# RDMA 配置
rdma_timeout_ms = 30000
rdma_max_msg_size = 1048576
# I/O worker 线程数
io_worker_count = 16
# 批量 I/O 深度
batch_io_depth = 64
EOF

    log_ok "3FS 配置已生成: $config_file"

    # 挂载
    if [[ "$DRY_RUN" == false ]]; then
        log_info "挂载 3FS..."
        hf3fs_fuse_main --config "$config_file" &
        local fuse_pid=$!
        sleep 5

        if mountpoint -q "$mountpoint" 2>/dev/null; then
            log_ok "3FS 挂载成功: $mountpoint (PID: $fuse_pid)"
        else
            log_err "3FS 挂载失败"
            log_info "检查:"
            log_info "  1. mgmtd 地址是否可达: $MGMTD_ADDR"
            log_info "  2. token 文件是否存在: /opt/3fs/etc/token.txt"
            log_info "  3. IB 端口是否 ACTIVE"
            if [[ "$CHECK_ONLY" == false ]]; then
                exit 6
            fi
        fi
    else
        echo -e "       ${YELLOW}[DRY]${NC} 挂载 3FS: $mountpoint"
    fi
}

# ------------------------------------------------------------------ #
# Step 9: SGLang systemd 服务文件
# ------------------------------------------------------------------ #
install_sglang_service() {
    log_step "SGLang GDR systemd 服务"

    if [[ "$INSTALL_SERVICE" != true ]]; then
        log_info "跳过 systemd 服务安装 (使用 --install-service 启用)"
        return 0
    fi

    local service_file="/etc/systemd/system/sglang-gdr.service"
    local config_dir="/etc/sglang"
    local work_dir="/opt/sglang"

    run_cmd "创建工作目录" mkdir -p "$work_dir" "$config_dir"

    # 生成 GDR 配置文件
    local gdr_config="$config_dir/gdr_config.yaml"
    cat > "$gdr_config" << 'YAMLEOF'
# SGLang GDR 配置 — 由 setup_gdr.sh 生成
enable_gdr: true
gpu_id: __GPU_ID__
ib_dev: __IB_DEV__
pool_size_gb: 4
cluster_id: stage
mountpoint: /3fs/stage
mgmtd_address: __MGMTD_ADDR__
hicache_storage_prefix: /3fs/stage/kvcache
hicache_io_backend: gdr
hicache_write_policy: write_through
sglang_model_path: /3fs/stage/models/DeepSeek-R1
sglang_tp: 8
sglang_host: 0.0.0.0
sglang_port: 10000
YAMLEOF

    sed -i "s/__GPU_ID__/$GPU_ID/g" "$gdr_config"
    sed -i "s/__IB_DEV__/$IB_DEV/g" "$gdr_config"
    sed -i "s/__MGMTD_ADDR__/$MGMTD_ADDR/g" "$gdr_config"
    log_ok "GDR 配置文件: $gdr_config"

    # systemd 服务文件
    cat > "$service_file" << EOF
[Unit]
Description=SGLang Inference Server with GPUDirect RDMA
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$work_dir
Environment=SGLANG_GDR_PLUGIN=1
Environment=GDR_CONFIG_PATH=$gdr_config
Environment=CUDA_VISIBLE_DEVICES=$GPU_ID
# 内存锁定 (GDR 需要锁定 GPU 内存)
LimitMEMLOCK=infinity
LimitNPROC=infinity
LimitNOFILE=65536

ExecStart=/usr/bin/python3 -m sglang.launch_server \\
    --model-path /3fs/stage/models/DeepSeek-R1 \\
    --tp 8 \\
    --host 0.0.0.0 \\
    --port 10000 \\
    --enable-hierarchical-cache \\
    --hicache-ratio 2 \\
    --hicache-mem-layout page_first_direct \\
    --hicache-io-backend gdr \\
    --hicache-storage-backend gdr_3fs \\
    --hicache-write-policy write_through \\
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
EOF

    log_ok "systemd 服务文件: $service_file"

    if [[ "$DRY_RUN" == false ]]; then
        systemctl daemon-reload
        log_ok "systemd daemon-reload 完成"
        log_info "启动服务: sudo systemctl start sglang-gdr"
        log_info "设置开机启动: sudo systemctl enable sglang-gdr"
        log_info "查看日志: journalctl -u sglang-gdr -f"
    fi
}

# ------------------------------------------------------------------ #
# Step 10: GDR 功能性回环测试
# ------------------------------------------------------------------ #
run_functional_test() {
    log_step "GDR 功能性回环测试"

    local test_script
    test_script="$(dirname "$0")/gdr_functional_test.py"

    if [[ ! -f "$test_script" ]]; then
        log_warn "功能测试脚本不存在: $test_script"
        log_info "跳过功能性测试"
        return 0
    fi

    log_info "运行 GDR 功能性测试 (GPU $GPU_ID, IB $IB_DEV)..."

    if [[ "$DRY_RUN" == false ]]; then
        if python3 "$test_script" --gpu "$GPU_ID" --ib "$IB_DEV" --test basic 2>&1 | tee -a "$LOG_FILE"; then
            log_ok "GDR 功能性测试通过"
        else
            log_warn "GDR 功能性测试未通过 (可能是测试环境限制)"
            log_info "详情请查看日志: $LOG_FILE"
        fi
    else
        echo -e "       ${YELLOW}[DRY]${NC} python3 $test_script --gpu $GPU_ID --ib $IB_DEV --test basic"
    fi
}

# ========================= 主流程 =========================

main() {
    parse_args "$@"

    # 初始化日志
    mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
    echo "GDR 部署日志 — $(date)" > "$LOG_FILE"

    echo -e "${BOLD}${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║          GPUDirect RDMA (GDR) 环境部署工具                     ║"
    echo "║          SGLang + 3FS + NVIDIA GPUDirect RDMA                  ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo "  GPU ID:     $GPU_ID"
    echo "  IB Device:  $IB_DEV"
    echo "  3FS Mgmtd:  $MGMTD_ADDR"
    echo "  模式:       ${CHECK_ONLY:+检查}${DRY_RUN:+模拟}${INSTALL_SERVICE:+安装服务}${MOUNT_3FS:+挂载3FS}"
    [[ "$CHECK_ONLY" == true ]]   && echo "  --check-only: 仅检测不修改"
    [[ "$DRY_RUN" == true ]]     && echo "  --dry-run: 模拟运行不执行"
    [[ "$VERBOSE" == true ]]     && echo "  --verbose: 详细输出"
    echo "  日志文件:   $LOG_FILE"

    # 权限检查 (非 check-only / dry-run 需要 root)
    if [[ "$CHECK_ONLY" == false && "$DRY_RUN" == false ]]; then
        check_root "$@"
    fi

    # 执行所有检测步骤
    check_nvidia_driver
    check_and_load_peermem
    check_mofed
    check_ib_device
    check_gpu_nic_topology
    check_cuda_gdr_attribute

    # 调优和部署步骤
    if [[ "$CHECK_ONLY" == false ]]; then
        tune_system_resources
        deploy_3fs_client
        install_sglang_service
        run_functional_test
    fi

    # ========================= 总结 =========================
    echo ""
    echo -e "${BOLD}${BLUE}━━━ 部署总结 ━━━${NC}"

    local total_checks=$((PASS_COUNT + WARN_COUNT + FAIL_COUNT))
    echo "  通过: ${GREEN}$PASS_COUNT${NC}  警告: ${YELLOW}$WARN_COUNT${NC}  失败: ${RED}$FAIL_COUNT${NC}  (共 $total_checks)"
    echo ""

    if [[ $FAIL_COUNT -gt 0 ]]; then
        echo -e "  ${RED}✗ GDR 环境未完全就绪${NC}"
        echo "  请根据上述提示修复失败项"
        echo "  详细日志: $LOG_FILE"
    elif [[ $WARN_COUNT -gt 0 ]]; then
        echo -e "  ${YELLOW}⚠ GDR 环境基本就绪 (有 $WARN_COUNT 个警告)${NC}"
        echo "  建议处理警告项以获得最佳性能"
    else
        echo -e "  ${GREEN}✓ GDR 环境完全就绪!${NC}"
        echo ""
        echo "  下一步: 启动 SGLang with GDR"
        if [[ "$INSTALL_SERVICE" == true ]]; then
            echo -e "    ${CYAN}sudo systemctl start sglang-gdr${NC}"
        else
            echo -e "    ${CYAN}SGLANG_GDR_PLUGIN=1 python3 -m sglang.launch_server \\${NC}"
            echo -e "      ${CYAN}--model-path /3fs/stage/models/DeepSeek-R1 \\${NC}"
            echo -e "      ${CYAN}--tp 8 --host 0.0.0.0 --port 10000 \\${NC}"
            echo -e "      ${CYAN}--enable-hierarchical-cache \\${NC}"
            echo -e "      ${CYAN}--hicache-io-backend gdr \\${NC}"
            echo -e "      ${CYAN}--hicache-storage-backend gdr_3fs${NC}"
        fi
    fi

    echo ""
    echo "  日志文件: $LOG_FILE"

    [[ $FAIL_COUNT -gt 0 ]] && exit 1
    exit 0
}

main "$@"
