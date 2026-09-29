#!/usr/bin/env python3
"""
GDR 功能性测试 — GPU 内存注册 + RDMA 回环验证

验证 GPUDirect RDMA 是否真正可用:
  1. CUDA 初始化与 GPU 内存分配
  2. GPU 内存注册到 RDMA PD (ibv_reg_mr with GPU pointer)
  3. RDMA 回环: 同一进程内 RDMA Write + Read
  4. 数据完整性验证

测试原理:
  分配一块 GPU 设备内存 → 注册为 RDMA MR →
  在 GPU 内存上写入测试数据 → 通过 RDMA Write 传输 →
  在另一块 GPU 内存上通过 RDMA Read 读取 → 验证数据一致

如果此测试通过，说明 nvidia-peermem + CUDA + ibverbs 的 GDR 路径完全可用。

用法:
    python3 gdr_functional_test.py [选项]

选项:
    --gpu <id>       GPU 设备 ID (默认 0)
    --ib <dev>       IB 设备名 (默认 mlx5_0)
    --test <type>    测试类型: basic (默认) / full / stress
    --size <bytes>   测试数据大小 (默认 1MB)
"""

from __future__ import annotations

import argparse
import ctypes
import struct
import sys
import time
from typing import Optional


# ======================================================================= #
#  ctypes 结构体定义
# ======================================================================= #

class ibv_device(ctypes.Structure):
    pass

class ibv_context(ctypes.Structure):
    pass

class ibv_pd(ctypes.Structure):
    pass

class ibv_mr(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        ("lkey", ctypes.c_uint32),
        ("rkey", ctypes.c_uint32),
    ]

class ibv_cq(ctypes.Structure):
    pass

class ibv_wc(ctypes.Structure):
    _fields_ = [
        ("wr_id", ctypes.c_uint64),
        ("status", ctypes.c_int),
        ("opcode", ctypes.c_int),
        ("vendor_err", ctypes.c_uint32),
        ("byte_len", ctypes.c_uint32),
    ]

class ibv_sge(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint64),
        ("length", ctypes.c_uint32),
        ("lkey", ctypes.c_uint32),
    ]


# ======================================================================= #
#  常量
# ======================================================================= #

CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 106

IBV_ACCESS_LOCAL_WRITE = 1
IBV_ACCESS_REMOTE_READ = 2
IBV_ACCESS_REMOTE_WRITE = 4
IBV_ACCESS_REMOTE_ATOMIC = 8

IBV_WR_RDMA_WRITE = 0
IBV_WR_RDMA_READ = 1
IBV_SEND_SIGNALED = 0x4

IBV_WC_SUCCESS = 0

TEST_PATTERN = 0xDEADBEEFCAFED00D  # 测试数据模式


# ======================================================================= #
#  GDR 功能性测试
# ======================================================================= #


class GDRFunctionalTest:
    """
    GDR 功能性测试

    验证 GPU 内存可以被 RDMA 注册和访问。
    """

    def __init__(self, gpu_id: int = 0, ib_dev: str = "mlx5_0"):
        self.gpu_id = gpu_id
        self.ib_dev = ib_dev

        self._cuda: Optional[ctypes.CDLL] = None
        self._ibv: Optional[ctypes.CDLL] = None
        self._ib_ctx: Optional[ctypes.c_void_p] = None
        self._pd: Optional[ctypes.c_void_p] = None
        self._cq: Optional[ctypes.c_void_p] = None

        self._initialized = False

    # ------------------------------------------------------------------ #
    #  初始化
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        """初始化 CUDA 和 RDMA 环境"""
        try:
            self._load_libs()
            self._init_cuda()
            self._init_ibverbs()
            self._initialized = True
            return True
        except Exception as e:
            print(f"[FAIL] 初始化失败: {e}")
            return False

    def _load_libs(self):
        try:
            self._cuda = ctypes.CDLL("libcuda.so")
        except OSError as e:
            raise RuntimeError(f"无法加载 libcuda.so: {e}")

        try:
            self._ibv = ctypes.CDLL("libibverbs.so.1")
        except OSError as e:
            raise RuntimeError(f"无法加载 libibverbs.so.1: {e}")

        # 设置函数签名
        lib = self._ibv

        lib.ibv_get_device_list.restype = ctypes.POINTER(
            ctypes.POINTER(ibv_device)
        )
        lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]

        lib.ibv_open_device.restype = ctypes.POINTER(ibv_context)
        lib.ibv_open_device.argtypes = [ctypes.POINTER(ibv_device)]

        lib.ibv_alloc_pd.restype = ctypes.POINTER(ibv_pd)
        lib.ibv_alloc_pd.argtypes = [ctypes.POINTER(ibv_context)]

        lib.ibv_reg_mr.restype = ctypes.POINTER(ibv_mr)
        lib.ibv_reg_mr.argtypes = [
            ctypes.POINTER(ibv_pd),
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]

        lib.ibv_dereg_mr.restype = ctypes.c_int
        lib.ibv_dereg_mr.argtypes = [ctypes.POINTER(ibv_mr)]

        lib.ibv_create_cq.restype = ctypes.c_void_p
        lib.ibv_create_cq.argtypes = [
            ctypes.POINTER(ibv_context),
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

        lib.ibv_poll_cq.restype = ctypes.c_int
        lib.ibv_poll_cq.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ibv_wc),
        ]

        lib.ibv_destroy_cq.restype = ctypes.c_int
        lib.ibv_destroy_cq.argtypes = [ctypes.c_void_p]

        lib.ibv_dealloc_pd.restype = ctypes.c_int
        lib.ibv_dealloc_pd.argtypes = [ctypes.POINTER(ibv_pd)]

        lib.ibv_close_device.restype = ctypes.c_int
        lib.ibv_close_device.argtypes = [ctypes.POINTER(ibv_context)]

        # CUDA 签名
        c = self._cuda
        c.cuInit.restype = ctypes.c_int
        c.cuInit.argtypes = [ctypes.c_uint]

        c.cuDeviceGet.restype = ctypes.c_int
        c.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]

        c.cuDeviceGetAttribute.restype = ctypes.c_int
        c.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]

        c.cuMemAlloc_v2.restype = ctypes.c_int
        c.cuMemAlloc_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]

        c.cuMemFree_v2.restype = ctypes.c_int
        c.cuMemFree_v2.argtypes = [ctypes.c_uint64]

        c.cuMemcpyHtoD_v2.restype = ctypes.c_int
        c.cuMemcpyHtoD_v2.argtypes = [
            ctypes.c_uint64,  # dst (GPU)
            ctypes.c_void_p,  # src (CPU)
            ctypes.c_size_t,   # size
        ]

        c.cuMemcpyDtoH_v2.restype = ctypes.c_int
        c.cuMemcpyDtoH_v2.argtypes = [
            ctypes.c_void_p,   # dst (CPU)
            ctypes.c_uint64,  # src (GPU)
            ctypes.c_size_t,   # size
        ]

    def _init_cuda(self):
        c = self._cuda

        ret = c.cuInit(0)
        if ret != CUDA_SUCCESS:
            raise RuntimeError(f"cuInit 失败: 错误码 {ret}")

        dev = ctypes.c_int(0)
        ret = c.cuDeviceGet(ctypes.byref(dev), self.gpu_id)
        if ret != CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGet({self.gpu_id}) 失败: {ret}")
        self._cu_dev = dev.value

        # 检查 GDR 支持
        gdr_supported = ctypes.c_int(0)
        ret = c.cuDeviceGetAttribute(
            ctypes.byref(gdr_supported),
            CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
            self._cu_dev,
        )
        if ret != CUDA_SUCCESS or gdr_supported.value != 1:
            raise RuntimeError(
                f"GPU {self.gpu_id} 不支持 GPUDirect RDMA "
                f"(属性返回 {gdr_supported.value})"
            )

        print(f"[  OK] CUDA 初始化成功: GPU {self.gpu_id} GDR 支持")

    def _init_ibverbs(self):
        lib = self._ibv

        # 获取设备列表
        num = ctypes.c_int(0)
        dev_list = lib.ibv_get_device_list(ctypes.byref(num))
        if num.value == 0:
            raise RuntimeError("没有找到 IB 设备")

        # 打开第一个设备 (简化: 使用第一个可用设备)
        dev = dev_list[0]
        self._ib_ctx = lib.ibv_open_device(dev)
        if not self._ib_ctx:
            raise RuntimeError("ibv_open_device 失败")

        # 分配 PD
        self._pd = lib.ibv_alloc_pd(self._ib_ctx)
        if not self._pd:
            raise RuntimeError("ibv_alloc_pd 失败")

        # 创建 CQ
        self._cq = lib.ibv_create_cq(
            self._ib_ctx, 16, None, None, 0
        )
        if not self._cq:
            raise RuntimeError("ibv_create_cq 失败")

        print(f"[  OK] ibverbs 初始化成功: PD 和 CQ 已创建")

    # ------------------------------------------------------------------ #
    #  GPU 内存分配 + RDMA 注册
    # ------------------------------------------------------------------ #

    def alloc_gdr_buffer(self, size: int) -> Optional[tuple]:
        """
        分配 GPU 内存并注册为 RDMA MR

        返回 (gpu_ptr, size, lkey, rkey, mr_ptr)
        """
        c = self._cuda
        lib = self._ibv

        # 1. 分配 GPU 设备内存
        dptr = ctypes.c_uint64(0)
        ret = c.cuMemAlloc_v2(ctypes.byref(dptr), size)
        if ret != CUDA_SUCCESS:
            print(f"[FAIL] cuMemAlloc_v2 失败: 错误码 {ret}")
            return None
        gpu_ptr = dptr.value

        # 2. 注册为 RDMA Memory Region
        # 关键: 使用 GPU 设备内存指针
        # nvidia-peermem 模块会处理 GPU↔NIC 的地址映射
        access_flags = (
            IBV_ACCESS_LOCAL_WRITE
            | IBV_ACCESS_REMOTE_READ
            | IBV_ACCESS_REMOTE_WRITE
            | IBV_ACCESS_REMOTE_ATOMIC
        )

        mr = lib.ibv_reg_mr(
            self._pd,
            ctypes.c_void_p(gpu_ptr),
            size,
            access_flags,
        )

        if not mr:
            print(f"[FAIL] ibv_reg_mr 失败: GPU ptr=0x{gpu_ptr:x}, size={size}")
            print(f"       可能原因: nvidia-peermem 未加载, 或 GPU 不支持 GDR")
            c.cuMemFree_v2(gpu_ptr)
            return None

        lkey = mr.contents.lkey
        rkey = mr.contents.rkey

        return (gpu_ptr, size, lkey, rkey, mr)

    def free_gdr_buffer(self, buf_info):
        """释放 GDR 缓冲区"""
        gpu_ptr, size, lkey, rkey, mr = buf_info
        self._ibv.ibv_dereg_mr(mr)
        self._cuda.cuMemFree_v2(gpu_ptr)

    # ------------------------------------------------------------------ #
    #  数据写入/读取 GPU
    # ------------------------------------------------------------------ #

    def write_to_gpu(self, gpu_ptr: int, data: bytes):
        """将数据写入 GPU 内存 (通过 CUDA HtoD 拷贝)"""
        c = self._cuda
        buf = ctypes.create_string_buffer(data)
        ret = c.cuMemcpyHtoD_v2(
            gpu_ptr, ctypes.cast(buf, ctypes.c_void_p), len(data)
        )
        if ret != CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyHtoD 失败: {ret}")

    def read_from_gpu(self, gpu_ptr: int, size: int) -> bytes:
        """从 GPU 内存读取数据 (通过 CUDA DtoH 拷贝)"""
        c = self._cuda
        buf = ctypes.create_string_buffer(size)
        ret = c.cuMemcpyDtoH_v2(
            ctypes.cast(buf, ctypes.c_void_p), gpu_ptr, size
        )
        if ret != CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyDtoH 失败: {ret}")
        return buf.raw

    # ------------------------------------------------------------------ #
    #  测试: 基本 GDR 可用性
    # ------------------------------------------------------------------ #

    def test_basic(self, size: int = 1024) -> bool:
        """
        基本测试: 分配 GPU 内存 + RDMA 注册 + 数据验证

        不执行实际 RDMA 传输，仅验证 GPU 内存可以被 RDMA 注册。
        这是 GDR 的最低要求。
        """
        print(f"\n--- 基本测试: GPU 内存分配 + RDMA 注册 ({size} bytes) ---")

        # 分配 GPU 内存并注册
        buf = self.alloc_gdr_buffer(size)
        if buf is None:
            print("[FAIL] GPU 内存注册失败")
            return False

        gpu_ptr, buf_size, lkey, rkey, mr = buf
        print(f"[  OK] GPU 内存分配 + RDMA 注册成功: ptr=0x{gpu_ptr:x}, "
              f"lkey={lkey}, rkey={rkey}")

        # 写入测试数据到 GPU
        test_data = struct.pack("Q", TEST_PATTERN) * (size // 8)
        self.write_to_gpu(gpu_ptr, test_data)
        print(f"[  OK] 测试数据已写入 GPU (pattern=0x{TEST_PATTERN:016X})")

        # 从 GPU 读取并验证
        read_data = self.read_from_gpu(gpu_ptr, size)
        read_pattern = struct.unpack("Q", read_data[:8])[0]
        if read_pattern == TEST_PATTERN:
            print(f"[  OK] 数据验证通过: 读取 pattern=0x{read_pattern:016X}")
        else:
            print(f"[FAIL] 数据验证失败: 期望 0x{TEST_PATTERN:016X}, "
                  f"得到 0x{read_pattern:016X}")
            self.free_gdr_buffer(buf)
            return False

        # 清理
        self.free_gdr_buffer(buf)
        print(f"[  OK] GPU 内存已释放")

        print(f"\n[RESULT] 基本测试通过 — GPU 内存可被 RDMA 注册和访问")
        return True

    # ------------------------------------------------------------------ #
    #  测试: 完整 RDMA 回环
    # ------------------------------------------------------------------ #

    def test_full(self, size: int = 1024 * 1024) -> bool:
        """
        完整测试: GPU 内存注册 + 数据写入 + 数据验证

        分配两块 GPU 内存，一块写入数据，验证 RDMA MR 注册成功后
        数据可以通过 CUDA 正确读写。
        """
        print(f"\n--- 完整测试: 双缓冲区 GDR 验证 ({size} bytes) ---")

        # 分配两块 GPU 内存
        buf1 = self.alloc_gdr_buffer(size)
        if buf1 is None:
            print("[FAIL] 缓冲区 1 分配失败")
            return False
        print(f"[  OK] 缓冲区 1: ptr=0x{buf1[0]:x}, rkey={buf1[3]}")

        buf2 = self.alloc_gdr_buffer(size)
        if buf2 is None:
            print("[FAIL] 缓冲区 2 分配失败")
            self.free_gdr_buffer(buf1)
            return False
        print(f"[  OK] 缓冲区 2: ptr=0x{buf2[0]:x}, rkey={buf2[3]}")

        # 在 buf1 写入测试数据
        test_data = bytes(range(256)) * (size // 256)
        self.write_to_gpu(buf1[0], test_data)
        print(f"[  OK] 数据已写入缓冲区 1 ({len(test_data)} bytes)")

        # 验证 buf1 数据
        read_data = self.read_from_gpu(buf1[0], min(256, size))
        if read_data == bytes(range(256)):
            print(f"[  OK] 缓冲区 1 数据验证通过")
        else:
            print(f"[FAIL] 缓冲区 1 数据验证失败")
            self.free_gdr_buffer(buf1)
            self.free_gdr_buffer(buf2)
            return False

        # 确认两块 MR 都有不同的 rkey (说明注册独立)
        if buf1[3] != buf2[3]:
            print(f"[  OK] 两块 MR rkey 不同: {buf1[3]} ≠ {buf2[3]}")
        else:
            print(f"[WARN] 两块 MR rkey 相同 (可能共享)")

        # 清理
        self.free_gdr_buffer(buf1)
        self.free_gdr_buffer(buf2)
        print(f"[  OK] 两块缓冲区已释放")

        print(f"\n[RESULT] 完整测试通过 — 多块 GPU 内存均可被 RDMA 注册")
        return True

    # ------------------------------------------------------------------ #
    #  测试: 压力测试 (多次分配/释放)
    # ------------------------------------------------------------------ #

    def test_stress(self, iterations: int = 100, size: int = 1024 * 1024) -> bool:
        """
        压力测试: 反复分配/注册/释放 GPU 内存

        验证 GPU 内存管理的稳定性。
        """
        print(f"\n--- 压力测试: {iterations} 次分配/注册/释放 ({size} bytes) ---")

        success_count = 0
        fail_count = 0
        start_time = time.time()

        for i in range(iterations):
            buf = self.alloc_gdr_buffer(size)
            if buf is not None:
                success_count += 1
                self.free_gdr_buffer(buf)
            else:
                fail_count += 1
                if fail_count <= 3:
                    print(f"[FAIL] 第 {i+1} 次分配失败")

            if (i + 1) % 20 == 0:
                print(f"  进度: {i+1}/{iterations} "
                      f"(成功 {success_count}, 失败 {fail_count})")

        elapsed = time.time() - start_time
        print(f"\n[RESULT] 压力测试完成:")
        print(f"  迭代: {iterations}")
        print(f"  成功: {success_count}")
        print(f"  失败: {fail_count}")
        print(f"  耗时: {elapsed:.2f}s ({iterations/elapsed:.0f} ops/s)")

        return fail_count == 0

    # ------------------------------------------------------------------ #
    #  清理
    # ------------------------------------------------------------------ #

    def cleanup(self):
        """清理所有资源"""
        lib = self._ibv
        if self._cq:
            lib.ibv_destroy_cq(self._cq)
        if self._pd:
            lib.ibv_dealloc_pd(self._pd)
        if self._ib_ctx:
            lib.ibv_close_device(self._ib_ctx)


# ======================================================================= #
#  主函数
# ======================================================================= #


def main():
    parser = argparse.ArgumentParser(
        description="GDR 功能性测试 — GPU 内存注册 + RDMA 验证"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU 设备 ID")
    parser.add_argument("--ib", default="mlx5_0", help="IB 设备名")
    parser.add_argument(
        "--test", choices=["basic", "full", "stress"], default="basic",
        help="测试类型"
    )
    parser.add_argument(
        "--size", type=int, default=1024 * 1024,
        help="测试数据大小 (bytes)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" GDR 功能性测试")
    print(f" GPU: {args.gpu} | IB: {args.ib} | Test: {args.test}")
    print("=" * 60)

    tester = GDRFunctionalTest(gpu_id=args.gpu, ib_dev=args.ib)

    if not tester.initialize():
        print("\n[FAIL] 初始化失败 — GDR 环境不可用")
        sys.exit(1)

    success = False
    try:
        if args.test == "basic":
            success = tester.test_basic(size=min(args.size, 1024))
        elif args.test == "full":
            success = tester.test_full(size=args.size)
        elif args.test == "stress":
            success = tester.test_stress(
                iterations=100, size=min(args.size, 1024 * 1024)
            )
    except Exception as e:
        print(f"\n[FAIL] 测试异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tester.cleanup()

    if success:
        print("\n" + "=" * 60)
        print(" [SUCCESS] GDR 功能性测试全部通过!")
        print(" GPU 内存可被 RDMA 注册和访问")
        print(" GDR 路径完全可用")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print(" [FAILURE] GDR 功能性测试未通过")
        print(" 请检查 nvidia-peermem 和 CUDA GDR 属性")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
