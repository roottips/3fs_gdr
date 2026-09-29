"""
RDMA 操作模块

封装基于 GPU 内存的 RDMA verbs 操作:
  - ibv_post_send (RDMA Write / RDMA Read)
  - ibv_post_recv (接收)
  - Completion Queue 轮询
  - 基于 GPU Buffer 的零拷贝传输

数据流:
  Sender:  GDRBuffer(gpu_ptr) → ibv_post_send(WR_RDMA_WRITE) → NIC DMA → 网络
  Receiver: GDRBuffer(gpu_ptr) ← NIC DMA ← ibv_post_recv → GDRBuffer(gpu_ptr)

关键: 全程 GPU 显存数据不经过 CPU 主机内存。
"""

from __future__ import annotations

import ctypes
import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Any

from .gpu_memory_manager import GDRBuffer, GPUMemoryManager

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------- #
#  ibverbs 结构体定义 (简化版)
# ----------------------------------------------------------------------- #


class ibv_sge(ctypes.Structure):
    """Scatter/Gather Entry — 描述数据缓冲区"""
    _fields_ = [
        ("addr", ctypes.c_uint64),      # GPU 设备内存地址
        ("length", ctypes.c_uint32),     # 数据长度
        ("lkey", ctypes.c_uint32),       # Local Key
    ]


class ibv_send_wr(ctypes.Structure):
    """Send Work Request"""
    pass  # 复杂结构体，通过 ctypes 手动构造


class ibv_wc(ctypes.Structure):
    """Work Completion"""
    _fields_ = [
        ("wr_id", ctypes.c_uint64),
        ("status", ctypes.c_int),
        ("opcode", ctypes.c_int),
        ("vendor_err", ctypes.c_uint32),
        ("byte_len", ctypes.c_uint32),
        ("imm_data", ctypes.c_uint32),
    ]


# IBV 操作码
class IBV_OPCODE(IntEnum):
    IBV_WR_RDMA_WRITE = 0
    IBV_WR_RDMA_READ = 1
    IBV_WR_SEND = 2
    IBV_WR_SEND_WITH_IMM = 3


# IBV WC 状态
class IBV_WC_STATUS(IntEnum):
    IBV_WC_SUCCESS = 0
    IBV_WC_LOC_LEN_QP_OP_ERR = 1
    IBV_WC_LOC_QP_OP_ERR = 2
    IBV_WC_LOC_EEC_OP_ERR = 3
    IBV_WC_LOC_PROT_ERR = 4
    IBV_WC_WR_FLUSH_ERR = 5
    IBV_WC_MW_BIND_ERR = 6
    IBV_WC_BAD_RESP_ERR = 7
    IBV_WC_RESP_TIMEOUT_ERR = 9
    IBV_WC_REM_ACCESS_ERR = 11


@dataclass
class RDMAEndpoint:
    """RDMA 端点信息"""
    lid: int               # Local ID (IB)
    qpn: int               # Queue Pair Number
    psn: int               # Packet Sequence Number
    rkey: int              # Remote Key
    remote_addr: int        # 远端 GPU 内存地址


@dataclass
class TransferResult:
    """传输结果"""
    success: bool
    bytes_transferred: int
    latency_us: float
    error: Optional[str] = None


class RDMATransport:
    """
    RDMA 传输层

    基于 GPU 内存的 RDMA verbs 操作封装。
    支持 RDMA Write (one-sided) 和 RDMA Read (one-sided) 两种模式。

    3FS 集成: 3FS 存储节点使用 RDMA Write 将数据写入客户端 GPU 内存;
              SGLang HiCache 通过 RDMA Read 从 3FS 读取 KV Cache 到 GPU。

    用法:
        transport = RDMATransport(mgr, ib_dev="mlx5_0")
        transport.connect(remote_endpoint)
        result = transport.rdma_write(
            local_buf=gdr_buf,
            remote_addr=endpoint.remote_addr,
            rkey=endpoint.rkey,
            length=1024,
        )
    """

    # IBV 常量
    IBV_QPT_RC = 2  # Reliable Connection

    def __init__(
        self,
        mem_mgr: GPUMemoryManager,
        ib_dev: str = "mlx5_0",
        cq_depth: int = 256,
        qp_depth: int = 128,
    ):
        self.mem_mgr = mem_mgr
        self.ib_dev = ib_dev
        self.cq_depth = cq_depth
        self.qp_depth = qp_depth

        self._lib_ibverbs: Optional[ctypes.CDLL] = None
        self._cq: Optional[ctypes.c_void_p] = None
        self._qp: Optional[ctypes.c_void_p] = None
        self._connected = False
        self._wr_id_counter = 0

    # ------------------------------------------------------------------ #
    #  初始化
    # ------------------------------------------------------------------ #

    def initialize(self) -> bool:
        """初始化 CQ 和 QP"""
        try:
            self._lib_ibverbs = ctypes.CDLL("libibverbs.so.1")
            self._setup_signatures()
            self._create_cq()
            self._create_qp()
            self._modify_qp_to_init()
            logger.info(
                f"RDMA Transport 初始化成功: {self.ib_dev}, "
                f"CQ depth={self.cq_depth}, QP depth={self.qp_depth}"
            )
            return True
        except Exception as e:
            logger.error(f"RDMA Transport 初始化失败: {e}")
            return False

    def _setup_signatures(self):
        lib = self._lib_ibverbs

        # ibv_create_cq
        lib.ibv_create_cq.restype = ctypes.c_void_p
        lib.ibv_create_cq.argtypes = [
            ctypes.c_void_p,   # context
            ctypes.c_int,       # cqe (depth)
            ctypes.c_void_p,   # cq_context
            ctypes.c_void_p,   # channel
            ctypes.c_int,        # comp_vector
        ]

        # ibv_create_qp
        lib.ibv_create_qp.restype = ctypes.c_void_p
        lib.ibv_create_qp.argtypes = [
            ctypes.c_void_p,   # pd
            ctypes.c_void_p,   # qp_init_attr
        ]

        # ibv_modify_qp
        lib.ibv_modify_qp.restype = ctypes.c_int
        lib.ibv_modify_qp.argtypes = [
            ctypes.c_void_p,   # qp
            ctypes.c_void_p,   # attr
            ctypes.c_int,        # attr_mask
        ]

        # ibv_post_send
        lib.ibv_post_send.restype = ctypes.c_int
        lib.ibv_post_send.argtypes = [
            ctypes.c_void_p,   # qp
            ctypes.c_void_p,   # wr
            ctypes.POINTER(ctypes.c_void_p),  # bad_wr
        ]

        # ibv_post_recv
        lib.ibv_post_recv.restype = ctypes.c_int
        lib.ibv_post_recv.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]

        # ibv_poll_cq
        lib.ibv_poll_cq.restype = ctypes.c_int
        lib.ibv_poll_cq.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ibv_wc),
        ]

    def _create_cq(self):
        """创建 Completion Queue"""
        lib = self._lib_ibverbs
        self._cq = lib.ibv_create_cq(
            self.mem_mgr._ib_context,
            self.cq_depth,
            None,
            None,
            0,
        )
        if not self._cq:
            raise RuntimeError("ibv_create_cq 失败")

    def _create_qp(self):
        """创建 Queue Pair (Reliable Connection)"""
        lib = self._lib_ibverbs

        # 构造 qp_init_attr 结构
        # 简化: 使用原始字节构造
        qp_init_attr = (ctypes.c_uint8 * 64)()

        # 设置必要字段:
        #   qp_context, send_cq, recv_cq, cap (max_send_wr, max_recv_wr, ...)
        #   qp_type = IBV_QPT_RC
        # 实际实现需按 ibv_qp_init_attr 结构体布局

        # send_cq 和 recv_cq 都用同一个
        ctypes.memmove(
            qp_init_attr,
            ctypes.addressof(ctypes.c_void_p(self._cq)),
            8,
        )  # send_cq
        ctypes.memmove(
            ctypes.addressof(qp_init_attr) + 8,
            ctypes.addressof(ctypes.c_void_p(self._cq)),
            8,
        )  # recv_cq
        struct.pack_into(
            "I", qp_init_attr, 16, self.qp_depth
        )  # max_send_wr
        struct.pack_into(
            "I", qp_init_attr, 20, self.qp_depth
        )  # max_recv_wr
        struct.pack_into("i", qp_init_attr, 40, self.IBV_QPT_RC)  # qp_type

        self._qp = lib.ibv_create_qp(
            self.mem_mgr._pd,
            ctypes.cast(qp_init_attr, ctypes.c_void_p),
        )
        if not self._qp:
            raise RuntimeError("ibv_create_qp 失败")

    def _modify_qp_to_init(self):
        """将 QP 状态修改为 INIT (必须的初始化步骤)"""
        lib = self._lib_ibverbs
        attr = (ctypes.c_uint8 * 64)()
        # 设置 port 和 access_flags
        port = 1
        struct.pack_into("H", attr, 0, port)  # port_num
        access_flags = (
            self.mem_mgr.IBV_ACCESS_LOCAL_WRITE
            | self.mem_mgr.IBV_ACCESS_REMOTE_READ
            | self.mem_mgr.IBV_ACCESS_REMOTE_WRITE
            | self.mem_mgr.IBV_ACCESS_REMOTE_ATOMIC
        )
        struct.pack_into("I", attr, 4, access_flags)  # qp_access_flags

        ret = lib.ibv_modify_qp(
            self._qp,
            ctypes.cast(attr, ctypes.c_void_p),
            0x1 | 0x8,  # IBV_QP_PORT | IBV_QP_ACCESS_FLAGS
        )
        if ret != 0:
            logger.warning(f"modify_qp INIT 返回 {ret}")

    # ------------------------------------------------------------------ #
    #  连接管理
    # ------------------------------------------------------------------ #

    def connect(self, remote: RDMAEndpoint):
        """
        连接到远端 RDMA 端点。

        执行 RTS/RTT (Ready-to-Send/Ready-to-Receive) 握手:
          INIT → RTR (Ready to Receive) → RTS (Ready to Send)
        """
        lib = self._lib_ibverbs

        # 修改 QP 到 RTR
        rtr_attr = (ctypes.c_uint8 * 128)()
        struct.pack_into("H", rtr_attr, 0, remote.lid)       # dlid
        struct.pack_into("I", rtr_attr, 4, remote.qpn)       # dest_qp_num
        struct.pack_into("I", rtr_attr, 8, remote.psn)       # rq_psn
        # 设置路径 MTU, 等

        ret = lib.ibv_modify_qp(
            self._qp,
            ctypes.cast(rtr_attr, ctypes.c_void_p),
            0x2 | 0x4 | 0x10 | 0x40,  # DEST_QPN | RQ_PSN | PATH_MTU | AV
        )
        if ret != 0:
            raise RuntimeError(f"modify_qp RTR 失败: {ret}")

        # 修改 QP 到 RTS
        rts_attr = (ctypes.c_uint8 * 32)()
        struct.pack_into("I", rts_attr, 0, remote.psn)       # sq_psn

        ret = lib.ibv_modify_qp(
            self._qp,
            ctypes.cast(rts_attr, ctypes.c_void_p),
            0x8 | 0x20,  # SQ_PSN | SQ_PSN
        )
        if ret != 0:
            raise RuntimeError(f"modify_qp RTS 失败: {ret}")

        self._connected = True
        logger.info(
            f"RDMA 连接建立: lid={remote.lid}, qpn={remote.qpn}"
        )

    def disconnect(self):
        """断开 RDMA 连接"""
        self._connected = False

    # ------------------------------------------------------------------ #
    #  RDMA 操作
    # ------------------------------------------------------------------ #

    def rdma_write(
        self,
        local_buf: GDRBuffer,
        remote_addr: int,
        rkey: int,
        length: int,
    ) -> TransferResult:
        """
        RDMA Write: 将本地 GPU 内存写入远端内存 (one-sided)。

        数据流: local_gpu_mem → NIC → 网络 → remote_mem
        远端不感知，不消耗远端 CPU。

        Args:
            local_buf:   本地 GDR 缓冲区 (GPU 内存)
            remote_addr: 远端内存地址
            rkey:        远端 Memory Region 的 rkey
            length:       传输长度

        Returns:
            TransferResult
        """
        if not self._initialized and not self.initialize():
            return TransferResult(False, 0, 0, "初始化失败")

        start = time.perf_counter()

        # 构造 SGE (Scatter/Gather Entry)
        sge = ibv_sge()
        sge.addr = local_buf.gpu_ptr     # GPU 设备内存地址
        sge.length = length
        sge.lkey = local_buf.mr_lkey

        # 构造 Send WR
        wr_id = self._wr_id_counter
        self._wr_id_counter += 1

        wr = self._build_send_wr(
            wr_id=wr_id,
            sge=sge,
            opcode=IBV_OPCODE.IBV_WR_RDMA_WRITE,
            remote_addr=remote_addr,
            rkey=rkey,
            send_flags=0x4,  # IBV_SEND_SIGNALED
        )

        # 提交 send
        lib = self._lib_ibverbs
        bad_wr = ctypes.c_void_p()
        ret = lib.ibv_post_send(self._qp, wr, ctypes.byref(bad_wr))
        if ret != 0:
            return TransferResult(False, 0, 0, f"post_send 失败: {ret}")

        # 等待完成
        wc = ibv_wc()
        polled = 0
        for _ in range(1000):  # 最多重试 1000 次
            polled = lib.ibv_poll_cq(self._cq, 1, ctypes.byref(wc))
            if polled > 0:
                break
            time.sleep(1e-5)

        end = time.perf_counter()
        latency = (end - start) * 1e6

        if polled > 0 and wc.status == IBV_WC_STATUS.IBV_WC_SUCCESS.value:
            return TransferResult(True, length, latency)
        else:
            return TransferResult(
                False, 0, latency,
                f"WC 状态: {wc.status}"
            )

    def rdma_read(
        self,
        local_buf: GDRBuffer,
        remote_addr: int,
        rkey: int,
        length: int,
    ) -> TransferResult:
        """
        RDMA Read: 从远端内存读取数据到本地 GPU 内存 (one-sided)。

        数据流: remote_mem → 网络 → NIC → local_gpu_mem

        3FS 场景: 从 3FS 存储节点读取数据到 GPU 显存。
        """
        if not self._initialized and not self.initialize():
            return TransferResult(False, 0, 0, "初始化失败")

        start = time.perf_counter()

        sge = ibv_sge()
        sge.addr = local_buf.gpu_ptr
        sge.length = length
        sge.lkey = local_buf.mr_lkey

        wr_id = self._wr_id_counter
        self._wr_id_counter += 1

        wr = self._build_send_wr(
            wr_id=wr_id,
            sge=sge,
            opcode=IBV_OPCODE.IBV_WR_RDMA_READ,
            remote_addr=remote_addr,
            rkey=rkey,
            send_flags=0x4,  # IBV_SEND_SIGNALED
        )

        lib = self._lib_ibverbs
        bad_wr = ctypes.c_void_p()
        ret = lib.ibv_post_send(self._qp, wr, ctypes.byref(bad_wr))
        if ret != 0:
            return TransferResult(False, 0, 0, f"post_send 失败: {ret}")

        wc = ibv_wc()
        polled = 0
        for _ in range(1000):
            polled = lib.ibv_poll_cq(self._cq, 1, ctypes.byref(wc))
            if polled > 0:
                break
            time.sleep(1e-5)

        end = time.perf_counter()
        latency = (end - start) * 1e6

        if polled > 0 and wc.status == IBV_WC_STATUS.IBV_WC_SUCCESS.value:
            return TransferResult(True, length, latency)
        else:
            return TransferResult(
                False, 0, latency, f"WC 状态: {wc.status}"
            )

    def _build_send_wr(
        self,
        wr_id: int,
        sge: ibv_sge,
        opcode: int,
        remote_addr: int = 0,
        rkey: int = 0,
        send_flags: int = 0,
    ) -> ctypes.c_void_p:
        """
        手动构造 ibv_send_wr 结构体的字节缓冲区。

        ibv_send_wr 内存布局 (64位):
          0-7:   wr_id (uint64)
          8-11:  next (ptr) — 取前4字节 (实际为8字节指针)
          ...
          不同平台布局不同，这里使用简化版本。
        """
        # 分配足够的内存
        buf = ctypes.create_string_buffer(128)

        # wr_id
        struct.pack_into("Q", buf, 0, wr_id)

        # opcode (int32, 偏移因平台而异)
        # 简化: 直接将 SGE 指针写入对应位置
        sge_ptr = ctypes.pointer(sge)
        struct.pack_into("P", buf, 8, ctypes.addressof(sge))  # sg_list

        # num_sge
        struct.pack_into("I", buf, 16, 1)

        # opcode
        struct.pack_into("I", buf, 20, int(opcode))

        # send_flags
        struct.pack_into("I", buf, 24, send_flags)

        # wr.rdma.remote_addr
        struct.pack_into("Q", buf, 28, remote_addr)

        # wr.rdma.rkey
        struct.pack_into("I", buf, 36, rkey)

        return ctypes.cast(buf, ctypes.c_void_p)

    def post_recv(
        self,
        local_buf: GDRBuffer,
        length: int,
    ) -> int:
        """
        投递接收请求 (用于 two-sided 通信)。

        返回 wr_id 用于追踪。
        """
        lib = self._lib_ibverbs
        wr_id = self._wr_id_counter
        self._wr_id_counter += 1

        sge = ibv_sge()
        sge.addr = local_buf.gpu_ptr
        sge.length = length
        sge.lkey = local_buf.mr_lkey

        buf = ctypes.create_string_buffer(64)
        struct.pack_into("Q", buf, 0, wr_id)
        struct.pack_into("P", buf, 8, ctypes.addressof(sge))
        struct.pack_into("I", buf, 16, 1)

        bad_wr = ctypes.c_void_p()
        ret = lib.ibv_post_recv(
            self._qp,
            ctypes.cast(buf, ctypes.c_void_p),
            ctypes.byref(bad_wr),
        )
        if ret != 0:
            logger.error(f"post_recv 失败: {ret}")
            return -1

        return wr_id

    def poll_completion(
        self,
        timeout_ms: int = 5000,
    ) -> Optional[ibv_wc]:
        """
        轮询 Completion Queue 获取完成事件。

        Args:
            timeout_ms: 超时毫秒数

        Returns:
            ibv_wc 或 None (超时)
        """
        lib = self._lib_ibverbs
        wc = ibv_wc()

        deadline = time.time() + timeout_ms / 1000
        while time.time() < deadline:
            polled = lib.ibv_poll_cq(self._cq, 1, ctypes.byref(wc))
            if polled > 0:
                return wc
            time.sleep(1e-4)

        return None

    def batch_write(
        self,
        buffers: list[tuple[GDRBuffer, int, int, int]],
    ) -> list[TransferResult]:
        """
        批量 RDMA Write (一次 post 多个 WR，一次 poll)。

        Args:
            buffers: [(local_buf, remote_addr, rkey, length), ...]

        Returns:
            每个传输的结果列表
        """
        lib = self._lib_ibverbs
        results: list[TransferResult] = []
        start = time.perf_counter()

        wr_ids = []
        wr_list = []

        for local_buf, remote_addr, rkey, length in buffers:
            sge = ibv_sge()
            sge.addr = local_buf.gpu_ptr
            sge.length = length
            sge.lkey = local_buf.mr_lkey

            wr_id = self._wr_id_counter
            self._wr_id_counter += 1
            wr_ids.append(wr_id)

            wr = self._build_send_wr(
                wr_id=wr_id,
                sge=sge,
                opcode=IBV_OPCODE.IBV_WR_RDMA_WRITE,
                remote_addr=remote_addr,
                rkey=rkey,
                send_flags=0x4,
            )
            wr_list.append(wr)

        # 批量提交
        for wr in wr_list:
            bad_wr = ctypes.c_void_p()
            lib.ibv_post_send(self._qp, wr, ctypes.byref(bad_wr))

        # 批量轮询
        expected = len(wr_ids)
        received = 0
        wc = ibv_wc()
        deadline = time.time() + 10.0

        while received < expected and time.time() < deadline:
            polled = lib.ibv_poll_cq(self._cq, 1, ctypes.byref(wc))
            if polled > 0:
                if wc.status == IBV_WC_STATUS.IBV_WC_SUCCESS.value:
                    results.append(TransferResult(True, wc.byte_len, 0))
                else:
                    results.append(TransferResult(False, 0, 0, f"WC: {wc.status}"))
                received += 1
            else:
                time.sleep(1e-5)

        end = time.perf_counter()
        total_latency = (end - start) * 1e6
        for r in results:
            r.latency_us = total_latency / max(1, len(results))

        return results


class RDMAOps:
    """
    RDMA 操作高层封装

    提供 SGLang 使用的便捷接口，封装底层 RDMATransport。
    """

    def __init__(
        self,
        gpu_id: int = 0,
        ib_dev: str = "mlx5_0",
        pool_size_gb: int = 4,
    ):
        self.mem_mgr = GPUMemoryManager(gpu_id, ib_dev, pool_size_gb)
        self.transport = RDMATransport(self.mem_mgr, ib_dev)
        self._initialized = False

    def initialize(self) -> bool:
        """初始化内存管理器和传输层"""
        if not self.mem_mgr.initialize():
            return False
        if not self.transport.initialize():
            return False
        self._initialized = True
        return True

    def transfer_to_gpu(
        self,
        remote_addr: int,
        rkey: int,
        size: int,
    ) -> TransferResult:
        """
        从远端内存 (3FS 存储) 读取数据直接到 GPU 显存。

        这是 SGLang HiCache GDR 后端的核心操作:
        3FS 存储节点 → RDMA Read → GPU 显存 (零拷贝)

        Args:
            remote_addr: 3FS 存储节点暴露的远端内存地址
            rkey:         远端 MR 的 rkey
            size:          读取字节数

        Returns:
            TransferResult
        """
        if not self._initialized and not self.initialize():
            return TransferResult(False, 0, 0, "初始化失败")

        buf = self.mem_mgr.alloc_buffer(size)
        if buf is None:
            return TransferResult(False, 0, 0, "内存分配失败")

        result = self.transport.rdma_read(
            local_buf=buf,
            remote_addr=remote_addr,
            rkey=rkey,
            length=size,
        )

        # 归还缓冲区
        self.mem_mgr.free_buffer(buf)
        return result

    def transfer_from_gpu(
        self,
        gpu_ptr: int,
        gpu_lkey: int,
        remote_addr: int,
        rkey: int,
        size: int,
    ) -> TransferResult:
        """
        从 GPU 显存写入远端内存 (3FS 存储写入路径)。

        GPU 显存 → RDMA Write → 3FS 存储节点内存 (零拷贝)
        """
        if not self._initialized and not self.initialize():
            return TransferResult(False, 0, 0, "初始化失败")

        # 使用传入的 GPU 内存（非池管理的）
        buf = GDRBuffer(
            gpu_ptr=gpu_ptr,
            size=size,
            gpu_addr=gpu_ptr,
            mr_handle=0,
            mr_lkey=gpu_lkey,
            mr_rkey=0,
        )

        return self.transport.rdma_write(
            local_buf=buf,
            remote_addr=remote_addr,
            rkey=rkey,
            length=size,
        )

    def cleanup(self):
        """清理所有资源"""
        if self.transport:
            self.transport.disconnect()
        if self.mem_mgr:
            self.mem_mgr.cleanup()
        self._initialized = False
