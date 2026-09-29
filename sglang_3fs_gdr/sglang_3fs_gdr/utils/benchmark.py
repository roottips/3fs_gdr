"""
GDR 性能基准测试工具

测试 GDR 传输性能:
  1. 延迟测试: 不同数据大小的单次传输延迟
  2. 带宽测试: 连续传输测量吞吐量
  3. 批量测试: 批量传输的延迟和吞吐
  4. 对比测试: GDR vs FUSE vs Mooncake

用法:
    python -m sglang_3fs_gdr.utils.benchmark --test latency
    python -m sglang_3fs_gdr.utils.benchmark --test bandwidth
    python -m sglang_3fs_gdr.utils.benchmark --test compare
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

from ..gdr_core.gdr_detector import GDRDetector
from ..gdr_core.gpu_memory_manager import GPUMemoryManager
from ..gdr_core.rdma_ops import RDMAOps

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    data_size_bytes: int
    num_iterations: int
    latencies_us: list = field(default_factory=list)
    bandwidths_gbps: list = field(default_factory=list)
    errors: int = 0

    @property
    def avg_latency_us(self) -> float:
        return statistics.mean(self.latencies_us) if self.latencies_us else 0

    @property
    def p50_latency_us(self) -> float:
        if not self.latencies_us:
            return 0
        return statistics.median(self.latencies_us)

    @property
    def p99_latency_us(self) -> float:
        if not self.latencies_us:
            return 0
        sorted_lat = sorted(self.latencies_us)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def avg_bandwidth_gbps(self) -> float:
        return statistics.mean(self.bandwidths_gbps) if self.bandwidths_gbps else 0

    def summary(self) -> str:
        return (
            f"[{self.test_name}] "
            f"size={self.data_size_bytes}B, "
            f"iters={self.num_iterations}, "
            f"avg={self.avg_latency_us:.1f}us, "
            f"p50={self.p50_latency_us:.1f}us, "
            f"p99={self.p99_latency_us:.1f}us, "
            f"bw={self.avg_bandwidth_gbps:.2f}GB/s, "
            f"errors={self.errors}"
        )


class GDRBenchmark:
    """GDR 性能基准测试"""

    # 测试数据大小序列
    LATENCY_SIZES = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
    BANDWIDTH_SIZES = [1048576, 4194304, 16777216, 67108864, 268435456]

    def __init__(
        self,
        gpu_id: int = 0,
        ib_dev: str = "mlx5_0",
    ):
        self.gpu_id = gpu_id
        self.ib_dev = ib_dev
        self._rdma_ops: Optional[RDMAOps] = None
        self._results: list[BenchmarkResult] = []

    def initialize(self) -> bool:
        """初始化测试环境"""
        # 检测 GDR
        detector = GDRDetector()
        report = detector.detect()
        if not report["gdr_available"]:
            logger.error(f"GDR 不可用: {report['warnings']}")
            return False

        # 初始化 RDMA
        self._rdma_ops = RDMAOps(self.gpu_id, self.ib_dev)
        if not self._rdma_ops.initialize():
            logger.error("RDMA 初始化失败")
            return False

        logger.info("基准测试环境初始化成功")
        return True

    def run_latency_test(
        self,
        sizes: Optional[list[int]] = None,
        iterations: int = 100,
    ) -> list[BenchmarkResult]:
        """
        延迟测试: 不同数据大小的单次传输延迟。

        对每个数据大小:
          1. 分配 GPU 内存
          2. 执行 RDMA Read (本地回环测试)
          3. 记录延迟
          4. 计算统计
        """
        sizes = sizes or self.LATENCY_SIZES
        results: list[BenchmarkResult] = []

        for size in sizes:
            result = BenchmarkResult(
                test_name="latency",
                data_size_bytes=size,
                num_iterations=iterations,
            )

            for _ in range(iterations):
                start = time.perf_counter()

                # 分配缓冲区并执行 RDMA 回环
                buf = self._rdma_ops.mem_mgr.alloc_buffer(size)
                if buf is None:
                    result.errors += 1
                    continue

                # 本地回环: RDMA Write to self
                transfer_result = self._rdma_ops.transport.rdma_write(
                    local_buf=buf,
                    remote_addr=buf.gpu_addr,
                    rkey=buf.mr_rkey,
                    length=size,
                )

                elapsed = (time.perf_counter() - start) * 1e6

                if transfer_result.success:
                    result.latencies_us.append(elapsed)
                    bw = size / (elapsed * 1e-6) / 1e9
                    result.bandwidths_gbps.append(bw)
                else:
                    result.errors += 1

                self._rdma_ops.mem_mgr.free_buffer(buf)

            results.append(result)
            print(result.summary())

        self._results.extend(results)
        return results

    def run_bandwidth_test(
        self,
        sizes: Optional[list[int]] = None,
        duration_seconds: int = 5,
    ) -> list[BenchmarkResult]:
        """
        带宽测试: 持续传输测量吞吐量。

        对每个数据大小:
          1. 分配 GPU 内存
          2. 持续 RDMA 传输直到超时
          3. 计算总吞吐量
        """
        sizes = sizes or self.BANDWIDTH_SIZES
        results: list[BenchmarkResult] = []

        for size in sizes:
            result = BenchmarkResult(
                test_name="bandwidth",
                data_size_bytes=size,
                num_iterations=0,
            )

            buf = self._rdma_ops.mem_mgr.alloc_buffer(size)
            if buf is None:
                result.errors = 1
                results.append(result)
                continue

            total_bytes = 0
            total_ops = 0
            deadline = time.time() + duration_seconds

            while time.time() < deadline:
                transfer_result = self._rdma_ops.transport.rdma_write(
                    local_buf=buf,
                    remote_addr=buf.gpu_addr,
                    rkey=buf.mr_rkey,
                    length=size,
                )

                if transfer_result.success:
                    total_bytes += size
                    total_ops += 1
                    result.latencies_us.append(transfer_result.latency_us)
                else:
                    result.errors += 1

            elapsed = duration_seconds
            result.num_iterations = total_ops
            bw = total_bytes / elapsed / 1e9
            result.bandwidths_gbps = [bw]

            self._rdma_ops.mem_mgr.free_buffer(buf)

            results.append(result)
            print(result.summary())

        self._results.extend(results)
        return results

    def run_batch_test(
        self,
        batch_sizes: list[int] = [1, 4, 8, 16, 32, 64],
        data_size: int = 65536,
        iterations: int = 50,
    ) -> list[BenchmarkResult]:
        """
        批量传输测试: 不同批量大小下的性能。

        对每个批量大小:
          1. 分配 batch_size 个 GPU 缓冲区
          2. 批量提交 RDMA Write
          3. 等待全部完成
          4. 记录延迟和吞吐
        """
        results: list[BenchmarkResult] = []

        for batch_size in batch_sizes:
            result = BenchmarkResult(
                test_name=f"batch_{batch_size}",
                data_size_bytes=data_size * batch_size,
                num_iterations=iterations,
            )

            for _ in range(iterations):
                # 分配批量缓冲区
                bufs = []
                for _ in range(batch_size):
                    buf = self._rdma_ops.mem_mgr.alloc_buffer(data_size)
                    if buf is None:
                        break
                    bufs.append(buf)

                if len(bufs) < batch_size:
                    result.errors += 1
                    for b in bufs:
                        self._rdma_ops.mem_mgr.free_buffer(b)
                    continue

                start = time.perf_counter()

                # 批量传输
                transfer_list = [
                    (buf, buf.gpu_addr, buf.mr_rkey, data_size)
                    for buf in bufs
                ]
                transfer_results = self._rdma_ops.transport.batch_write(
                    transfer_list
                )

                elapsed = (time.perf_counter() - start) * 1e6

                success_count = sum(1 for r in transfer_results if r.success)
                if success_count == batch_size:
                    result.latencies_us.append(elapsed)
                    bw = (data_size * batch_size) / (elapsed * 1e-6) / 1e9
                    result.bandwidths_gbps.append(bw)
                else:
                    result.errors += 1

                for b in bufs:
                    self._rdma_ops.mem_mgr.free_buffer(b)

            results.append(result)
            print(result.summary())

        self._results.extend(results)
        return results

    def get_report(self) -> dict:
        """生成完整测试报告"""
        return {
            "environment": {
                "gpu_id": self.gpu_id,
                "ib_dev": self.ib_dev,
            },
            "results": [
                {
                    "test": r.test_name,
                    "size_bytes": r.data_size_bytes,
                    "iterations": r.num_iterations,
                    "avg_latency_us": r.avg_latency_us,
                    "p50_latency_us": r.p50_latency_us,
                    "p99_latency_us": r.p99_latency_us,
                    "avg_bandwidth_gbps": r.avg_bandwidth_gbps,
                    "errors": r.errors,
                }
                for r in self._results
            ],
        }

    def cleanup(self):
        """清理测试资源"""
        if self._rdma_ops:
            self._rdma_ops.cleanup()


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="GDR 性能基准测试工具"
    )
    parser.add_argument(
        "--test", choices=["latency", "bandwidth", "batch", "all"],
        default="all",
        help="测试类型",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument("--ib-dev", default="mlx5_0", help="IB 设备名")
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="延迟测试迭代次数",
    )
    parser.add_argument(
        "--duration", type=int, default=5,
        help="带宽测试持续时间 (秒)",
    )

    args = parser.parse_args()

    bench = GDRBenchmark(args.gpu_id, args.ib_dev)
    if not bench.initialize():
        print("初始化失败，退出")
        return

    if args.test in ("latency", "all"):
        print("\n=== 延迟测试 ===")
        bench.run_latency_test(iterations=args.iterations)

    if args.test in ("bandwidth", "all"):
        print("\n=== 带宽测试 ===")
        bench.run_bandwidth_test(duration_seconds=args.duration)

    if args.test in ("batch", "all"):
        print("\n=== 批量测试 ===")
        bench.run_batch_test()

    print("\n=== 测试完成 ===")
    bench.cleanup()


if __name__ == "__main__":
    main()
