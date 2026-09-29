"""
SGLang GDR 集成入口点

在 SGLang 启动时自动注册 GDR 后端。
通过 monkey-patch 方式注入 SGLang 的 HiCache 和通信层。

用法:
    # 方式1: 作为 SGLang 插件导入
    import sglang_3fs_gdr  # 在 SGLang 启动脚本中

    # 方式2: 通过环境变量自动加载
    # SGLANG_GDR_PLUGIN=1 python -m sglang.launch_server ...

    # 方式3: 直接运行
    python -m sglang_3fs_gdr.scripts.sglang_gdr_entrypoint
"""

import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)


def register_gdr_backends():
    """
    注册 GDR 后端到 SGLang。

    在 SGLang 的 HiCache 和通信层中注入 GDR 支持:

    1. HiCache 存储后端:
       注册 "gdr_3fs" 后端 → GDRStorageBackend

    2. 通信层 Relay:
       注册 "gdr" 后端 → GDRRelayBackend

    3. I/O 后端:
       注册 "gdr" 和 "gdr_async" → GDRIOBackend

    4. Router 扩展:
       在传输路由中增加 GDR 选项
    """
    from sglang_3fs_gdr.hicache.gdr_storage_backend import (
        GDRStorageBackend, GDRStorageConfig,
    )
    from sglang_3fs_gdr.hicache.gdr_io_backend import (
        GDRIOBackend, GDRIOConfig, IOBackendType,
    )
    from sglang_3fs_gdr.comm.gdr_transport import GDRRelayBackend, GDRTransportConfig
    from sglang_3fs_gdr.utils.config import load_config

    # 加载配置
    config_path = os.environ.get("GDR_CONFIG_PATH")
    gdr_config = load_config(config_path)

    if not gdr_config.enable_gdr:
        logger.info("GDR 已禁用，跳过注册")
        return False

    # 1. 注册 HiCache 存储后端
    storage_config = GDRStorageConfig(
        storage_prefix=gdr_config.hicache_storage_prefix,
        enable_gdr=gdr_config.enable_gdr,
        gpu_id=gdr_config.gpu_id,
        ib_dev=gdr_config.ib_dev,
        pool_size_gb=gdr_config.pool_size_gb,
        write_policy=gdr_config.hicache_write_policy,
    )
    storage_backend = GDRStorageBackend(storage_config)
    if not storage_backend.initialize():
        logger.error("GDR 存储后端初始化失败")
        return False

    # 尝试注册到 SGLang HiCache
    try:
        from sglang.srt.hicache import StorageBackendRegistry
        StorageBackendRegistry.register("gdr_3fs", storage_backend)
        logger.info("HiCache GDR 存储后端注册成功")
    except ImportError:
        # SGLang 未安装或版本不兼容，使用 monkey-patch
        logger.warning("SGLang HiCache 模块未找到，使用 monkey-patch")
        _monkeypatch_hicache(storage_backend)

    # 2. 注册通信层 Relay 后端
    transport_config = GDRTransportConfig(
        gpu_id=gdr_config.gpu_id,
        ib_dev=gdr_config.ib_dev,
        pool_size_gb=gdr_config.pool_size_gb,
        cq_depth=gdr_config.comm_cq_depth,
        qp_depth=gdr_config.comm_qp_depth,
    )
    relay_backend = GDRRelayBackend(transport_config)
    if relay_backend.initialize():
        try:
            from sglang_omni.relay import backend_registry
            backend_registry.register("gdr", relay_backend)
            logger.info("通信层 GDR Relay 后端注册成功")
        except ImportError:
            logger.warning("sglang_omni 未安装，跳过通信层注册")
    else:
        logger.warning("GDR Relay 后端初始化失败，跳过通信层注册")

    # 3. 注册 I/O 后端
    io_config = GDRIOConfig(
        backend_type=IOBackendType.GDR if gdr_config.hicache_io_backend == "gdr"
        else IOBackendType.GDR_ASYNC,
        io_depth=gdr_config.io_depth,
        batch_size=gdr_config.batch_size,
    )
    try:
        usrbio_ext = storage_backend.client._usrbio
        io_backend = GDRIOBackend(io_config, usrbio_ext)
        if io_backend.initialize():
            logger.info("GDR I/O 后端注册成功")
    except Exception as e:
        logger.warning(f"GDR I/O 后端初始化失败: {e}")

    logger.info("GDR 后端注册完成")
    return True


def _monkeypatch_hicache(storage_backend):
    """Monkey-patch SGLang HiCache (兼容旧版本)"""
    try:
        import sglang.srt.hicache.storage as hicache_storage

        # 保存原始 __init__
        original_init = hicache_storage.StorageBackend.__init__

        def patched_create_backend(backend_name, *args, **kwargs):
            if backend_name in ("gdr_3fs", "gdr"):
                return storage_backend
            return original_init(backend_name, *args, **kwargs)

        hicache_storage.create_storage_backend = patched_create_backend
        logger.info("HiCache monkey-patch 成功")
    except ImportError:
        logger.error("无法 monkey-patch HiCache: 模块未找到")


def auto_register():
    """自动注册 (通过环境变量触发)"""
    if os.environ.get("SGLANG_GDR_PLUGIN", "").lower() in ("1", "true", "yes"):
        logging.basicConfig(level=logging.INFO)
        register_gdr_backends()


# 模块导入时自动注册
auto_register()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = register_gdr_backends()
    if success:
        print("GDR 后端注册成功")
    else:
        print("GDR 后端注册失败")
        sys.exit(1)
