
import sys
import time
import torch
import psutil

from pathlib import Path

# 添加项目路径到Python路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "reference"))
from reference import Log, create_feature_database, merge_feature_database

def get_system_stats():
    """获取系统资源使用情况"""
    process = psutil.Process()
    return {
        'cpu_percent': process.cpu_percent(),
        'memory_mb': process.memory_info().rss / 1024 / 1024,
        'gpu_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
    }


def main():
    logger = Log("init_features").logger

    # 系统信息
    logger.info(f"CPU核心数: {psutil.cpu_count()}")
    logger.info(
        f"总内存: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(
            f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.1f} GB")

    logger.info("初始化特征数据库...")

    start_time = time.time()
    start_stats = get_system_stats()

    index = "pet"
    db_path = "data/features"
    md_path = "data/TsinghuaDogs/model/proxynca-resnet50.pth"
    img_path = "data/TsinghuaDogs/high-resolution"
    n_process = None
    batch_size = 500
    total = create_feature_database(
        index, db_path, md_path, img_path, n_process, batch_size)

    end_time = time.time()
    end_stats = get_system_stats()

    # 性能报告
    duration = end_time - start_time
    memory_used = end_stats['memory_mb'] - start_stats['memory_mb']

    logger.info("=" * 50)
    logger.info("性能统计报告:")
    logger.info(f"  处理样本数: {total}")
    logger.info(f"  总耗时: {duration:.2f} 秒")
    logger.info(f"  平均每个样本: {duration/total*1000:.2f} 毫秒")
    logger.info(f"  内存使用增加: {memory_used:.1f} MB")
    if torch.cuda.is_available():
        logger.info(f"  GPU内存峰值: {end_stats['gpu_memory_mb']:.1f} MB")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
