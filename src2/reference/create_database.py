import torch
import gc
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional
from contextlib import contextmanager
import numpy as np

from faiss_database import FeatureDatabase
from image_data import ImgDataSet
from log import Log
from model import PetNet50


class CreateFeatureDatabase:
    """优化的特征数据库创建器"""

    loger = Log(__qualname__)

    def __init__(self,
                 index_name: str,
                 db_path: str,
                 md_path: str,
                 img_path: str,
                 n_processes: Optional[int] = None,
                 save_interval: int = 500):
        """
        初始化特征数据库创建器

        Args:
            index_name: 索引名称
            db_path: 数据库路径
            md_path: 模型路径
            n_processes: 进程数，默认为CPU核心数的一半
            save_interval: 保存间隔
        """
        self.logger = self.loger.logger

        # 进程配置
        if n_processes is None:
            n_processes = max(1, cpu_count() // 2)
        self.n_processes: int = n_processes
        self.save_interval: int = save_interval

        self.logger.info(
            f"特征提取配置 - 进程数: {n_processes}, 批量大小: {save_interval}")

        # 初始化组件
        self.index_name: str = index_name
        self.database: FeatureDatabase = FeatureDatabase(db_path)
        self.database.clear_features(index_name)

        self.model: PetNet50 = PetNet50(md_path)
        self.dataset: ImgDataSet = ImgDataSet(self.model.transform, img_path)
        self.shards: List[Tuple[int, int, int]] = self._get_shards()

    def _get_shards(self) -> List[Tuple[int, int, int]]:
        """计算数据分片"""
        total_samples = len(self.dataset)
        shard_size = total_samples // self.n_processes
        shards = []

        for i in range(self.n_processes):
            start_idx = i * shard_size
            # 最后一个进程处理到数据集末尾
            end_idx = start_idx + shard_size if i < self.n_processes - 1 else total_samples
            shards.append((i, start_idx, end_idx))

        self.logger.info(
            f"数据分片完成: 总样本数 {total_samples}, 分片数 {self.n_processes}")
        return shards

    @contextmanager
    def _managed_executor(self):
        """上下文管理器确保执行器正确关闭"""
        executor = ProcessPoolExecutor(max_workers=self.n_processes)
        try:
            yield executor
        finally:
            executor.shutdown(wait=True)
            self.logger.info("进程池已关闭")

    @loger.performance_monitor("特征库创建")
    def create(self) -> int:
        """
        创建特征数据库

        Returns:
            总处理样本数
        """
        total_processed = 0
        failed_shards = 0

        with self._managed_executor() as executor:
            # 提交所有分片任务
            futures = [executor.submit(self._process_shard, shard)
                       for shard in self.shards]

            # 收集结果
            completed_tasks = 0
            for future in as_completed(futures):
                try:
                    shard_id, shard_count = future.result(
                        timeout=3600*5)  # 1小时超时
                    total_processed += shard_count
                    completed_tasks += 1

                    progress = (completed_tasks / len(self.shards)) * 100
                    self.logger.info(
                        f"分片 {shard_id} 完成，处理 {shard_count} 个样本，"
                        f"总进度: {completed_tasks}/{len(self.shards)} ({progress:.1f}%)"
                    )

                except TimeoutError:
                    self.logger.error("分片处理超时")
                    failed_shards += 1
                except Exception as e:
                    self.logger.error(f"分片处理失败: {e}")
                    failed_shards += 1
                finally:
                    # 强制垃圾回收
                    gc.collect()

        # 最终状态报告
        success_rate: float = ((len(self.shards) - failed_shards) /
                               len(self.shards)) * 100

        self.logger.info(
            f"特征提取完成 - 成功处理: {total_processed} 个样本, "
            f"失败分片: {failed_shards}, 成功率: {success_rate:.1f}%"
        )

        if failed_shards > 0:
            self.logger.warning(f"有 {failed_shards} 个分片处理失败，请检查日志")
        else:
            self._merge_sub_indices()

        return total_processed

    def _process_shard(self, shard: Tuple[int, int, int]) -> Tuple[int, int]:
        """
        处理单个数据分片

        Args:
            shard: (分片ID, 起始索引, 结束索引)

        Returns:
            (分片ID, 处理样本数)
        """
        shard_id, start_idx, end_idx = shard
        shard_size = end_idx - start_idx
        total_processed = 0

        self.logger.info(
            f"开始处理分片 {shard_id}: 索引 {start_idx}-{end_idx-1}，共 {shard_size} 个样本"
        )

        try:
            # 将分片拆分为小批次
            sub_batches = self._create_sub_batches(start_idx, end_idx)

            # 处理每个子批次
            for batch_idx, batch_indices in enumerate(sub_batches):
                batch_count = self._process_batch(shard_id, batch_indices)
                total_processed += batch_count
                batch_range = f"[{batch_indices[0]}:{batch_indices[1]}]"
                self.logger.info(
                    f"分片 {shard_id} 批次 {batch_idx+1}/{len(sub_batches)} {batch_range} "
                    f"处理 {batch_count} 个样本，累计 {total_processed} 个样本"
                )

        except Exception as e:
            self.logger.error(f"分片 {shard_id} 处理过程中发生错误: {e}")
            raise

        finally:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.logger.info(f"分片 {shard_id} 处理完成，共处理 {total_processed} 个样本")
        return shard_id, total_processed

    def _create_sub_batches(self, start_idx: int, end_idx: int) -> List[Tuple[int, int]]:
        """
        创建子批次索引

        Args:
            start_idx: 起始索引（包含）
            end_idx: 结束索引（不包含）

        Returns:
            批次索引列表 [(start, end), ...]

        Example:
            >>> self.save_interval = 3
            >>> self._create_sub_batches(0, 10)
            [(0, 3), (3, 6), (6, 9), (9, 10)]
        """
        # 输入验证
        if start_idx >= end_idx:
            self.logger.warning(
                f"无效的索引范围: start_idx({start_idx}) >= end_idx({end_idx})")
            return []

        if start_idx < 0:
            raise ValueError(f"起始索引不能为负数: {start_idx}")

        total_size = end_idx - start_idx
        self.logger.debug(f"创建子批次: 范围[{start_idx}:{end_idx}), 总大小{total_size}")

        # 计算批次数量
        num_batches = (total_size + self.save_interval -
                       1) // self.save_interval  # 向上取整

        sub_batches = []
        current_start = start_idx

        # 均匀分割批次
        for batch_idx in range(num_batches):
            batch_end = min(current_start + self.save_interval, end_idx)
            sub_batches.append((current_start, batch_end))
            current_start = batch_end

            self.logger.debug(
                f"批次 {batch_idx}: [{current_start-self.save_interval}:{batch_end})")

        return sub_batches

    @torch.no_grad()
    def _process_batch(self, shard_id: int, batch_indices: Tuple[int, int]) -> int:
        """
        处理单个批次

        Args:
            shard_id: 分片ID
            batch_indices: (起始索引, 结束索引)

        Returns:
            处理的样本数（当前批次的实际样本数，不是累积总数）
        """
        start_idx, end_idx = batch_indices
        batch_embeddings = []
        batch_labels = []

        try:
            # 批量处理样本
            for idx in range(start_idx, end_idx):
                if idx >= len(self.dataset):
                    break

                try:
                    # 获取样本
                    image_tensor, label = self.dataset[idx]

                    # 模型推理
                    embedding = self.model.embedding(image_tensor.unsqueeze(0))

                    batch_embeddings.append(embedding)
                    batch_labels.append(label)

                except Exception as e:
                    self.logger.error(f"样本 {idx} 处理失败: {e}")
                    continue

            # 合并批次结果
            if batch_embeddings:
                # batch_embeddings 包含的是 numpy 数组，直接堆叠
                final_embeddings = np.vstack(batch_embeddings)
                current_batch_size = len(batch_labels)

                # 为每个批次创建唯一的索引名称，避免追加模式
                sub_index_name = f"{self.index_name}_{shard_id}"

                # 保存到数据库
                _, index_file = self.database.save_features(
                    final_embeddings, batch_labels, sub_index_name
                )

                self.logger.info(
                    f"分片 {shard_id} 批次 [{start_idx}:{end_idx}]，"
                    f"处理 {current_batch_size} 个样本，文件: {index_file}，保存完成"
                )
                return current_batch_size  # 返回当前批次的实际样本数
            else:
                return 0

        except Exception as e:
            self.logger.error(f"批次处理失败 [{start_idx}:{end_idx}]: {e}")
            raise

        finally:
            # 清理批次内存
            del batch_embeddings, batch_labels
            if 'final_embeddings' in locals():
                del final_embeddings

    def _merge_sub_indices(self) -> Tuple[int, str]:
        """
        合并所有子索引为完整索引

        Returns:
            (合并后向量总数, 合并后索引文件路径)
        """

        self.logger.info(f"索引合并开始: {self.index_name}")

        # 合并索引
        file_size, merged_count, merged_file = self.database.merge_indices_streaming(
            self.index_name)

        self.logger.info(
            f"合并文件数: {file_size}, "
            f"索引合并完成: {self.index_name}, "
            f"总向量数: {merged_count}, 文件: {merged_file}"
        )


def create_feature_database(index_name: str, db_path: str, md_path: str, img_path: str,
                            n_processes: Optional[int] = None,
                            save_interval: int = 500) -> int:
    """
    创建特征数据库的便利函数

    Returns:
        总处理样本数
    """
    creator = CreateFeatureDatabase(
        index_name=index_name,
        db_path=db_path,
        md_path=md_path,
        img_path=img_path,
        n_processes=n_processes,
        save_interval=save_interval,
    )
    return creator.create()


def merge_feature_database(index_name: str, db_path: str, md_path: str, img_path: str,
                           n_processes: Optional[int] = None,
                           save_interval: int = 500) -> int:
    """
    创建特征数据库的便利函数

    Returns:
        总处理样本数
    """
    creator = CreateFeatureDatabase(
        index_name=index_name,
        db_path=db_path,
        md_path=md_path,
        img_path=img_path,
        n_processes=n_processes,
        save_interval=save_interval,
    )
    return creator._merge_sub_indices()
