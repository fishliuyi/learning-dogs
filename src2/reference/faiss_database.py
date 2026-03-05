import faiss
import pickle
import torch
import re

from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Union
from contextlib import contextmanager
import numpy as np

from log import Log


class FeatureDatabase:
    """优化的FAISS特征数据库管理器"""

    # 类常量定义
    MIN_TRAIN_SIZE = 1000
    MAX_TRAIN_SIZE = 50000
    BATCH_SIZE_DEFAULT = 10000
    MIN_NLIST = 100
    MAX_NLIST = 4000

    loger = Log(__qualname__)

    def __init__(self, db_path: str = "data/features", clear_existing: bool = False):
        """
        初始化特征数据库

        Args:
            db_path: 数据库存储路径
            clear_existing: 是否清空现有数据
        """
        self.logger = self.loger.logger
        self.db_path: Path = Path(db_path).resolve()
        self.db_path.mkdir(parents=True, exist_ok=True)

        if clear_existing:
            self.clear_all_features()

        self.logger.info(f"特征数据库初始化完成: {self.db_path}")

    def _validate_embeddings(self, embeddings: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """验证并转换嵌入向量"""
        try:
            # 转换为numpy数组
            if isinstance(embeddings, torch.Tensor):
                embeddings_np = embeddings.detach().cpu().numpy()
            else:
                embeddings_np = np.asarray(embeddings)

            # 验证数组有效性
            if embeddings_np.ndim != 2:
                raise ValueError(f"嵌入向量必须是2D数组，当前维度: {embeddings_np.ndim}")

            if embeddings_np.shape[0] == 0:
                raise ValueError("嵌入向量不能为空")

            if not np.isfinite(embeddings_np).all():
                raise ValueError("嵌入向量包含无效数值(NaN/Inf)")

            return embeddings_np.astype(np.float32)

        except Exception as e:
            self.logger.error(f"嵌入向量验证失败: {e}")
            raise ValueError(f"无效的嵌入向量格式: {e}") from e

    def _calculate_optimal_train_size(self, n_total: int) -> int:
        """计算最优训练样本数"""
        if n_total < 1000:
            return n_total  # 小数据集全部训练
        elif n_total < 100000:
            return max(self.MIN_TRAIN_SIZE, n_total // 20)  # 中等数据集
        else:
            return min(self.MAX_TRAIN_SIZE, n_total // 50)  # 大数据集

    def _create_ivf_index(self, dim: int, n_total: int) -> faiss.Index:
        """创建优化的IVF索引"""
        # 智能计算聚类中心数
        nlist = min(self.MAX_NLIST, max(self.MIN_NLIST, int(np.sqrt(n_total))))

        # 创建量化器
        quantizer = faiss.IndexFlatL2(dim)

        # 创建IVF索引
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

        self.logger.info(f"创建 IndexIVFFlat 索引，维度: {dim}, 聚类中心数: {nlist}")
        return index

    def _create_optimized_index(self, dim: int, n_total: int) -> faiss.Index:
        """根据数据规模自动选择最优FAISS索引"""

        if n_total < 5000:
            # 超小规模数据：使用精确搜索
            index = faiss.IndexFlatL2(dim)
            self.logger.info(
                "创建 IndexFlatL2 (精确搜索)，为 {n_total} 个 {dim}维向量选择索引类型")
        elif n_total < 500000:
            # 中小规模数据：使用IVF索引
            index = self._create_ivf_index(dim, n_total)
        else:
            # 大规模数据：使用HNSW索引
            index = faiss.IndexHNSWFlat(dim, 32)
            self.logger.info(
                "创建 IndexHNSWFlat (近似搜索)，为 {n_total} 个 {dim}维向量选择索引类型")

        return index

    def save_features(self, embeddings: Union[torch.Tensor, np.ndarray],
                      labels: List[str],
                      index_name: str = "reference",
                      validate: bool = True) -> Tuple[int, str]:
        """
        保存特征向量到FAISS索引

        Args:
            embeddings: 特征向量
            labels: 对应标签列表
            index_name: 索引名称
            validate: 是否验证数据
            append: 是否追加到现有索引（默认False，即覆盖模式）

        Returns:
            (特征向量数量, 索引文件路径)
        """
        try:
            # 数据验证
            if validate:
                embeddings_np = self._validate_embeddings(embeddings)
                if len(embeddings_np) != len(labels):
                    raise ValueError(
                        f"嵌入向量数量({len(embeddings_np)})与标签数量({len(labels)})不匹配")
            else:
                embeddings_np = embeddings if isinstance(
                    embeddings, np.ndarray) else embeddings.cpu().numpy()

            dim = embeddings_np.shape[1]
            n_total = embeddings_np.shape[0]

            # 初始化训练大小
            train_size = 0

            if self._index_exists(index_name):
                # 追加模式：加载现有索引
                index, existing_labels = self.load_features(index_name)

                # 验证维度一致性
                if index.d != dim:
                    raise ValueError(f"维度不匹配：现有索引维度 {index.d}，新数据维度 {dim}")

                # 添加新向量到现有索引
                index.add(embeddings_np)
                all_labels = existing_labels + labels
                final_count = index.ntotal

                self.logger.info(f"向现有索引追加 {n_total} 个向量，总向量数: {final_count}")
            else:
                # 创建新索引
                index = self._create_optimized_index(dim, n_total)

                # 训练索引（对于需要训练的索引类型）
                if hasattr(index, 'train') and not index.is_trained:
                    train_size = self._calculate_optimal_train_size(n_total)
                    train_vectors = embeddings_np[:train_size]
                    index.train(train_vectors)
                    self.logger.info(f"索引训练完成，使用 {train_size} 个样本")

                # 添加向量到索引
                index.add(embeddings_np)
                all_labels = labels
                final_count = n_total

                self.logger.info(f"创建新索引并添加 {n_total} 个向量")

            # 生成文件路径
            index_file = self.db_path / f"{index_name}.index"
            metadata_file = self.db_path / f"{index_name}_metadata.pkl"

            # 保存索引
            faiss.write_index(index, str(index_file))

            # 保存增强元数据
            metadata = {
                "labels": all_labels,
                "dimensions": dim,
                "total_vectors": final_count,
                "index_type": type(index).__name__,
                "creation_time": datetime.now().isoformat(),
                "train_size": train_size,
                "last_update": datetime.now().isoformat()
            }

            with open(metadata_file, "wb") as f:
                pickle.dump(metadata, f)

            self.logger.info(f"特征保存成功: {index_file}, 向量数: {final_count}")
            return final_count, str(index_file)

        except Exception as e:
            self.logger.error(f"保存特征失败: {e}")
            raise

    def load_features(self, index_name: str = "reference") -> Tuple[faiss.Index, List[str]]:
        """
        加载FAISS索引和元数据

        Args:
            index_name: 索引名称

        Returns:
            (索引对象, 标签列表)
        """
        index_file = self.db_path / f"{index_name}.index"
        metadata_file = self.db_path / f"{index_name}_metadata.pkl"

        # 验证文件存在性
        if not index_file.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_file}")

        try:
            # 加载索引
            index = faiss.read_index(str(index_file))

            # 加载元数据
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)

            self.logger.info(f"特征加载成功: {index_file}, 向量数: {index.ntotal}")
            return index, metadata["labels"]

        except Exception as e:
            self.logger.error(f"加载特征失败: {e}")
            raise

    def clear_features(self, index_name: str = "reference") -> None:
        """清除指定名称的特征数据库"""
        index_file = self.db_path / f"{index_name}.index"
        metadata_file = self.db_path / f"{index_name}_metadata.pkl"

        deleted_files = []

        if index_file.exists():
            index_file.unlink()
            deleted_files.append(str(index_file))

        if metadata_file.exists():
            metadata_file.unlink()
            deleted_files.append(str(metadata_file))

        if deleted_files:
            self.logger.info(f"已删除特征文件: {', '.join(deleted_files)}")
        else:
            self.logger.info(f"未找到要删除的特征文件: {index_name}")

    def clear_all_features(self) -> None:
        """清除所有特征数据库"""
        deleted_files = []

        # 删除所有.index文件
        for index_file in self.db_path.glob("*.index"):
            index_file.unlink()
            deleted_files.append(str(index_file))

        # 删除所有_metadata.pkl文件
        for metadata_file in self.db_path.glob("*_metadata.pkl"):
            metadata_file.unlink()
            deleted_files.append(str(metadata_file))

        if deleted_files:
            self.logger.info(f"已删除 {len(deleted_files)} 个特征文件")
        else:
            self.logger.info("未找到任何特征文件")

    def get_feature_count(self, index_name: str = "reference") -> int:
        """获取特征数据库中特征数量"""
        index_file = self.db_path / f"{index_name}.index"

        if not index_file.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_file}")

        try:
            index = faiss.read_index(str(index_file))
            count = index.ntotal
            self.logger.info(f"索引 {index_name} 包含 {count} 个特征向量")
            return count
        except Exception as e:
            raise RuntimeError(f"无法读取索引 {index_name}: {e}")

    def get_index_info(self, index_name: str = "reference") -> dict:
        """获取索引详细信息"""
        index_file = self.db_path / f"{index_name}.index"
        metadata_file = self.db_path / f"{index_name}_metadata.pkl"

        if not (index_file.exists() and metadata_file.exists()):
            raise FileNotFoundError(f"索引或元数据文件不存在: {index_name}")

        try:
            # 读取索引信息
            index = faiss.read_index(str(index_file))

            # 读取元数据
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)

            return {
                "index_name": index_name,
                "index_type": metadata.get("index_type", "Unknown"),
                "dimensions": metadata.get("dimensions", 0),
                "total_vectors": index.ntotal,
                "creation_time": metadata.get("creation_time", "Unknown"),
                "train_size": metadata.get("train_size", 0)
            }
        except Exception as e:
            raise RuntimeError(f"无法获取索引信息 {index_name}: {e}")

    def get_sub_indices(self, base_index_name: str = "reference") -> List[str]:
        """
        获取指定基础名称的所有子索引

        Args:
            base_index_name: 基础索引名称

        Returns:
            子索引名称列表
        """
        sub_indices = []
        pattern = f"{base_index_name}_*"

        for index_file in self.db_path.glob(f"{pattern}.index"):
            sub_name = index_file.stem
            sub_indices.append(sub_name)

        # 按数字顺序排序
        sub_indices.sort(
            key=lambda x: self._extract_shard_id(x, base_index_name))
        return sub_indices

    def _index_exists(self, index_name: str) -> bool:
        """检查索引是否已存在"""
        index_file = self.db_path / f"{index_name}.index"
        metadata_file = self.db_path / f"{index_name}_metadata.pkl"
        return index_file.exists() and metadata_file.exists()

    def _extract_shard_id(self, index_name: str, base_name: str) -> int:
        """从索引名称中提取分片ID"""
        try:
            # 移除基础名称和下划线
            suffix = index_name[len(base_name) + 1:]  # +1 for underscore
            match = re.search(r'^(\d+)', suffix)
            return int(match.group(1)) if match else 0
        except (ValueError, AttributeError, TypeError) as e:  # 优化：明确异常类型
            self.logger.error(f"解析分片ID失败: {e}")
            return 0

    @loger.performance_monitor("索引合并")
    def merge_indices_streaming(self, index_name: str = "reference", batch_size: int = 10000) -> Tuple[int, int, str]:
        """
        流式合并多个子索引文件，避免内存溢出

        Args:
            index_name: 基础索引名称
            batch_size: 每批处理的向量数量，默认10000

        Returns:
            (子索引数量, 合并后向量总数, 索引文件路径)

        Raises:
            ValueError: 当没有找到子索引时
            RuntimeError: 当合并过程中出现错误时
        """
        sub_index_names: List[str] = self.get_sub_indices(index_name)

        if not sub_index_names:
            raise ValueError("子索引名称列表不能为空")

        self.logger.info(f"开始流式合并 {len(sub_index_names)} 个子索引到 {index_name}")

        # 使用上下文管理确保资源正确释放
        merged_index = None
        try:
            # 第一步：统计总向量数和维度
            total_vectors = 0
            dimensions = None

            for sub_name in sub_index_names:
                sub_index, _ = self.load_features(sub_name)
                if dimensions is None:
                    # 采样获取维度信息
                    sample_vector = sub_index.reconstruct(0)
                    dimensions = len(sample_vector)
                total_vectors += sub_index.ntotal

            self.logger.info(f"总计向量数: {total_vectors}, 维度: {dimensions}")

            # 第二步：分批合并处理
            merged_index = self._create_optimized_index(dimensions, total_vectors)
            
            # 检查并训练索引（如果需要）
            if hasattr(merged_index, 'train') and not merged_index.is_trained:
                self.logger.info("检测到需要训练的索引类型，准备训练数据...")
                # 收集训练数据 - 从第一个子索引中采样
                first_sub_index, _ = self.load_features(sub_index_names[0])
                train_size = min(self._calculate_optimal_train_size(
                    total_vectors), first_sub_index.ntotal)
                # 采样训练向量
                train_vectors = first_sub_index.reconstruct_n(0, train_size)
                merged_index.train(train_vectors)
                self.logger.info(f"索引训练完成，使用 {train_size} 个样本")

            all_labels = []
            processed_vectors = 0

            for sub_name in sub_index_names:
                sub_index, sub_labels = self.load_features(sub_name)

                # 分批重建向量以控制内存使用
                ntotal = sub_index.ntotal
                for start_idx in range(0, ntotal, batch_size):
                    end_idx = min(start_idx + batch_size, ntotal)
                    try:
                        batch_vectors = sub_index.reconstruct_n(
                            start_idx, end_idx - start_idx)

                        # 添加到合并索引
                        merged_index.add(batch_vectors)
                        processed_vectors += len(batch_vectors)

                        self.logger.debug(
                            f"处理进度: {processed_vectors}/{total_vectors}")
                    except Exception as e:
                        self.logger.error(
                            f"处理批次 [{start_idx}:{end_idx}] 失败: {e}")
                        raise

                all_labels.extend(sub_labels)

            # 第三步：保存合并结果
            index_file = self.db_path / f"{index_name}.index"
            metadata_file = self.db_path / f"{index_name}_metadata.pkl"

            # 原子性保存：先保存索引再保存元数据
            faiss.write_index(merged_index, str(index_file))

            metadata = {
                "labels": all_labels,
                "dimensions": dimensions,
                "total_vectors": total_vectors,
                "index_type": type(merged_index).__name__,
                "creation_time": datetime.now().isoformat()
            }

            with open(metadata_file, "wb") as f:
                pickle.dump(metadata, f)

            self.logger.info(f"流式合并完成: {index_name}, 总向量数: {total_vectors}")

            # 清理子索引
            self._cleanup_sub_indices(sub_index_names)

            return len(sub_index_names), total_vectors, str(index_file)

        except Exception as e:
            self.logger.error(f"流式合并失败: {e}")
            # 如果有部分写入，考虑回滚
            self._rollback_partial_merge(index_name)
            raise
        finally:
            # 确保FAISS索引被正确释放
            if merged_index is not None:
                del merged_index

    def _rollback_partial_merge(self, index_name: str) -> None:
        """
        回滚部分完成的合并操作

        Args:
            index_name: 索引名称
        """
        try:
            # 删除可能创建的部分文件
            partial_files = [
                self.db_path / f"{index_name}.index",
                self.db_path / f"{index_name}_metadata.pkl"
            ]

            for file_path in partial_files:
                if file_path.exists():
                    file_path.unlink()
                    self.logger.info(f"回滚删除文件: {file_path}")

        except Exception as e:
            self.logger.warning(f"回滚操作失败: {e}")

    def validate_merge_consistency(self, original_counts: List[int], merged_count: int) -> bool:
        """
        验证合并前后向量数量一致性

        Args:
            original_counts: 原始各子索引的向量数量列表
            merged_count: 合并后的向量总数

        Returns:
            是否一致

        Example:
            >>> db.validate_merge_consistency([1000, 2000, 1500], 4500)
            True
        """
        expected_total = sum(original_counts)
        is_consistent = expected_total == merged_count

        if not is_consistent:
            self.logger.warning(
                f"合并数量不一致: 期望 {expected_total}, 实际 {merged_count}")

        return is_consistent

    def get_merge_progress(self, index_name: str) -> dict:
        """
        获取合并进度信息

        Args:
            index_name: 基础索引名称

        Returns:
            包含合并进度信息的字典
            {
                "status": "ready" | "no_sub_indices" | "error",
                "sub_count": int,
                "total_vectors": int,
                "sub_indices": List[dict]
            }
        """
        sub_indices = self.get_sub_indices(index_name)
        if not sub_indices:
            return {"status": "no_sub_indices", "sub_count": 0}

        total_vectors = 0
        sub_info = []

        for sub_name in sub_indices:
            try:
                info = self.get_index_info(sub_name)
                total_vectors += info["total_vectors"]
                sub_info.append(info)
            except Exception as e:
                self.logger.warning(f"无法获取子索引 {sub_name} 信息: {e}")

        return {
            "status": "ready",
            "sub_count": len(sub_indices),
            "total_vectors": total_vectors,
            "sub_indices": sub_info
        }

    def _cleanup_sub_indices(self, sub_index_names: List[str]) -> None:
        """清理子索引文件"""
        cleaned_count = 0
        for sub_name in sub_index_names:
            try:
                self.clear_features(sub_name)
                cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"清理子索引 {sub_name} 失败: {e}")

        if cleaned_count > 0:
            self.logger.info(f"已清理 {cleaned_count} 个子索引文件")

    def append_features(self, embeddings: Union[torch.Tensor, np.ndarray],
                        labels: List[str],
                        index_name: str = "reference",
                        validate: bool = True) -> Tuple[int, str]:
        """
        追加特征向量到现有FAISS索引（便利方法）

        Args:
            embeddings: 特征向量
            labels: 对应标签列表
            index_name: 索引名称
            validate: 是否验证数据

        Returns:
            (更新后的总向量数量, 索引文件路径)
        """
        return self.save_features(embeddings, labels, index_name, validate)

    @contextmanager
    def managed_index(self, index_name: str = "reference"):
        """上下文管理器确保索引正确加载和释放"""
        index, labels = self.load_features(index_name)
        try:
            yield index, labels
        finally:
            # FAISS索引会自动垃圾回收
            del index
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# 便利函数
def create_feature_db(db_path: str = "data/features", clear_existing: bool = False) -> FeatureDatabase:
    """创建特征数据库实例的便利函数"""
    return FeatureDatabase(db_path, clear_existing)


def list_available_indexes(db_path: str = "data/features") -> List[str]:
    """列出可用的索引名称"""
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    indexes = []
    for index_file in db_path.glob("*.index"):
        index_name = index_file.stem
        indexes.append(index_name)
    return indexes
