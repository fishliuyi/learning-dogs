import os
import torch
from pathlib import Path
from torchvision.datasets import ImageFolder

from natsort import natsorted
from PIL import Image, ImageFile
from typing import Dict, Any, Optional, Callable, Tuple, List
from log import Log
from datetime import datetime

# 允许加载截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImgDataSet(torch.utils.data.Dataset):
    """优化的图像数据集类"""

    logger = Log(__qualname__).logger

    def __init__(self, transform: Optional[Callable] = None,
                 labeled_folders: bool = True,
                 images_dir: str = 'data/TsinghuaDogs/high-resolution'):
        """
        初始化图像数据集

        Args:
            transform: 图像变换函数
            labeled_folders: 是否使用文件夹名作为标签
            images_dir: 图像数据目录路径
        """
        self.transform = transform
        self.images_dir: Path = Path(images_dir).resolve()

        if not self.images_dir.exists():
            raise ValueError(f"数据集路径不存在: {self.images_dir}")

        # 支持的图像格式
        self.allowed_extensions: Tuple[str, ...] = (
            '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

        # 是否使用文件夹标签
        self.labeled: bool = labeled_folders
                
        # 缓存配置
        self.use_cache: bool = True  # 启用内存缓存
        self.image_cache: Dict[int, Image.Image] = {}  # 图像缓存
        self.cache_hit_count: int = 0
        self.cache_miss_count: int = 0
        
        # 预加载样本列表
        self.samples: List[Tuple[str, str]] = self._get_all_samples()
        self.logger.info(f"完成采集样本总数：{len(self.samples)}")
                
        # 预加载所有图像到内存（利用 64GB 内存优势）
        if self.use_cache:
            self._preload_all_images()

        # 统计信息
        self._log_dataset_statistics()
    
    def _preload_all_images(self):
        """
        预加载所有图像到内存（利用大内存优势）
        一次性加载所有图像，避免训练时重复读取磁盘
        """
        self.logger.info(f"开始预加载 {len(self.samples)} 张图像到内存...")
        start_time = datetime.now()
        
        loaded_count = 0
        failed_count = 0
        
        for idx, (image_path, label) in enumerate(self.samples):
            try:
                # 加载图像到缓存
                image = Image.open(image_path).convert('RGB')
                self.image_cache[idx] = image
                loaded_count += 1
                
                # 每加载 1000 张显示一次进度
                if (idx + 1) % 1000 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    self.logger.info(f"已预加载 {idx+1}/{len(self.samples)} 张图像 "
                                   f"({elapsed:.1f}s, {failed_count} 失败)")
                    
            except Exception as e:
                self.logger.warning(f"无法加载图像 {image_path}: {e}")
                failed_count += 1
        
        elapsed = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"✓ 预加载完成：成功 {loaded_count} 张，失败 {failed_count} 张，耗时 {elapsed:.1f}s")
        self.logger.info(f"  缓存大小：{len(self.image_cache)} 张图像")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """
        获取单个样本（从缓存中读取）

        Args:
            index: 样本索引

        Returns:
            图像张量和标签
        """
        try:
            # 从缓存获取图像
            if self.use_cache and index in self.image_cache:
                self.cache_hit_count += 1
                image = self.image_cache[index].copy()  # 复制一份，避免修改原图
            else:
                # 缓存未命中，从磁盘读取
                self.cache_miss_count += 1
                image_path, _ = self.samples[index]
                image = Image.open(image_path).convert('RGB')

            # 应用变换
            if self.transform:
                image = self.transform(image)

            _, label = self.samples[index]
            return image, label

        except Exception as e:
            self.logger.error(f"加载图像失败：{e}")
            raise

    def _get_all_samples(self) -> List[Tuple[str, str]]:
        """高效获取所有样本路径和标签"""
        samples: List[Tuple[str, str]] = []

        try:
            # 一次性获取所有文件，减少系统调用
            all_files = []
            for root, _, files in os.walk(self.images_dir):
                for file in files:
                    if self._is_valid_image(file):
                        full_path = os.path.join(root, file)
                        all_files.append((full_path, root))

            # 一次性排序所有文件
            sorted_files = natsorted(all_files, key=lambda x: x[0])

            # 构建样本列表
            for image_path, root_dir in sorted_files:
                label = ""
                if self.labeled:
                    label = os.path.basename(root_dir)
                samples.append((image_path, label))

        except Exception as e:
            self.logger.error(f"扫描数据集时出错: {e}")
            raise

        return samples

    def _is_valid_image(self, filename: str) -> bool:
        """检查文件是否为有效的图像文件"""
        return filename.lower().endswith(self.allowed_extensions)

    def _load_image(self, image_path: str) -> Image.Image:
        """安全加载图像文件"""
        try:
            image = Image.open(image_path)
            # 确保转换为RGB格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            self.logger.error(f"无法加载图像 {image_path}: {e}")
            raise

    def _log_dataset_statistics(self):
        """记录数据集统计信息"""
        if not self.samples:
            self.logger.warning("数据集中未找到任何有效图像")
            return

        # 统计标签分布
        if self.labeled:
            label_counts = {}
            for _, label in self.samples:
                label_counts[label] = label_counts.get(label, 0) + 1

            self.logger.info(f"数据集包含 {len(label_counts)} 个类别")
            # 显示所有类别的样本数，按样本数量降序排列
            all_classes = sorted(label_counts.items(),
                                 key=lambda x: x[1], reverse=True)
            for label, count in all_classes:
                self.logger.info(f"  {label}: {count} 个样本")

    def get_label_distribution(self) -> dict:
        """获取标签分布统计"""
        if not self.labeled:
            return {}

        distribution = {}
        for _, label in self.samples:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def filter_by_labels(self, allowed_labels: List[str]) -> 'ImgDataSet':
        """根据标签过滤数据集"""
        filtered_samples = [
            (path, label) for path, label in self.samples
            if label in allowed_labels
        ]

        # 创建新的数据集实例
        filtered_dataset = ImgDataSet(
            transform=self.transform,
            labeled_folders=self.labeled,
            images_dir=str(self.images_dir)
        )
        filtered_dataset.samples = filtered_samples
        filtered_dataset.logger.info(f"过滤后剩余 {len(filtered_samples)} 个样本")

        return filtered_dataset


class ImgFolderDataset(ImageFolder):
    """
    扩展的图像文件夹数据集类，在torchvision.ImageFolder基础上增加额外功能
    
    主要增强：
    - 提供idx到类名的反向映射
    - 改进的错误处理
    - 更好的类型注解支持
    """
    
    def __init__(self, images_dir: str, transform: Optional[Callable] = None):
        """
        初始化图像文件夹数据集
        
        Args:
            images_dir: 图像文件夹路径
            transform: 图像变换函数
        """
        super().__init__(images_dir, transform=transform)
        
        # 创建索引到类名的映射字典
        self.idx_to_class: Dict[int, str] = {
            idx: class_name 
            for class_name, idx in self.class_to_idx.items()
        }
        
        # 验证数据集完整性
        self._validate_dataset()
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.imgs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        获取指定索引的样本
        
        Args:
            index: 样本索引
            
        Returns:
            图像张量和类别索引
            
        Raises:
            IndexError: 索引超出范围
            Exception: 图像加载或其他错误
        """
        if index >= len(self):
            raise IndexError(f"索引 {index} 超出数据集范围 [0, {len(self)-1}]")
            
        try:
            path, target = self.samples[index]
            sample = self.loader(path)
            
            if self.transform is not None:
                sample = self.transform(sample)
                
            return sample, target
            
        except Exception as e:
            raise RuntimeError(f"加载样本 {index} 失败: {str(e)}") from e
    
    def _validate_dataset(self) -> None:
        """验证数据集的基本完整性"""
        if len(self.classes) == 0:
            raise ValueError("数据集中未找到任何类别文件夹")
            
        if len(self.imgs) == 0:
            raise ValueError("数据集中未找到任何图像文件")
    
    def get_class_name(self, idx: int) -> str:
        """
        根据类别索引获取类别名称
        
        Args:
            idx: 类别索引
            
        Returns:
            类别名称
            
        Raises:
            KeyError: 索引不存在
        """
        if idx not in self.idx_to_class:
            raise KeyError(f"类别索引 {idx} 不存在")
        return self.idx_to_class[idx]
    
    def get_sample_info(self, index: int) -> Dict[str, Any]:
        """
        获取样本的详细信息
        
        Args:
            index: 样本索引
            
        Returns:
            包含路径、类别索引、类别名称的字典
        """
        if index >= len(self):
            raise IndexError(f"索引 {index} 超出范围")
            
        path, target = self.samples[index]
        return {
            'path': path,
            'target': target,
            'class_name': self.idx_to_class[target],
            'index': index
        }
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Returns:
            包含类别数量、总样本数等统计信息的字典
        """
        # 统计各类别样本数量
        class_counts = {}
        for _, target in self.samples:
            class_name = self.idx_to_class[target]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'num_classes': len(self.classes),
            'total_samples': len(self.samples),
            'class_names': self.classes,
            'class_counts': class_counts,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class
        }
