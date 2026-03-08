import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

# from torch_optimizer import RAdam  # 注释掉，使用PyTorch内置优化器
from pathlib import Path
from torchvision import models, transforms
from pprint import pformat
from typing import Dict, Any, Optional, Tuple

from log import Log


def get_optimal_device() -> torch.device:
    """获取最优计算设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_config(conf_path: Path) -> Dict[str, Any]:
    """获取配置参数"""
    if not conf_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{conf_path}")
    if not conf_path.is_file():
        raise ValueError(f"配置路径不是文件：{conf_path}")
    with open(conf_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


class Resnet50(nn.Module):
    """改进的ResNet50特征提取器"""

    def __init__(self, embedding_size: int = 128):
        super().__init__()
        # 使用最新的预训练权重API
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(model.children())[:-1])
        # 自定义嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(in_features=2048, out_features=embedding_size),
            nn.ReLU(inplace=True),  # 添加激活函数
            nn.Dropout(p=0.2)       # 添加dropout防止过拟合
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # 特征提取
        features = self.features(image)
        features = features.flatten(start_dim=1)

        # 嵌入映射
        embedding = self.embedding(features)
        # L2归一化
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding


class TrainModel:
    logger = Log(__qualname__).logger

    def __init__(self, model_path: str = 'data/TsinghuaDogs/model/proxynca-resnet50.pth', conf_path: str = "src2/config/soft_triple_loss.yaml"):
        self.model_path: Path = Path(model_path).resolve()
        self.conf_path: Path = Path(conf_path).resolve()

        self.device: torch.device = get_optimal_device()
        self.logger.info(f"使用设备: {self.device}")

        self.config: Optional[Dict[str, Any]] = get_config(self.conf_path)
        self.logger.info(f"加载配置: {pformat(self.config)}")

        self.model: Optional[nn.Module] = self._init_model()
        # 使用PyTorch内置的Adam优化器替代RAdam
        self.optimizer: Optional[torch.optim.Adam] = torch.optim.Adam(
            self.model.parameters(), lr=self.config["lr"])
        self.logger.info(f"优化器初始化完成: {self.optimizer}")

        self.transform: Optional[transforms.Compose] = self._init_transform()
        self.test_transform: Optional[transforms.Compose] = self._init_test_transform(
        )

    def _init_model(self):
        """初始化模型组件"""
        model = nn.DataParallel(Resnet50(
            embedding_size=self.config["embedding_size"]
        ))
        model = model.to(self.device)
        self.logger.info(f"模型初始化完成：{model}")
        return model

    def _init_transform(self) -> transforms.Compose:
        """初始化图像变换"""
        image_size = self.config.get("image_size", 224)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=5, scale=(
                0.8, 1.2), translate=(0.2, 0.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _init_test_transform(self) -> transforms.Compose:
        image_size = self.config.get("image_size", 224)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class LossModel(nn.Module):

    def __init__(self, n_classes: int, conf_path: str = "src2/config/soft_triple_loss.yaml"):
        super().__init__()
        self.conf_path: Path = Path(conf_path).resolve()
        self.config: Optional[Dict[str, Any]] = get_config(self.conf_path)

        self.device: torch.device = get_optimal_device()
        self.n_classes: int = n_classes
        self.embedding_size: int = self.config.get("embedding_size", 128)
        self.n_centers_per_class: int = self.config.get(
            "n_centers_per_class", 32)
        self.lambda_: float = self.config.get("lambda", 20.0)
        self.gamma: float = self.config.get("gamma", 10.0)
        self.tau: float = self.config.get("tau", 0.1)
        self.margin: float = self.config.get("margin", 0.3)

        # 验证参数合理性
        self._validate_parameters()

        # 每个类别有 n 个中心
        self.centers: torch.Tensor = nn.Parameter(
            torch.randn(self.embedding_size,
                        self.n_centers_per_class * self.n_classes)
        )
        # 移动到设备并初始化
        self.centers.data = self.centers.data.to(self.device)
        nn.init.kaiming_uniform_(self.centers, a=5**0.5)

        # 预计算一些不变的值以提高性能
        self._precompute_constants()

        # 正则化项的权重 - 类内中心距离的掩码
        self.weight: torch.Tensor = torch.zeros(
            n_classes * self.n_centers_per_class,
            self.n_classes * self.n_centers_per_class,
            dtype=torch.float32,
            device=self.device
        )

        # 向量化构建权重矩阵以提高性能
        self._build_weight_matrix()

    def _validate_parameters(self) -> None:
        """验证模型参数的合理性"""
        if self.n_classes <= 0:
            raise ValueError(f"类别数必须大于0，当前值: {self.n_classes}")
        if self.embedding_size <= 0:
            raise ValueError(f"嵌入维度必须大于0，当前值: {self.embedding_size}")
        if self.n_centers_per_class <= 0:
            raise ValueError(f"每类中心数必须大于0，当前值: {self.n_centers_per_class}")

    def _precompute_constants(self) -> None:
        """预计算常量以提高性能"""
        # 预计算中心总数
        self.total_centers: int = self.n_classes * self.n_centers_per_class

    def _build_weight_matrix(self) -> None:
        """高效构建权重矩阵"""
        # 使用向量化操作构建权重矩阵
        for i in range(self.n_classes):
            start_idx = i * self.n_centers_per_class
            end_idx = (i + 1) * self.n_centers_per_class
            # 为同一类内的中心设置权重
            self.weight[start_idx:end_idx, start_idx:end_idx] = (
                torch.ones(self.n_centers_per_class, self.n_centers_per_class) -
                torch.eye(self.n_centers_per_class)
            )

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        计算 SoftTriple 损失。

        参数:
            embeddings: 输入嵌入，形状为 [batch_size, embedding_size]
            labels: 真实标签，形状为 [batch_size]

        返回:
            损失值元组 (loss_value, dummy_value_for_compatibility)
        """
        # 如需要，将标签移到指定设备
        if labels.device != self.device:
            labels = labels.to(self.device)

        # 归一化中心向量
        centers = F.normalize(self.centers, p=2, dim=0)

        # 计算嵌入与所有中心之间的余弦相似度
        # 形状: [batch_size, n_classes * n_centers_per_class]
        similarity = embeddings.matmul(centers)
        # 重塑为: [batch_size, n_classes, n_centers_per_class]
        similarity = similarity.view(-1, self.n_classes,
                                     self.n_centers_per_class)

        # 使用 softmax 计算注意力权重
        attention_weights = F.softmax(similarity * self.gamma, dim=2)

        # 计算每个类别的加权平均相似度
        # 这为每个类别创建了一个"软"代表
        soft_similarity = torch.sum(attention_weights * similarity, dim=2)

        # 仅对正类应用边界
        margin_mask = torch.zeros_like(soft_similarity)
        margin_mask.scatter_(1, labels.unsqueeze(1), self.margin)

        # 计算分类损失
        logits = self.lambda_ * (soft_similarity - margin_mask)
        soft_triple_loss = F.cross_entropy(logits, labels)

        # 如果启用则添加正则化项
        if self.tau > 0 and self.n_centers_per_class > 1:
            # 计算同类中心之间的成对距离
            # 为了数值稳定性，限制值的范围
            center_similarities = centers.t().matmul(centers)
            # 确保值在 arccos 的有效范围 [-1, 1] 内
            center_similarities = torch.clamp(
                center_similarities, -1.0 + 1e-6, 1.0 - 1e-6)

            # 使用权重掩码提取类内中心相似度
            intra_class_sims = center_similarities[self.weight.bool()]

            # 将余弦相似度转换为角度距离
            angular_distances = torch.acos(intra_class_sims)
            regularization = torch.mean(angular_distances)

            return soft_triple_loss + self.tau * regularization, 0
        else:
            return soft_triple_loss, 0


class PetNet50:
    """优化的宠物识别模型包装器"""

    logger = Log(__qualname__).logger

    def __init__(self, model_path: str = 'data/TsinghuaDogs/model/proxynca-resnet50.pth'):
        """
        初始化PetNet50模型

        Args:
            model_path: 模型文件路径
        """
        self.model_path: Path = Path(model_path).resolve()
        self.device: torch.device = get_optimal_device()
        self.logger.info(f"使用设备: {self.device}")

        self.model: Optional[nn.Module] = None
        self.config: Optional[Dict[str, Any]] = None
        self.transform: Optional[transforms.Compose] = None

        # 模型加载和初始化
        self._initialize_model()

    def _validate_model_path(self) -> bool:
        """验证模型路径有效性"""
        if not self.model_path.exists():
            self.logger.error(f"模型文件不存在: {self.model_path}")
            return False
        if not self.model_path.is_file():
            self.logger.error(f"模型路径不是文件: {self.model_path}")
            return False
        return True

    def _initialize_model(self):
        """初始化模型组件"""
        # 验证模型文件
        if not self._validate_model_path():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        try:
            # 加载检查点
            checkpoint = self._load_checkpoint()
            self.config = checkpoint["config"]

            # 创建和加载模型
            self.model = self._create_model()
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # 移动到设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()

            # 初始化变换
            self.transform = self._init_transform()

            self.logger.info(f"模型加载成功: {self.model_path}")
            self.logger.info(f"模型配置: embedding_size={self.config.get('embedding_size', 128)}, "
                             f"image_size={self.config.get('image_size', 224)}")

        except Exception as e:
            self.logger.error(f"模型初始化失败: {e}")
            raise

    def _load_checkpoint(self) -> Dict[str, Any]:
        """安全加载模型检查点"""
        try:
            # CPU上加载避免GPU内存占用
            checkpoint = torch.load(self.model_path, map_location="cpu")

            # 验证检查点完整性
            required_keys = ["config", "model_state_dict"]
            for key in required_keys:
                if key not in checkpoint:
                    raise ValueError(f"检查点缺少必要键: {key}")

            return checkpoint
        except Exception as e:
            raise RuntimeError(f"加载模型检查点失败: {e}")

    def _create_model(self) -> nn.Module:
        """创建模型实例"""
        embedding_size = self.config.get("embedding_size", 128)
        model = Resnet50(embedding_size=embedding_size)
        return model

    def _init_transform(self) -> transforms.Compose:
        """初始化图像变换"""
        image_size = self.config.get("image_size", 224)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def embedding(self, image: torch.Tensor) -> np.ndarray:
        """
        提取图像特征向量

        Args:
            image: 输入图像张量 [B, C, H, W]

        Returns:
            特征向量 numpy数组
        """
        self._validate_input(image)

        try:
            with torch.no_grad():
                # 设备迁移优化 - 只在必要时迁移
                if image.device != self.device:
                    image = image.to(self.device, non_blocking=True)

                # 根据设备选择合适的精度
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        features = self.model(image)
                else:
                    features = self.model(image)

            # 转换为numpy并确保在CPU上
            return features.cpu().numpy()

        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
            raise

    def _validate_input(self, image: torch.Tensor):
        """验证输入张量"""
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"输入必须是torch.Tensor类型，当前类型: {type(image)}")

        if image.dim() != 4:
            raise ValueError(
                f"输入图像必须是4D tensor [B, C, H, W]，当前维度: {image.dim()}")

        if image.size(1) != 3:
            raise ValueError(f"输入图像必须是RGB格式(3通道)，当前通道数: {image.size(1)}")

        # 检查数值范围
        if image.min() < -3 or image.max() > 3:  # 考虑标准化后的范围
            self.logger.warning("输入张量数值范围可能不正确")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        if self.model is None or self.config is None:
            return {}

        return {
            "model_path": str(self.model_path),
            "device": str(self.device),
            "embedding_size": self.config.get("embedding_size", 128),
            "image_size": self.config.get("image_size", 224),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

    def __repr__(self) -> str:
        """字符串表示"""
        info = self.get_model_info()
        if not info:
            return "PetNet50(未初始化)"
        return f"PetNet50(embedding_size={info['embedding_size']}, device={info['device']})"
