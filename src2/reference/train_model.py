
import os
import datetime
import numpy as np
import torch
import random
import json

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

from typing import List, Dict, Tuple, Any, Union

from log import Log
from model import TrainModel, LossModel
from image_data import ImgFolderDataset


class CreateTrainMOdel:

    loger = Log(__qualname__)

    def __init__(self,
                 config_path: str,
                 md_path: str,
                 data_path: str,
                 test_path: str,
                 chk_path: str = "checkpoints",
                 train_epochs: int = 100,
                 validate_frequency: int = 1000,
                 random_seed: int = 12345):
        """
        初始化训练模型
        
        Args:
            config_path: 配置文件路径
            md_path: 模型文件路径
            data_path: 训练数据路径
            test_path: 测试数据路径
            chk_path: 检查点保存目录，默认 "checkpoints"
            train_epochs: 训练轮次，默认 100
            validate_frequency: 验证频率，默认 1000
            random_seed: 随机种子，默认 12345
        """

        self.logger = self.loger.logger

        self.random_seed = random_seed
        self._set_random_seed()

        self.model = TrainModel(md_path, config_path)
        self.logger.info(f"初始化模型: {self.model}")

        self.dataset: ImgFolderDataset = ImgFolderDataset(
            data_path, self.model.transform)
        batch_size: int = self.model.config.get("batch_size", 32)
        # 根据 CPU 核心数动态设置 worker 数量
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # 使用 CPU 核心数的 50-75% 作为 worker 数
        default_workers = max(1, min(cpu_count // 2, 8))
        num_workers: int = self.model.config.get("n_workers", default_workers)
        
        self.logger.info(f"检测到 CPU 核心数：{cpu_count}")
        self.logger.info(f"使用 DataLoader worker 数量：{num_workers}")
                
        # 根据是否有 GPU 来决定是否使用 pin_memory
        # CPU 训练时也建议使用 pin_memory 以加速数据传输
        # 注意：Windows 上可能会有警告，但实际能提升数据加载速度
        use_pin_memory = True
        
        # 输出警告过滤（可选）
        import warnings
        warnings.filterwarnings('ignore', message='.*pin_memory.*accelerator.*')
                
        self.train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=use_pin_memory)
        self.logger.info(f"训练数据加载器：{self.train_loader.dataset}")

        n_classes = len(self.dataset.classes)
        self.loss_model = LossModel(n_classes)
        self.logger.info(f"初始化损失模型: {self.loss_model}")

        self.test_dataset: ImgFolderDataset = ImgFolderDataset(
            test_path, self.model.test_transform)
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.logger.info(f"测试数据加载器: {self.test_loader.dataset}")

        self.reference_set: ImgFolderDataset = ImgFolderDataset(
            data_path, self.model.test_transform)
        self.reference_loader = DataLoader(
            dataset=self.reference_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.logger.info(f"参考数据加载器: {self.reference_loader.dataset}")

        time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.checkpoint_dir: str = os.path.join(chk_path, time)
                        
        # 确保检查点目录存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)
                        
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir)
        self.logger.info(f"检查点保存目录：{self.checkpoint_dir}")

        self.train_epochs = train_epochs
        self.validate_frequency = validate_frequency
        self.output_dict: Dict[str, Any] = {
            "total_epoch": train_epochs,
            "current_epoch": 0,
            "current_iter": 0,
            "metrics": {
                "mean_average_precision": 0.0,
                "average_precision_at_1": 0.0,
                "average_precision_at_5": 0.0,
                "average_precision_at_10": 0.0,
                "top_1_accuracy": 0.0,
                "top_5_accuracy": 0.0,
                "normalized_mutual_information": 0.0,
            }
        }

    def _set_random_seed(self) -> None:
        """
        设置随机种子以确保结果可复现
        影响范围：random、numpy、pytorch
        """
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @loger.performance_monitor("训练")
    def train(self):
        """
        主训练循环
            
        Returns:
            Dict[str, Any]: 最终训练结果和指标
        """
        try:
            self.logger.info(f"开始训练，共 {self.train_epochs} 个 epoch")
                
            for epoch in range(1, self.train_epochs + 1):
                self.logger.info(f"Epoch {epoch}/{self.train_epochs}")
                self._train_one_epoch()
    
            self.logger.info(f"完成所有训练轮次：{self.train_epochs} epochs")
                
            # 可视化嵌入向量
            self._visualize_embeddings()
                
            # 保存最终配置和指标
            self._save_final_results()
                
            # 保存最终模型到 md_path 附近
            self._save_final_model_to_md_path()
                
            return self.output_dict
                
        except KeyboardInterrupt:
            self.logger.warning("训练被用户中断")
            raise
        except Exception as e:
            self.logger.error(f"训练过程中发生严重错误：{e}", exc_info=True)
            raise

    def _train_one_epoch(self):
        """
        执行一个完整的训练周期
        
        Returns:
            Dict[str, float]: 平均损失和其他统计信息
        """
        self.output_dict["current_epoch"] += 1
        current_epoch = self.output_dict["current_epoch"]
        self.logger.info(f"开始训练周期 {current_epoch}")

        running_loss: float = 0.0
        running_fraction_hard_triplets: float = 0.0
        batch_count = 0
        
        # 获取梯度累积步数
        accumulation_steps: int = self.model.config.get("accumulation_steps", 1)

        try:
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                self.output_dict["current_iter"] += 1
                current_iter: int = self.output_dict["current_iter"]

                # 定期验证
                if (current_iter == 1 or current_iter % self.validate_frequency == 0):
                    metrics: Dict[str, Any] = self._calculate_all_metrics()
                    self._log_info_metrics(metrics)

                    # 记录到 TensorBoard
                    for metric_name, value in metrics.items():
                        self.writer.add_scalar(
                            f"test/{metric_name}", value, current_iter)

                    # 保存最佳模型
                    if metrics['mean_average_precision'] > self.output_dict["metrics"]["mean_average_precision"]:
                        self.output_dict["metrics"] = metrics
                        checkpoint_path = self._save_checkpoint()
                        self.logger.info(f"保存最佳模型：{checkpoint_path}")

                # 训练单个批次（支持梯度累积）
                is_accumulation_step = (batch_idx + 1) % accumulation_steps != 0
                output_batch: Dict[str, Any] = self._train_one_batch_with_accumulation(
                    images, labels, is_accumulation_step)
                running_loss += output_batch["loss"]
                running_fraction_hard_triplets += output_batch["fraction_hard_triplets"]
                batch_count += 1

                # 记录到 TensorBoard
                for metric_name, value in output_batch.items():
                    self.writer.add_scalar(
                        f"train/{metric_name}", value, current_iter)

                # 打印训练日志
                log_frequency = self.model.config.get("log_frequency", 100)
                if current_iter % log_frequency == 0 and batch_count > 0:
                    average_loss: float = running_loss / batch_count
                    average_hard_triplets: float = running_fraction_hard_triplets / batch_count * 100
                    self.logger.info(
                        f"TRAINING\t[{current_epoch}|{current_iter}]\t"
                        f"train_loss: {average_loss:.6f}\t"
                        f"hard triplets: {average_hard_triplets:.2f}%"
                    )
                    running_loss = 0.0
                    running_fraction_hard_triplets = 0.0
                    batch_count = 0

            # 在最后一个 epoch 运行验证
            if current_epoch == self.output_dict["total_epoch"]:
                metrics: Dict[str, Any] = self._calculate_all_metrics()
                self._log_info_metrics(metrics)

                for metric_name, value in metrics.items():
                    self.writer.add_scalar(
                        f"test/{metric_name}", value, current_iter)

                if metrics['mean_average_precision'] > self.output_dict["metrics"]["mean_average_precision"]:
                    self.output_dict["metrics"] = metrics
                    self._save_checkpoint()
                    
        except Exception as e:
            self.logger.error(f"训练周期 {current_epoch} 失败：{e}", exc_info=True)
            raise

    def _log_info_metrics(self, metrics: Dict[str, Any]):
        """
        打印评估指标信息到日志
        
        Args:
            metrics: 包含各项评估指标的字典
        """
        self.logger.info("#" * 130)
        current_epoch = self.output_dict['current_epoch']
        current_iter = self.output_dict['current_iter']
        self.logger.info(
            f"VALIDATING\t[{current_epoch}|{current_iter}]\t"
            f"MAP: {metrics['mean_average_precision']:.2f}%\t"
            f"AP@1: {metrics['average_precision_at_1']:.2f}%\t"
            f"AP@5: {metrics['average_precision_at_5']:.2f}%\t"
            f"Top-1: {metrics['top_1_accuracy']:.2f}%\t"
            f"Top-5: {metrics['top_5_accuracy']:.2f}%\t"
            f"NMI: {metrics['normalized_mutual_information']:.2f}\t"
        )
        self.logger.info("#" * 130)

    def _visualize_embeddings(self):
        """
        可视化嵌入向量到 TensorBoard
        包括训练集、参考集和测试集的嵌入向量和模型计算图
        """
        try:
            self.logger.info("Calculating train embeddings for visualization...")
            self._log_embeddings_to_tensorboard(self.train_loader, tag="train")
            
            self.logger.info("Calculating reference embeddings for visualization...")
            self._log_embeddings_to_tensorboard(
                self.reference_loader, tag="reference")
                
            self.logger.info("Calculating test embeddings for visualization...")
            self._log_embeddings_to_tensorboard(self.test_loader, tag="test")
            
            # 可视化模型计算图
            self.logger.info("Visualizing model's graph...")
            with torch.no_grad():
                dummy_input = torch.zeros(
                    1, 3, self.model.config["image_size"], self.model.config["image_size"]
                ).to(self.model.device)
                self.writer.add_graph(
                    self.model.model.module.features, dummy_input)
                    
        except Exception as e:
            self.logger.error(f"可视化嵌入向量失败：{e}", exc_info=True)
            raise

    def _save_final_results(self):
        """
        保存最终训练结果和超参数到 TensorBoard 和 JSON 文件
        """
        try:
            self.logger.info("Saving all hyper-parameters and final metrics...")
            
            # 保存超参数和指标
            self.writer.add_hparams(
                self.model.config,
                metric_dict={f"hyperparams/{key}": value for key,
                             value in self.output_dict["metrics"].items()})

            # 保存完整的输出字典
            with open(os.path.join(self.checkpoint_dir, "output_dict.json"), "w", encoding="utf-8") as f:
                json.dump(self.output_dict, f, indent=4, ensure_ascii=False)

            self.logger.info(f"Dumped output_dict.json at {self.checkpoint_dir}")
            
        except Exception as e:
            self.logger.error(f"保存最终结果失败：{e}", exc_info=True)
            raise

    def _save_final_model_to_md_path(self):
        """
        将训练完成的最佳模型保存到 md_path 附近
        方便后续推理使用
        
        注意：应该在所有训练和验证完成后调用，确保复制的是最佳模型
        """
        try:
            # 从 output_dict 中获取最佳模型的检查点信息
            # 由于每次 MAP 提升都会保存，最后一次保存的就是最佳模型
            # 我们需要找到最近保存的检查点文件
            import glob
            
            # 查找 checkpoint_dir 下所有的 .pth 文件
            checkpoint_pattern = os.path.join(self.checkpoint_dir, "*.pth")
            checkpoint_files = glob.glob(checkpoint_pattern)
            
            if not checkpoint_files:
                self.logger.warning("未找到任何检查点文件")
                return
            
            # 按修改时间排序，最新的就是最佳模型
            best_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            
            # 从 md_path 提取目录和文件名
            from pathlib import Path
            md_path_obj = Path(self.model.model_path)
            model_dir = md_path_obj.parent
            model_name = md_path_obj.stem  # 不含扩展名的文件名
            
            # 构建最终模型保存路径
            final_model_name = f"{model_name}_trained.pth"
            final_model_path = os.path.join(model_dir, final_model_name)
            
            # 复制文件到最终位置
            import shutil
            shutil.copy2(best_checkpoint, final_model_path)
            
            self.logger.info(f"✓ 最终模型已保存到：{final_model_path}")
            self.logger.info(f"  (从检查点 {best_checkpoint} 复制)")
            
        except Exception as e:
            self.logger.error(f"保存最终模型失败：{e}", exc_info=True)
            # 不抛出异常，避免影响主流程

    def _log_embeddings_to_tensorboard(self, loader: DataLoader, tag: str):
        """
        将数据加载器的嵌入向量记录到 TensorBoard
        
        Args:
            loader: 数据加载器
            tag: TensorBoard 中的标签名称
        """
        if tag == "train":
            if hasattr(loader.sampler, "sequential_sampling"):
                loader.sampler.sequential_sampling = True

        # Calculating embedding of training set for visualization
        embeddings, labels = self._get_embeddings_from_dataloader(loader)
        self.writer.add_embedding(
            embeddings, metadata=labels.tolist(), tag=tag)

    def _calculate_all_metrics(self, k: Tuple[int, int, int] = (1, 5, 10)) -> Dict[str, Any]:
        """
        计算所有评估指标
        
        Args:
            k: 用于计算 precision@k 的 k 值元组，默认 (1, 5, 10)
            
        Returns:
            Dict[str, Any]: 包含所有指标的字典
        """
        embeddings_test, labels_test = self._get_embeddings_from_dataloader(
            self.test_loader)
        embeddings_ref, labels_ref = self._get_embeddings_from_dataloader(
            self.reference_loader)

        # Expand dimension for batch calculating
        embeddings_test = embeddings_test.unsqueeze(
            dim=0)  # [M x K] -> [1 x M x embedding_size]
        embeddings_ref = embeddings_ref.unsqueeze(
            dim=0)  # [N x K] -> [1 x N x embedding_size]
        labels_test = labels_test.unsqueeze(dim=1)  # [M] -> [M x 1]

        # Pairwise distance of all embeddings between test set and reference set
        distances: torch.Tensor = torch.cdist(
            embeddings_test, embeddings_ref, p=2).squeeze()  # [M x N]

        # Calculate precision_at_k on test set with k=1, k=5 and k=10
        metrics: Dict[str, float] = {}
        for i in k:
            metrics[f"average_precision_at_{i}"] = self._calculate_precision_at_k(
                distances, labels_test, labels_ref, k=i)

        # Calculate mean average precision (MAP)
        mean_average_precision: float = sum(
            precision_at_k for precision_at_k in metrics.values()) / len(metrics)
        metrics["mean_average_precision"] = mean_average_precision

        # Calculate top-1 and top-5 and top-10 accuracy
        for i in k:
            metrics[f"top_{i}_accuracy"] = self._calculate_topk_accuracy(
                distances, labels_test, labels_ref, top_k=i)

        # Calculate NMI score
        n_classes: int = len(self.test_loader.dataset.classes)
        metrics["normalized_mutual_information"] = self._calculate_normalized_mutual_information(
            embeddings_test.squeeze(), labels_test.squeeze(), n_classes
        )
        
        # 释放不再使用的张量以节省内存
        del embeddings_test, embeddings_ref, labels_test, labels_ref, distances
        
        # CPU/GPU 垃圾回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    def _get_embeddings_from_dataloader(self,
                                        loader: DataLoader,
                                        return_numpy_array=False,
                                        return_image_paths=False,) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]:
        """
        从数据加载器提取嵌入向量
        
        Args:
            loader: 数据加载器
            return_numpy_array: 是否返回 numpy 数组，默认 False
            return_image_paths: 是否返回图像路径，默认 False
            
        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], Tuple[np.ndarray, np.ndarray]]: 
                (嵌入向量，标签) 或 (嵌入向量，标签，图像路径)
        """
        self.model.model.eval()
        embeddings_ls: List[torch.Tensor] = []
        labels_ls: List[torch.Tensor] = []
        
        with torch.no_grad():  # 使用 no_grad 上下文管理器，不计算梯度
            for images_, labels_ in loader:
                images: torch.Tensor = images_.to(
                    self.model.device, non_blocking=True)
                labels: torch.Tensor = labels_.to(
                    self.model.device, non_blocking=True)
                embeddings: torch.Tensor = self.model.model(images)
                embeddings_ls.append(embeddings)
                labels_ls.append(labels)
                
                # 及时释放每个 batch 的张量
                del images, labels, embeddings
        
        # 合并所有嵌入向量
        embeddings: torch.Tensor = torch.cat(
            embeddings_ls, dim=0)  # shape: [N x embedding_size]
        labels: torch.Tensor = torch.cat(labels_ls, dim=0)  # shape: [N]
        
        # 清空列表以释放内存
        del embeddings_ls, labels_ls

        if return_numpy_array:
            embeddings = embeddings.cpu().numpy()
            labels = labels.cpu().numpy()

        if return_image_paths:
            images_paths: List[str] = []
            for path, _ in loader.dataset.samples:
                images_paths.append(path)
            return (embeddings, labels, images_paths)

        return (embeddings, labels)

    def _calculate_precision_at_k(self,
                                  distances: torch.Tensor,
                                  labels_test: torch.Tensor,
                                  labels_ref: torch.Tensor,
                                  k: int
                                  ) -> float:
        """
        计算 Precision@k 指标
        
        Args:
            distances: 测试集和参考集之间的距离矩阵 [M x N]
            labels_test: 测试集标签 [M]
            labels_ref: 参考集标签 [N]
            k: 前 k 个最近邻
            
        Returns:
            float: Precision@k 百分比值
        """
        _, indices = distances.topk(
            k=k, dim=1, largest=False)  # indices shape: [M x k]

        y_pred = []
        for i in range(k):
            indices_at_k: torch.Tensor = indices[:, i]  # [M]
            y_pred_at_k: torch.Tensor = labels_ref[indices_at_k].unsqueeze(
                dim=1)  # [M x 1]
            y_pred.append(y_pred_at_k)

        y_pred: torch.Tensor = torch.hstack(y_pred)  # [M x k]
        labels_test = torch.hstack((labels_test,) * k)  # [M x k]

        precision_at_k: float = (
            (y_pred == labels_test).sum(dim=1) / k).mean().item() * 100
        return precision_at_k

    def _calculate_normalized_mutual_information(self,
                                                 embeddings: torch.Tensor,
                                                 labels_test: torch.Tensor,
                                                 n_classes: int
                                                 ) -> float:
        """
        计算归一化互信息（NMI）分数
        
        Args:
            embeddings: 嵌入向量
            labels_test: 真实标签
            n_classes: 类别数量
            
        Returns:
            float: NMI 分数
        """
        embeddings = embeddings.cpu().numpy()
        y_test: np.ndarray = labels_test.cpu().numpy().astype(np.int)

        y_pred: np.ndarray = KMeans(
            n_clusters=n_classes).fit(embeddings).labels_
        NMI_score: float = normalized_mutual_info_score(y_test, y_pred)

        return NMI_score

    def _calculate_topk_accuracy(self,
                                 distances: torch.Tensor,
                                 labels_test: torch.Tensor,
                                 labels_ref: torch.Tensor,
                                 top_k: int
                                 ) -> float:
        """
        计算 Top-k 准确率
        
        Args:
            distances: 测试集和参考集之间的距离矩阵 [M x N]
            labels_test: 测试集标签 [M]
            labels_ref: 参考集标签 [N]
            top_k: 前 k 个预测
            
        Returns:
            float: Top-k 准确率百分比值
        """
        _, indices = distances.topk(
            k=top_k, dim=1, largest=False)  # indices shape: [M x k]

        y_pred = []
        for i in range(top_k):
            indices_at_k: torch.Tensor = indices[:, i]  # [M]
            y_pred_at_k: torch.Tensor = labels_ref[indices_at_k].unsqueeze(
                dim=1)  # [M x 1]
            y_pred.append(y_pred_at_k)

        y_pred: torch.Tensor = torch.hstack(y_pred)  # [M x k]
        labels_test = torch.hstack((labels_test,) * top_k)  # [M x k]

        n_predictions: int = y_pred.shape[0]
        n_true_predictions: int = (
            (y_pred == labels_test).sum(dim=1) > 0).sum().item()
        topk_accuracy: float = n_true_predictions / n_predictions * 100
        return topk_accuracy

    def _save_checkpoint(self) -> str:
        """
        保存模型检查点
        
        Returns:
            str: 检查点文件路径
        """
        current_epoch = self.output_dict["current_epoch"]
        current_iter = self.output_dict["current_iter"]
        checkpoint_name: str = f"epoch{current_epoch}-iter{current_iter}"
        mean_average_precision = self.output_dict["metrics"]['mean_average_precision']

        if mean_average_precision is not None:
            checkpoint_name += f"-map{mean_average_precision:.2f}"
        checkpoint_name += ".pth"

        checkpoint_path: str = os.path.join(
            self.checkpoint_dir, checkpoint_name)
        torch.save(
            {
                "config": self.model.config,
                "model_state_dict": self.model.model.module.state_dict(),
            },
            checkpoint_path
        )
        return checkpoint_path

    def _train_one_batch(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        """
        训练单个批次
        
        Args:
            images: 图像张量 [batch_size, C, H, W]
            labels: 标签张量 [batch_size]
            
        Returns:
            Tuple[float, float]: (损失值，困难三元组比例)
        """
        self.model.model.train()
        self.model.optimizer.zero_grad()

        images: torch.Tensor = images.to(self.model.device, non_blocking=True)
        labels: torch.Tensor = labels.to(self.model.device, non_blocking=True)

        # 前向传播
        embeddings: torch.Tensor = self.model.model(images)
        loss, fraction_hard_triplets = self.loss_model(embeddings, labels)

        # 反向传播
        loss.backward()
        self.model.optimizer.step()
        
        # 释放不再使用的张量以节省内存
        del images, labels, embeddings, loss
        
        # CPU/GPU 垃圾回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "loss": loss.item(),
            "fraction_hard_triplets": float(fraction_hard_triplets)
        }
    
    def _train_one_batch_with_accumulation(self, images: torch.Tensor, labels: torch.Tensor, 
                                          is_accumulation_step: bool) -> Dict[str, float]:
        """
        训练单个批次（支持梯度累积）
        
        Args:
            images: 图像张量 [batch_size, C, H, W]
            labels: 标签张量 [batch_size]
            is_accumulation_step: 是否是累积步骤（不更新权重）
            
        Returns:
            Dict[str, float]: 包含损失和困难三元组比例的字典
        """
        self.model.model.train()
        
        # 如果不是累积步骤，才清零梯度
        if not is_accumulation_step:
            self.model.optimizer.zero_grad()

        images: torch.Tensor = images.to(self.model.device, non_blocking=True)
        labels: torch.Tensor = labels.to(self.model.device, non_blocking=True)

        # 前向传播
        embeddings: torch.Tensor = self.model.model(images)
        loss, fraction_hard_triplets = self.loss_model(embeddings, labels)

        # 反向传播（梯度累积）
        # 将损失除以累积步数，使得累积后的梯度等价于大批次的梯度
        scaled_loss = loss / self.model.config.get("accumulation_steps", 1)
        scaled_loss.backward()
        
        # 只在非累积步骤更新权重
        if not is_accumulation_step:
            self.model.optimizer.step()
        
        # 释放不再使用的张量以节省内存
        del images, labels, embeddings
        
        # CPU/GPU 垃圾回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "loss": loss.item(),  # 返回原始损失值用于日志记录
            "fraction_hard_triplets": float(fraction_hard_triplets)
        }


