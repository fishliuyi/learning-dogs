from typing import List, Tuple

import PIL.Image as Image
import numpy as np
import torch

from faiss_database import FeatureDatabase
from log import Log
from model import PetNet50

# 尝试导入 OpenVINO 推理器
try:
    from openvino_infer import create_openvino_inferencer
    USE_OPENVINO = True
except ImportError:
    USE_OPENVINO = False
    print("⚠️ OpenVINO 推理器不可用，将使用 PyTorch CPU 推理")

class InferDog:
    loger = Log(__qualname__)

    def __init__(self, model_path: str, db_path: str, index_name: str, query_size: int = 20,
                 use_openvino: bool = True):
        """
        初始化推理器
        
        Args:
            model_path: 模型路径
            db_path: 特征数据库路径
            index_name: 索引名称
            query_size: 查询数量
            use_openvino: 是否使用 OpenVINO GPU 加速（默认 True）
        """
        self.query_size = query_size
        
        # 加载特征数据库
        database = FeatureDatabase(db_path)
        index, features = database.load_features(index_name)
        self.index = index
        self.features = features
        
        # 选择推理方式
        if USE_OPENVINO and use_openvino:
            print("🚀 使用 OpenVINO GPU/NPU 加速推理...")
            try:
                self.openvino_inferencer = create_openvino_inferencer(model_path, prefer_gpu=True)
                self.use_openvino = True
                self.model = None  # 不需要 PyTorch 模型
                print(f"✅ OpenVINO 推理器初始化成功")
            except Exception as e:
                print(f"⚠️ OpenVINO 初始化失败：{e}，降级到 PyTorch CPU 推理")
                self.use_openvino = False
                self.model = PetNet50(model_path)
        else:
            print("⚠️ 使用 PyTorch CPU 推理")
            self.use_openvino = False
            self.model = PetNet50(model_path)

    @loger.performance_monitor("品种比对")
    def find(self, image: Image, sort: int = 3) -> List[Tuple[str, float]]:
        """
        根据输入图像查找相似犬种
    
        Args:
            image: 输入的狗狗图像
            sort: 返回前 N 个最相似的结果
    
        Returns:
            包含犬种名称和相似度分数的元组列表
        """
        try:
            # 图像预处理
            if self.use_openvino:
                input_tensor: torch.Tensor = self.openvino_inferencer.inferencer.model.transform(image).unsqueeze(dim=0)
            else:
                input_tensor: torch.Tensor = self.model.transform(image).unsqueeze(dim=0)
                
            # 特征提取（使用 OpenVINO 或 PyTorch）
            if self.use_openvino:
                # OpenVINO GPU/NPU加速推理
                embedding_np = self.openvino_inferencer.infer(input_tensor)
                query_embedding: np.ndarray = embedding_np
            else:
                # PyTorch CPU 推理
                input_tensor = input_tensor.to(self.model.device, non_blocking=True)
                embedding: torch.Tensor = self.model.model(input_tensor)
                query_embedding: np.ndarray = embedding.detach().cpu().numpy()
    
            # 特征匹配
            distances, indices = self.index.search(
                query_embedding, self.query_size)
            indices: List[int] = indices.ravel().tolist()
            distances: List[float] = distances.ravel().tolist()
            indices_labels: List[str] = [self.features[i] for i in indices]
    
            self.loger.info(
                f"完成查询 找到 {len(indices)} 个最相似的参考特征向量")
    
            # 计算相似犬种
            return self._query_similar_breeds(indices, distances, indices_labels, sort)
    
        except Exception as e:
            self.loger.error(f"品种识别过程中发生错误：{str(e)}")
            raise

    def _query_similar_breeds(self, indices: List[int], distances: List[float], indices_labels: List[str], sort: int) -> List[Tuple[str, float]]:
        """
        根据匹配结果计算相似犬种的综合得分

        Args:
            indices: FAISS搜索返回的索引列表
            distances: 对应的距离值列表
            indices_labels: 索引对应的犬种标签
            sort: 返回前N个结果

        Returns:
            按相似度排序的犬种及其得分列表
        """
        # 统计每个犬种的投票数和距离
        breed_votes = {}
        breed_distances = {}

        for i, (idx, distance) in enumerate(zip(indices, distances)):
            breed = indices_labels[i]  # 使用循环索引而不是FAISS索引
            if breed not in breed_votes:
                breed_votes[breed] = 0
                breed_distances[breed] = []
            breed_votes[breed] += 1
            breed_distances[breed].append(distance)

        self.loger.debug(f"统计到 {len(breed_votes)} 个不同的犬种")

        # 计算每个犬种的综合得分
        breed_scores = []
        k = len(indices)  # 使用实际的搜索结果数量

        for breed in breed_votes:
            # 投票得分：该犬种被选中的频率
            vote_score = breed_votes[breed] / k

            # 距离得分：平均距离的倒数映射
            avg_distance = np.mean(breed_distances[breed])
            distance_score = 1.0 / (1.0 + avg_distance)

            # 综合得分：投票权重0.7 + 距离权重0.3
            final_score = vote_score * 0.7 + distance_score * 0.3
            breed_scores.append((breed, final_score))

            self.loger.debug(f"犬种 {breed}: 投票得分={vote_score:.3f}, "
                             f"距离得分={distance_score:.3f}, 综合得分={final_score:.3f}")

        # 按综合得分降序排序
        breed_scores.sort(key=lambda x: x[1], reverse=True)

        self.loger.info(f"预测排名 返回前 {min(sort, len(breed_scores))} 个结果")
        return breed_scores[:sort]


def infer_dog(image: Image, model_path: str, db_path: str, index_name: str, query_size: int = 20, sort: int = 3) -> List[Tuple[str, float]]:
    infer_dog = InferDog(model_path, db_path, index_name, query_size)
    return infer_dog.find(image, sort)
