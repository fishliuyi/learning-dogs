# src2 快速使用手册

## 📦 安装依赖

```bash
# 创建环境
conda create --name pet python=3.11.14 -y
conda activate pet

# 安装 PyTorch（根据你的 CUDA 版本）
pip install torch torchvision

# 安装其他依赖
pip install faiss-gpu psutil requests pillow pyyaml scikit-learn matplotlib tensorboard
```

---

## 🗂️ 下载训练数据

https://cg.cs.tsinghua.edu.cn/ThuDogs/
h resolution images	high-resolution.001	38.8GB
high-resolution.002
high-resolution.003
high-resolution.004
high-resolution.005
high-resolution.006
high-resolution.007
high-resolution.008
high-resolution.009


## 🗂️ 目录结构准备

```
data/
└── TsinghuaDogs/
    ├── train/              # 训练数据（按类别分文件夹）
    ├── val/                # 验证数据（按类别分文件夹）
    ├── high-resolution/    # 高分辨率参考图像
    └── model/              # 模型文件存放处
        └── proxynca-resnet50.pth  # 预训练模型
```

**数据准备脚本：**
```bash
python src/scripts/prepare_TsinghuaDogs.py --output_dir data/
```

---

## 🎯 核心功能使用

### 1️⃣ 训练模型

**步骤 1: 编辑配置文件**

编辑 `src2/trains_dog.py`，修改配置区域：

```python
# ==================== 配置区域 ====================
CONFIG_PATH = "src2\\config\\soft_triple_loss.yaml"
MODEL_PATH = "data\\TsinghuaDogs\\model\\train-resnet50.pth"
DATA_PATH = "data\\TsinghuaDogs\\train\\"
TEST_PATH = "data\\TsinghuaDogs\\val\\"
CHECKPOINT_PATH = "data\\TsinghuaDogs\\checkpoints\\"

TRAIN_EPOCHS = 100          # 训练轮次
VALIDATE_FREQUENCY = 1000   # 验证频率
RANDOM_SEED = 12345         # 随机种子
# ================================================
```

**步骤 2: 启动训练**

```bash
cd src2
python trains_dog.py
```

**训练输出示例：**
```
[INFO] 开始训练宠物识别模型
[INFO] 使用设备：cuda
[INFO] 初始化模型：PetNet50(...)
[INFO] 训练数据加载器：ImgFolderDataset(...)
#####################################################################################
[INFO] VALIDATING  [1|1]  MAP: 75.23%  AP@1: 76.12%  AP@5: 78.45%  Top-1: 76.12%  NMI: 0.82
#####################################################################################
[INFO] 保存最佳模型：data\TsinghuaDogs\checkpoints\2024-03-05_10-30-00\epoch1-iter1000-map75.23.pth
```

---

### 2️⃣ 初始化特征数据库

**用途：** 为推理系统建立特征索引库

```bash
python src2/init_database.py
```

**输出示例：**
```
[INFO] CPU 核心数：8
[INFO] 总内存：64.0 GB
[INFO] GPU: NVIDIA RTX 3080
[INFO] 初始化特征数据库...
==================================================
性能统计报告:
  处理样本数：65228
  总耗时：125.34 秒
  平均每个样本：1.92 毫秒
  内存使用增加：2048.5 MB
==================================================
```

**生成的数据库：**
```
data/features/
├── pet.index      # FAISS 索引文件
└── pet_metadata.json  # 元数据（标签、路径等）
```

---

### 3️⃣ 推理识别

**方法 1: 命令行（推荐）**

```bash
python src2/infer_dog.py --url https://example.com/dog.jpg
```

**方法 2: 本地图片**

修改 `src2/infer_dog.py` 中的 `query` 函数：

```python
def query(image_path: str):
    image_loader = ImageLoader()
    image = image_loader.load_from_path(image_path)  # 从路径加载
    
    md_path = "data\\TsinghuaDogs\\model\\proxynca-resnet50.pth"
    db_path = "data\\features"
    index_name = "pet"
    query_size = 50
    sort = 3
    
    return infer_dog(image, md_path, db_path, index_name, query_size, sort)
```

然后运行：
```bash
python src2/infer_dog.py --url /path/to/your/dog.jpg
```

**输出示例：**
```
🐕 宠物犬种识别结果：
最可能犬种：金毛寻回犬 (Golden Retriever) ===== 识别置信度：0.9234
最可能犬种：拉布拉多寻回犬 (Labrador Retriever) ===== 识别置信度：0.0512
最可能犬种：英国史宾格犬 (English Springer Spaniel) ===== 识别置信度：0.0198
```

---

## 🔧 高级用法

### 自定义配置训练

**修改 `src2/config/soft_triple_loss.yaml`：**

```yaml
{
    "lr": 0.0001,              # 学习率
    "image_size": 448,         # 图像尺寸
    "embedding_size": 128,     # 特征维度
    "batch_size": 512,         # 批次大小
    "n_workers": 8,            # DataLoader worker 数量
    "accumulation_steps": 2,   # 梯度累积步数
    "n_centers_per_class": 5,  # 每类中心数
    "lambda": 20,              # SoftTriple 参数
    "gamma": 0.1,
    "tau": 0.,
    "margin": 0.01,
    "pretrained": True,        # 使用 ImageNet 预训练权重
}
```

### 批量翻译犬种名称

```python
from src2.util.breed_dictionary_translator import translate_breeds

breeds = ["Bichon Frise", "Poodle", "Shiba Inu"]
chinese_names = translate_breeds(breeds)
print(chinese_names)
# ['比熊犬', '贵宾犬', '柴犬']
```

### 添加自定义犬种到词典

```python
from src2.util.breed_dictionary_translator import BreedDictionaryTranslator

translator = BreedDictionaryTranslator()
translator.add_breed("Cane Corso", "卡斯罗犬")
translator.add_breed("Boerboel", "布尔伯尔犬")

# 使用
print(translator.translate_to_chinese("Cane Corso"))
# 输出：卡斯罗犬
```

---

## 📊 查看训练日志

**TensorBoard 可视化：**

```bash
tensorboard --logdir=data/TsinghuaDogs/checkpoints
```

浏览器打开：http://localhost:6006

**查看内容：**
- 训练损失曲线
- 验证指标（MAP, Top-1, Top-5, NMI）
- 嵌入向量可视化（t-SNE）
- 模型计算图

---

## ⚡ 常见问题解决

### Q1: CUDA Out of Memory

**解决方案 1: 减小批次大小**
```yaml
"batch_size": 256,      # 改为 128 或 64
"accumulation_steps": 4 # 增加累积步数补偿
```

**解决方案 2: 减少图像尺寸**
```yaml
"image_size": 224,  # 从 448 改为 224
```

---

### Q2: DataLoader worker 崩溃

**解决方案：**
```yaml
"n_workers": 2,  # 减少 worker 数量（Windows）
# 或
"n_workers": 0,  # 禁用多进程（调试用）
```

---

### Q3: 找不到模型文件

**检查路径格式：**
```python
# ✅ 正确
MODEL_PATH = "data\\TsinghuaDogs\\model\\proxynca-resnet50.pth"

# ❌ 错误（反斜杠转义）
MODEL_PATH = "data\TsinghuaDogs\model\proxynca-resnet50.pth"
```

---

### Q4: 特征提取速度慢

**优化方案：**
```python
# 在 init_database.py 中调整
n_process = None      # 自动使用 CPU 核心数的一半
batch_size = 1000     # 增加批量大小
```

---

## 🎯 性能调优建议

### GPU 训练（推荐配置）

```yaml
# 8GB+ 显存 GPU
"batch_size": 512
"image_size": 448
"n_workers": 8

# 4GB 显存 GPU
"batch_size": 128
"image_size": 224
"n_workers": 4

# 无 GPU（CPU 训练）
"batch_size": 32
"image_size": 224
"n_workers": 2
```

### 多进程特征提取

```python
# CPU 核心数决定进程数
n_process = None  # 自动检测（推荐）
# 或手动指定
n_process = 4     # 固定 4 个进程
```

---

## 📈 训练进度监控

**实时日志：**
```bash
# Linux/Mac
tail -f src/logs/*.txt

# Windows PowerShell
Get-Content src/logs/*.txt -Tail 50 -Wait
```

**关键指标解读：**
- **MAP (Mean Average Precision)**: 平均精度均值，越高越好
- **AP@1**: 前 1 个检索结果的精度
- **Top-1 Accuracy**: 前 1 个预测的准确率
- **NMI (Normalized Mutual Information)**: 归一化互信息，聚类质量指标

---

## 🔄 模型导出与部署

### 导出 ONNX 格式（计划中）

```python
# TODO: 未来版本支持
import torch

model = PetNet50("path/to/model.pth")
dummy_input = torch.randn(1, 3, 448, 448)
torch.onnx.export(model, dummy_input, "model.onnx")
```

### 模型压缩（计划中）

```python
# TODO: 量化压缩
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

---

## 📞 技术支持

**遇到问题？**

1. 查看详细日志：`src/logs/`
2. 检查 TensorBoard 可视化
3. 参考完整文档：`md/src2 升级说明.md`
4. 检查 GitHub Issues

---

## 🎓 进阶学习

**推荐阅读顺序：**

1. ✅ 本快速手册（入门）
2. 📖 `md/src2 升级说明.md`（详细文档）
3. 📖 原项目 README（理论背景）
4. 📖 Deep Metric Learning 论文

**相关论文：**
- [Proxy-NCA Loss](https://arxiv.org/abs/1703.07464)
- [Soft-Triple Loss](https://arxiv.org/abs/1909.05235)
- [Deep Metric Learning Survey](https://arxiv.org/abs/2202.05388)
