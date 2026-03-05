import sys
from pathlib import Path
import os

# Windows 系统必须使用 spawn 方式启动多进程
import multiprocessing
if sys.platform == 'win32':
    multiprocessing.set_start_method('spawn', force=True)

# 禁用 TensorFlow oneDNN 提示
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "reference"))
from reference import Log, CreateTrainMOdel


def main():
    """
    主函数 - 用于启动训练流程

    使用方法：
    1. 修改下方的配置参数
    2. 运行 python train_model.py
    """
    # ==================== 配置区域 ====================
    CONFIG_PATH = "src2/config/soft_triple_loss.yaml"  # 配置文件路径
    MODEL_PATH = "data/TsinghuaDogs/model/train-resnet50.pth"  # 预训练模型路径名
    DATA_PATH = "data/TsinghuaDogs/train/"  # 训练数据路径
    TEST_PATH = "data/TsinghuaDogs/val/"  # 验证数据路径
    CHECKPOINT_PATH = "data/TsinghuaDogs/checkpoints/"  # 检查点保存路径

    TRAIN_EPOCHS = 100  # 训练轮次
    VALIDATE_FREQUENCY = 1000  # 验证频率（迭代次数）
    RANDOM_SEED = 12345  # 随机种子
    # ================================================

    try:
        logger = Log('trains_dog').logger
        logger.info("开始训练宠物识别模型")

        # 初始化训练器
        trainer = CreateTrainMOdel(
            config_path=CONFIG_PATH,
            md_path=MODEL_PATH,
            data_path=DATA_PATH,
            test_path=TEST_PATH,
            chk_path=CHECKPOINT_PATH,
            train_epochs=TRAIN_EPOCHS,
            validate_frequency=VALIDATE_FREQUENCY,
            random_seed=RANDOM_SEED
        )

        # 开始训练
        result = trainer.train()

        logger.info("=" * 80)
        logger.info("训练完成！")
        logger.info(
            f"最终 MAP: {result['metrics']['mean_average_precision']:.2f}%")
        logger.info("=" * 80)

    except FileNotFoundError as e:
        print(f"❌ 文件未找到错误：{e}")
        print("\n请检查以下路径是否正确:")
        print(f"  - 配置文件：{CONFIG_PATH}")
        print(f"  - 模型文件：{MODEL_PATH}")
        print(f"  - 数据目录：{DATA_PATH}")
        print(f"  - 测试目录：{TEST_PATH}")
        sys.exit(1)

    except Exception as e:
        print(f"❌ 训练过程中发生错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
