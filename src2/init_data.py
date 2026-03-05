import os
import glob
import shutil


def split_train_val(images_dir: str, output_dir: str, lst_path: str):
    """
    根据 lst 文件中的图片列表，将图片复制到对应的输出目录

    Args:
        images_dir: 原始图片目录 (high-resolution/)
        output_dir: 输出目录 (train/ 或 val/)
        lst_path: 图片列表文件 (train.lst 或 validation.lst)
    """
    with open(lst_path, "r", encoding="utf-8") as f:
        # 读取图片名称列表
        lst = set([os.path.basename(line.strip()) for line in f.readlines()])

    for image_name in lst:
        # 找到图片路径
        image_path = glob.glob(os.path.join(
            images_dir, "*", image_name), recursive=True)[0]
        class_name = os.path.basename(os.path.dirname(image_path))

        # 创建类别目录并复制图片
        dest_dir = os.path.join(output_dir, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        shutil.copy(image_path, dest_dir)

    
if __name__ == "__main__":
    split_train_val("data/TsinghuaDogs/high-resolution/", "data/TsinghuaDogs/train/", "data/TsinghuaDogs/TrainAndValList/train.lst")
    split_train_val("data/TsinghuaDogs/high-resolution/", "data/TsinghuaDogs/val/", "data/TsinghuaDogs/TrainAndValList/validation.lst")
