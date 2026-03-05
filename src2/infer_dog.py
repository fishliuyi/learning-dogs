

# 添加项目路径到Python路径
from pathlib import Path
import argparse
import sys
import PIL.Image as Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "reference"))
sys.path.insert(0, str(Path(__file__).parent / "util"))
from util.imgage import ImageLoader
from util.breed_dictionary_translator import translate_breed
from reference import Log, infer_dog

loger = Log("infer_dog")


def query(url: str):
    image_loader = ImageLoader()
    image: Image = image_loader.load_from_url(url)

    md_path = "data/TsinghuaDogs/model/proxynca-resnet50.pth"
    db_path = "data/features"
    index_name = "pet"
    query_size = 50
    sort = 3
    
    return infer_dog(image, md_path, db_path, index_name, query_size, sort)


def main():
    parser = argparse.ArgumentParser(description="犬种识别系统")
    parser.add_argument("--url", type=str, required=True, help="输入图片路径")
    args = parser.parse_args()
    breed_scores = query(args.url)
    print("🐕 宠物犬种识别结果：")
    for label, score in breed_scores:
        name = translate_breed(label)
        print(f"最可能犬种: {name} ({label}) ===== 识别置信度: {score:.4f}")


if __name__ == "__main__":
    main()
