#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""基于本地词典的犬种翻译器 - 最可靠的解决方案"""

from typing import Dict, List, Optional

class BreedDictionaryTranslator:
    """基于本地词典的犬种翻译器"""
    
    # 完整的犬种英中对照词典
    BREED_DICTIONARY = {
        # 常见犬种
        "Bichon Frise": "比熊犬",
        "Golden Retriever": "金毛寻回犬",
        "Labrador Retriever": "拉布拉多寻回犬",
        "German Shepherd": "德国牧羊犬",
        "French Bulldog": "法国斗牛犬",
        "Poodle": "贵宾犬",
        "Chihuahua": "吉娃娃",
        "Pomeranian": "博美犬",
        "Shih Tzu": "西施犬",
        "Yorkshire Terrier": "约克夏梗",
        "Maltese": "马尔济斯犬",
        "Dachshund": "腊肠犬",
        "Boxer": "拳师犬",
        "Bulldog": "英国斗牛犬",
        "Rottweiler": "罗威纳犬",
        "Doberman": "杜宾犬",
        "Siberian Husky": "西伯利亚哈士奇",
        "Alaskan Malamute": "阿拉斯加雪橇犬",
        "Samoyed": "萨摩耶犬",
        "Border Collie": "边境牧羊犬",
        "Australian Shepherd": "澳大利亚牧羊犬",
        "Cocker Spaniel": "可卡犬",
        "Cardigan Welsh Corgi": "卡迪根威尔士柯基犬",
        "Pembroke Welsh Corgi": "彭布罗克威尔士柯基犬",
        "Welsh Corgi": "威尔士柯基犬",
        "Springer Spaniel": "史宾格犬",
        "Beagle": "比格犬",
        "Dalmatian": "大麦町犬",
        "Great Dane": "大丹犬",
        "Saint Bernard": "圣伯纳德犬",
        "Chow Chow": "松狮犬",
        "Shar Pei": "沙皮犬",
        "Akita": "秋田犬",
        "Shiba Inu": "柴犬",
        "Basenji": "巴仙吉犬",
        "Afghan Hound": "阿富汗猎犬",
        "Greyhound": "灵缇犬",
        "Whippet": "惠比特犬",
        "Italian Greyhound": "意大利灵缇",
        "Bloodhound": "寻血猎犬",
        "Pointer": "指示犬",
        "Setter": "赛特犬",
        "Spaniel": "史宾格犬",
        "Terrier": "梗犬",
        "Hound": "猎犬",
        "Mastiff": "獒犬",
        "Mountain Dog": "山地犬",
        "Shepherd": "牧羊犬",
        "Retriever": "寻回犬",
        "Toy": "玩具犬",
        "Working": "工作犬",
        "Herding": "牧羊犬",
        "Hunting": "猎犬",
        "Companion": "伴侣犬",
        
        # 更多犬种
        "Cavalier King Charles Spaniel": "骑士查理王小猎犬",
        "Boston Terrier": "波士顿梗",
        "Scottish Terrier": "苏格兰梗",
        "West Highland White Terrier": "西高地白梗",
        "Soft Coated Wheaten Terrier": "软毛小麦色梗",
        "Irish Setter": "爱尔兰赛特犬",
        "English Setter": "英国赛特犬",
        "Gordon Setter": "戈登赛特犬",
        "Brittany": "布列塔尼犬",
        "Clumber Spaniel": "克拉姆勃猎犬",
        "English Springer Spaniel": "英国史宾格犬",
        "Welsh Springer Spaniel": "威尔士史宾格犬",
        "Sussex Spaniel": "苏塞克斯猎犬",
        "Irish Water Spaniel": "爱尔兰水猎犬",
        "Nova Scotia Duck Tolling Retriever": "新斯科舍猎鸭寻回犬",
        "Curly Coated Retriever": "卷毛寻回犬",
        "Flat Coated Retriever": "平毛寻回犬",
        "Chesapeake Bay Retriever": "切萨皮克湾寻回犬",
        "Vizsla": "维兹拉犬",
        "Wirehaired Vizsla": "刚毛维兹拉犬",
        "German Wirehaired Pointer": "德国刚毛指示犬",
        "German Shorthaired Pointer": "德国短毛指示犬",
        "Weimaraner": "魏玛犬",
        "Rhodesian Ridgeback": "罗得西亚背脊犬",
        "Pharaoh Hound": "法老猎犬",
        "Icelandic Sheepdog": "冰岛牧羊犬",
        "Finnish Spitz": "芬兰狐狸犬",
        "Swedish Vallhund": "瑞典农场犬",
        "Norwegian Elkhound": "挪威猎鹿犬",
        "Finnish Lapphund": "芬兰拉普猎犬",
        "Lapponian Herder": "拉普兰牧犬",
        "Chinese Rural Dog": "中华田园犬",
        "Pomeranian": "博美犬",
        "Malamute": "阿拉斯加雪橇犬",
        "Alaskan Malamute": "阿拉斯加雪橇犬",
        "Samoyed": "萨摩耶犬",
    }
    
    def __init__(self):
        self.dictionary = self.BREED_DICTIONARY.copy()
        # 创建反向词典用于双向翻译
        self.reverse_dictionary = {v: k for k, v in self.dictionary.items()}
        
    def translate_to_chinese(self, label: str) -> str:
        """
        英文犬种翻译成中文
        
        Args:
            english_breed: 英文犬种名称
            
        Returns:
            中文犬种名称，找不到则返回原文
        """
        if not label:
            return ""
        
        # 处理标签格式: n000128-teddy -> teddy
        parts = label.split("-")
        if len(parts) > 1:
            english_breed = parts[-1]  # 取最后一位
            # 替换下划线为空格，然后首字母大写
            english_breed = english_breed.replace("_", " ").title()
        else:
            english_breed = label.replace("_", " ").title()
        
        # 精确匹配
        if english_breed in self.dictionary:
            return self.dictionary[english_breed]
        
        # 去掉复数形式后匹配
        singular_form = english_breed.rstrip('s')
        if singular_form in self.dictionary:
            return self.dictionary[singular_form]
        
        # 部分匹配（模糊搜索）
        for key, value in self.dictionary.items():
            if key.lower() in english_breed.lower() or english_breed.lower() in key.lower():
                return value
        
        # 返回原文
        return english_breed
    
    def translate_to_english(self, chinese_breed: str) -> str:
        """
        中文犬种翻译成英文
        
        Args:
            chinese_breed: 中文犬种名称
            
        Returns:
            英文犬种名称，找不到则返回原文
        """
        if not chinese_breed:
            return ""
            
        # 精确匹配
        if chinese_breed in self.reverse_dictionary:
            return self.reverse_dictionary[chinese_breed]
        
        # 部分匹配
        for key, value in self.reverse_dictionary.items():
            if key in chinese_breed or chinese_breed in key:
                return value
        
        # 返回原文
        return chinese_breed
    
    def add_breed(self, english: str, chinese: str):
        """添加新的犬种翻译"""
        self.dictionary[english] = chinese
        self.reverse_dictionary[chinese] = english
    
    def batch_translate_to_chinese(self, english_breeds: List[str]) -> List[str]:
        """批量翻译英文犬种到中文"""
        return [self.translate_to_chinese(breed) for breed in english_breeds]
    
    def batch_translate_to_english(self, chinese_breeds: List[str]) -> List[str]:
        """批量翻译中文犬种到英文"""
        return [self.translate_to_english(breed) for breed in chinese_breeds]
    
    def get_all_breeds(self) -> Dict[str, str]:
        """获取所有犬种翻译"""
        return self.dictionary.copy()

# 便利函数
def translate_breed(english_breed: str) -> str:
    """快速翻译单个犬种"""
    translator = BreedDictionaryTranslator()
    return translator.translate_to_chinese(english_breed)

def translate_breeds(english_breeds: List[str]) -> List[str]:
    """批量翻译犬种列表"""
    translator = BreedDictionaryTranslator()
    return translator.batch_translate_to_chinese(english_breeds)

def enhance_breed_display(breed_scores: List[tuple]) -> List[tuple]:
    """
    增强犬种显示，添加中文翻译
    
    Args:
        breed_scores: [(英文犬种, 得分), ...] 的列表
        
    Returns:
        [(英文犬种, 中文犬种, 得分), ...] 的列表
    """
    translator = BreedDictionaryTranslator()
    enhanced_results = []
    
    for breed, score in breed_scores:
        chinese_name = translator.translate_to_chinese(breed)
        enhanced_results.append((breed, chinese_name, score))
        
    return enhanced_results

# 测试函数
def test_breed_translator():
    """测试犬种翻译器"""
    print("=" * 50)
    print("🐕 犬种词典翻译器测试")
    print("=" * 50)
    
    translator = BreedDictionaryTranslator()
    
    # 测试用例
    test_breeds = [
        "Bichon Frise",
        "Golden Retriever",
        "Labrador Retriever", 
        "German Shepherd",
        "French Bulldog",
        "Unknown Breed"  # 测试未知品种
    ]
    
    print("\n🔤 英文转中文测试:")
    print("-" * 30)
    for breed in test_breeds:
        chinese = translator.translate_to_chinese(breed)
        print(f"{breed:25} -> {chinese}")
    
    # 双向翻译测试
    print("\n🔄 双向翻译测试:")
    print("-" * 30)
    chinese_breed = "比熊犬"
    english_result = translator.translate_to_english(chinese_breed)
    print(f"{chinese_breed} -> {english_result}")
    
    # 批量翻译测试
    print("\n📦 批量翻译测试:")
    print("-" * 30)
    batch_result = translator.batch_translate_to_chinese(test_breeds[:3])
    for eng, chn in zip(test_breeds[:3], batch_result):
        print(f"{eng} -> {chn}")

if __name__ == "__main__":
    test_breed_translator()
    
    print("\n" + "=" * 50)
    print("✅ 犬种翻译器测试完成")
    print("=" * 50)