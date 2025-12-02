import json
import re
import os
import random

# ============================================
# 固定输入输出
# ============================================
INPUT = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src/data_travel_bias/travel_bias_hard_v1.jsonl"
OUTPUT = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src/data_travel_bias/travel_bias_hard_v2.jsonl"
REMOVED = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src/data_travel_bias/removed_v2.jsonl"

print(">>> Cleaning dataset")
print(">>> INPUT :", INPUT)
print(">>> OUTPUT:", OUTPUT)


# ============================================
# 1) 长度过滤（neutral 更宽松）
# ============================================
def length_valid(text, min_w=6, max_w=95):
    wc = len(text.split())
    return min_w <= wc <= max_w


# ============================================
# 2) LLM-style stereotype（必须删除）
# ============================================
FAKE_STEREO = [
    r"residents tend to",
    r"locals tend to",
    r"people tend to",
    r"residents often",
    r"locals often",
    r"people often",
    r"are widely seen as",
    r"are commonly perceived as",
    r"is stereotyped as",
]

FAKE_STEREO = [re.compile(p, re.IGNORECASE) for p in FAKE_STEREO]


# ============================================
# 3) Semantic stereotype protector
#   —— 避免你的 semantic stereotype 被错误删掉
# ============================================
SEMANTIC_STEREO_KEYWORDS = [
    "momentum", "reflection", "order", "constraint",
    "identity", "surface", "impression", "refinement",
    "structure", "pattern", "narrative", "assertiveness",
    "restraint", "composition", "calibrated",
    "symbolic", "curated", "tension"
]

SEMANTIC_STEREO_REGEX = re.compile(
    "(" + "|".join(SEMANTIC_STEREO_KEYWORDS) + ")", re.IGNORECASE
)


def is_semantic_stereo(text):
    """判断是否属于真正的语义 stereotype"""
    return SEMANTIC_STEREO_REGEX.search(text) is not None


# ============================================
# 4) Neutral 允许的句型（不再误删）
# ============================================
NEUTRAL_ALLOWED = [
    r"is the capital of",
    r"is home to",
    r"contains",
    r"includes",
    r"features",
    r"is divided into",
    r"developed through",
    r"expanded during",
    r"historical phase",
    r"administrative district",
]

NEUTRAL_ALLOWED = [re.compile(p, re.IGNORECASE) for p in NEUTRAL_ALLOWED]


def is_allowed_neutral(text):
    return any(p.search(text) for p in NEUTRAL_ALLOWED)


# ============================================
# 5) 城市名匿名化（更强 regex）
# ============================================
CITY_PATTERNS = [
    "Shanghai","Beijing","Guangzhou","Shenzhen","Chengdu",
    "Hangzhou","Xian","Xi’an","Xi'an","Chongqing","Nanjing",
    "Wuhan","Shenyang","Tianjin","Qingdao","Jinan",
    "Changsha","Zhengzhou","Xinjiang","Kunming","Xiamen",
    "Harbin","Suzhou","Fuzhou","Dalian"
]

CITY_REGEX = re.compile("|".join([fr"\b{c}\b" for c in CITY_PATTERNS]), re.IGNORECASE)

def anonymize_city(text):
    return CITY_REGEX.sub("{CITY}", text)


# ============================================
# 6) LLM 模板句（但 neutral 不再删除）
# ============================================
STYLE_TEMPLATES = [
    r"inviting visitors to",
    r"offers a tranquil",
    r"is renowned for",
    r"in ways that are not essential",
    r"though interpretations vary widely",
]

STYLE_TEMPLATES = [re.compile(p, re.IGNORECASE) for p in STYLE_TEMPLATES]


# ============================================
# 7) 主清洗流程（v3）
# ============================================
def clean_dataset(input_path, output_path):
    cleaned = []
    removed = []
    seen = set()

    with open(input_path, "r") as fin:
        for line in fin:
            if not line.strip():
                continue

            item = json.loads(line)
            text = item["text"]
            label = item["label"]

            # ---- (1) 匿名化 ----
            text = anonymize_city(text)

            # ---- (2) 长度 ----
            if not length_valid(text):
                removed.append(item)
                continue

            # ---- (3) 假 stereotype 删除 —— 但语义 stereotype 必须保留！
            if any(p.search(text) for p in FAKE_STEREO):
                if not is_semantic_stereo(text):
                    removed.append(item)
                    continue

            # ---- (4) Neutral 不删除 allowed pattern
            if label == 0 and is_allowed_neutral(text):
                pass  # 强制保留

            # ---- (5) stereotype 不允许 AI 模板句
            if label == 1 and any(p.search(text) for p in STYLE_TEMPLATES):
                removed.append(item)
                continue

            # ---- (6) 去重 ----
            h = hash(text)
            if h in seen:
                continue
            seen.add(h)

            cleaned.append({"text": text, "label": label})

    # ---- (7) 打乱 ----
    random.shuffle(cleaned)

    # ---- 输出 ----
    with open(output_path, "w", encoding="utf-8") as fout:
        for obj in cleaned:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(REMOVED, "w", encoding="utf-8") as fout:
        for obj in removed:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ---- 统计 ----
    c0 = sum(1 for x in cleaned if x["label"] == 0)
    c1 = sum(1 for x in cleaned if x["label"] == 1)

    print("\n>>> 清洗完成!")
    print("原始数据:", len(cleaned) + len(removed))
    print("保留数据:", len(cleaned))
    print(f"其中 Neutral = {c0}")
    print(f"其中 Stereotype = {c1}")
    print("移除数据:", len(removed))


# ============================================
# RUN
# ============================================
if __name__ == "__main__":
    clean_dataset(INPUT, OUTPUT)
