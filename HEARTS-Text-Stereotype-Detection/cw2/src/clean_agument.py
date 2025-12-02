import json
import re
import random
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# ============================================
# 输入 / 输出路径（根据你自己的环境修改）
# ============================================
INPUT = "data_travel_bias/travel_bias_hard_v2.jsonl"
OUTPUT = "data_travel_bias/travel_bias_clean_aug.jsonl"
REMOVED = "data_travel_bias/removed_clean_aug.jsonl"

print(">>> Loading:", INPUT)


# ============================================
# 1) destroy_style：破坏所有写作风格
# ============================================

PUNCT = r"[.,!?;:\(\)\[\]\"']"

def destroy_style(text):
    # 删除标点
    text = re.sub(PUNCT, " ", text)

    # 分词
    words = word_tokenize(text)

    # 打乱词序（核心）
    random.shuffle(words)

    return " ".join(words)


# ============================================
# 2) syntax_destroy：破坏句法结构但不破坏语义 token
# ============================================

def syntax_destroy(text):
    """
    打乱局部邻域 + 随机剪断句子结构
    """
    words = word_tokenize(text)

    # 若句子太短，不做句法破坏
    if len(words) < 6:
        return text

    # 局部窗口打乱（不破坏整体主题词）
    window = 3
    new_words = []

    for i in range(0, len(words), window):
        chunk = words[i:i+window]
        random.shuffle(chunk)
        new_words.extend(chunk)

    # 随机截断（20% 概率）
    if random.random() < 0.2:
        cut = random.randint(5, len(new_words))
        new_words = new_words[:cut]

    return " ".join(new_words)


# ============================================
# 3) semantic_augmentation：生成语义保持版本
# ============================================

SEMANTIC_TEMPLATES = [
    "{CITY} is described in various ways depending on context, without implying shared behaviors.",
    "Descriptions of {CITY} often refer to its structural or spatial characteristics rather than assumptions about people.",
    "{CITY}'s appearance has been shaped by geography and planning choices, not collective cultural traits.",
    "Observers of {CITY} may highlight different aspects of its landscape, none of which assert behavioral tendencies.",
    "The identity of {CITY} can be discussed in terms of form and layout without implying social generalizations.",
]

STEREO_TEMPLATES = [
    "{CITY} gives an impression of movement driven by underlying expectations rather than explicit choices.",
    "{CITY} presents a rhythm that feels shaped by collective tendencies beneath the visible surface.",
    "{CITY} conveys a tone suggesting shared patterns influencing how the city seems to function.",
]


def semantic_augment(item):
    """
    针对 neutral / stereotype 分别增强语义
    """

    text = item["text"]
    label = item["label"]

    aug_items = []

    # ---- neutral 数据增强 ----
    if label == 0:
        for _ in range(2):
            template = random.choice(SEMANTIC_TEMPLATES)
            aug_items.append({
                "text": template,
                "label": 0
            })

    # ---- stereotype 数据增强 ----
    else:
        for _ in range(2):
            template = random.choice(STEREO_TEMPLATES)
            aug_items.append({
                "text": template,
                "label": 1
            })

    return aug_items


# ============================================
# 4) 主清洗 + 风格破坏 + 增强流程
# ============================================

def process_dataset(input_path, output_path, removed_path):
    cleaned = []
    removed = []

    with open(input_path, "r") as fin:
        for line in fin:
            if not line.strip():
                continue

            item = json.loads(line)
            text = item["text"]

            # ---- 破坏风格（强制打乱）----
            t1 = destroy_style(text)

            # ---- 破坏句法 ----
            t2 = syntax_destroy(t1)

            # 创建新 item
            new_item = {
                "text": t2,
                "label": item["label"]
            }

            cleaned.append(new_item)

            # ---- 添加语义增强 ----
            aug_items = semantic_augment(item)
            cleaned.extend(aug_items)

    # 打乱整个 dataset（非常关键）
    random.shuffle(cleaned)

    # 保存
    with open(output_path, "w") as fout:
        for x in cleaned:
            fout.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f">>> Clean + Aug Done")
    print(f"Output: {len(cleaned)} samples → {output_path}")


# ============================================
# RUN
# ============================================

process_dataset(INPUT, OUTPUT, REMOVED)

print(">>> All done!")

