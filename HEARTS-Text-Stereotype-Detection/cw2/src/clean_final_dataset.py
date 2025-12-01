import json
import os

BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias")

INPUT_FILE = os.path.join(DATA_DIR, "travel_bias_dataset.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "travel_bias_dataset_clean.jsonl")

# 典型拒答 / 安全机制触发关键字
REFUSAL_PATTERNS = [
    "I apologize", "I apologise",
    "I cannot", "I can’t",
    "cannot assist",
    "I do not feel comfortable", "I don't feel comfortable",
    "I prefer not",
    "As an AI",
    "I am unable", "I’m not able",
    "I strive", "I aim",
    "I avoid",
    "I will not", "I won't",
    "I cannot generate", "I do not generate",
]


def is_refusal(text: str) -> bool:
    """判断是否是拒绝生成 stereotype 的句子。"""
    t = text.lower()
    for pat in REFUSAL_PATTERNS:
        if pat.lower() in t:
            return True
    return False


def main():

    print("\n==== 开始清洗数据（拒答过滤 + 去重） ====\n")

    cleaned = []
    dropped_refusal = []
    dropped_duplicate = []

    seen_texts = set()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            txt = obj["text"].strip()

            # 1) 拒答过滤
            if is_refusal(txt):
                dropped_refusal.append(obj)
                continue

            # 2) 文本去重
            key = txt.lower()
            if key in seen_texts:
                dropped_duplicate.append(obj)
                continue

            seen_texts.add(key)
            cleaned.append(obj)

    # 写 clean 文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for obj in cleaned:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("==== 清洗完成 ====\n")
    print(f"原始样本数:           {len(cleaned) + len(dropped_refusal) + len(dropped_duplicate)}")
    print(f"保留样本数:           {len(cleaned)}")
    print(f"删除拒答样本数:       {len(dropped_refusal)}")
    print(f"删除重复样本数:       {len(dropped_duplicate)}")

    # 打印示例
    print("\n部分被删除的拒答句:")
    for d in dropped_refusal[:5]:
        print(" -", d["text"][:120])

    print("\n部分被删除的重复句:")
    for d in dropped_duplicate[:5]:
        print(" -", d["text"][:120])


if __name__ == "__main__":
    main()
