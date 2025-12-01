import json
import os

BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias")

INPUT_FILE = os.path.join(DATA_DIR, "final_training_dataset.jsonl")   # 你的合并后数据
OUTPUT_FILE = os.path.join(DATA_DIR, "final_training_dataset_clean.jsonl")


# 典型拒答 / 安全机制触发关键字（可继续扩展）
REFUSAL_PATTERNS = [
    "I apologize",
    "I apologise",
    "I cannot",
    "I can’t",
    "cannot assist",
    "I do not feel comfortable",
    "I don't feel comfortable",
    "I prefer not",
    "As an AI",
    "I am unable",
    "I strive",
    "I aim",
    "I avoid",
    "I will not",
    "I cannot generate",
    "I do not generate",
    "I’m not able",
    "I won't provide",
    "I must avoid",
]


def is_refusal(text: str) -> bool:
    """判断这个文本是否属于 AI 拒答 / 安全提示"""
    t = text.lower()
    for pat in REFUSAL_PATTERNS:
        if pat.lower() in t:
            return True
    return False


def main():

    cleaned = []
    dropped = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            txt = obj["text"]

            if is_refusal(txt):
                dropped.append(obj)
            else:
                cleaned.append(obj)

    # 输出干净数据
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for obj in cleaned:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"\n==== 清洗完成 ====")
    print(f"原始样本数: {len(cleaned) + len(dropped)}")
    print(f"保留样本数: {len(cleaned)}")
    print(f"删除拒答样本数: {len(dropped)}")

    # 打印几个被删除的例子检查
    print("\n被删除的典型拒答句子示例:")
    for d in dropped[:5]:
        print(" -", d["text"][:120])


if __name__ == "__main__":
    main()

