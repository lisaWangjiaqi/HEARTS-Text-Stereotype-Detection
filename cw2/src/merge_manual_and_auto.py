import os
import json
import csv

BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias")

AUTO_FILE = os.path.join(DATA_DIR, "travel_bias_scored.jsonl")      # 自动生成的500条
MANUAL_FILE = os.path.join(DATA_DIR, "travel_bias_for_annotation.csv")  # 你已经标注好的80条
OUTPUT_FILE = os.path.join(DATA_DIR, "final_training_dataset.jsonl")


def load_auto_dataset(path):
    """自动生成数据（已有label字段）"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append({
                "text": obj["text"],
                "label": int(obj.get("label", 0))   # 自动生成已有label
            })
    print(f"Loaded {len(data)} auto-generated samples")
    return data


def load_manual_dataset(path):
    """人工标注的CSV → 转换为jsonl格式"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["human_label"].strip()
            if label == "":
                continue  # 跳过未标注的
            data.append({
                "text": row["text"],
                "label": int(label)
            })
    print(f"Loaded {len(data)} manually labeled samples")
    return data


def main():
    auto_data = load_auto_dataset(AUTO_FILE)
    manual_data = load_manual_dataset(MANUAL_FILE)

    final = auto_data + manual_data

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved final merged dataset: {OUTPUT_FILE}")
    print(f"Total samples: {len(final)}")


if __name__ == "__main__":
    main()

