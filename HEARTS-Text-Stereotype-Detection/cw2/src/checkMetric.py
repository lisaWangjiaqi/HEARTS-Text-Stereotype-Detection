# filename: checkMetric.py
import os
import json
import argparse
import numpy as np
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# --------------------------------------------------
# 1. 解析命令行参数
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True,
                    help="Path to trained model folder")
parser.add_argument("--data_path", type=str, default="travel_bias_dataset.jsonl",
                    help="Path to your travel dataset JSONL")
args = parser.parse_args()

model_dir = args.model_dir
data_path = args.data_path

print(f"🟢 Using model: {model_dir}")

# --------------------------------------------------
# 2. 加载模型 & tokenizer
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval().cuda()

# --------------------------------------------------
# 3. 加载数据集（JSONL → HF Dataset）
# --------------------------------------------------
def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

raw_data = load_jsonl(data_path)
texts = [d["text"] for d in raw_data]
labels = [d["label"] for d in raw_data]

dataset = Dataset.from_dict({"text": texts, "label": labels})

# --------------------------------------------------
# 4. Tokenization
# --------------------------------------------------
def preprocess(batch):
    enc = tokenizer(batch["text"], truncation=True, padding=False)
    enc["labels"] = batch["label"]
    return enc

ds_enc = dataset.map(preprocess)

# 去掉无法 tensor 化的列
cols_to_remove = [c for c in ds_enc.column_names if c not in ["input_ids", "attention_mask", "labels"]]
ds_enc = ds_enc.remove_columns(cols_to_remove)

# --------------------------------------------------
# 5. 推理
# --------------------------------------------------
all_preds = []
all_labels = labels

print("🔄 Running inference...")

for i in range(len(ds_enc)):
    item = {k: torch.tensor([v]).cuda() for k, v in ds_enc[i].items()}
    with torch.no_grad():
        logits = model(**item).logits
        pred = torch.argmax(logits, dim=1).item()
        all_preds.append(pred)

print("✅ Inference complete")

# --------------------------------------------------
# 6. 计算指标
# --------------------------------------------------
accuracy = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average="macro")
weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

report = classification_report(
    all_labels,
    all_preds,
    digits=4
)

print("\n======== 📊 FINAL METRICS ========")
print("Accuracy:", accuracy)
print("Macro-F1:", macro_f1)
print("Weighted-F1:", weighted_f1)

print("\nPer-class report:\n")
print(report)

# --------------------------------------------------
# 7. 保存结果到 metrics.json
# --------------------------------------------------
metrics = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,
    "samples": len(all_labels)
}

save_path = os.path.join(model_dir, "metrics.json")
with open(save_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\n📁 Saved metric → {save_path}")
