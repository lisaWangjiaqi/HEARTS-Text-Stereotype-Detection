import os
import json
from typing import List, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ============================================================
# 配置区
# ============================================================

BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias")

INPUT_FILE = os.path.join(DATA_DIR, "travel_bias_dataset.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "travel_bias_scored.jsonl")

# 训练时使用的 tokenizer （非常关键）
TOKENIZER_DIR = "albert-base-v2"

# 你的 fine-tuned 模型权重（只包含模型，不包含 tokenizer）
MODEL_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/results/emgsd/checkpoint-8580"

BATCH_SIZE = 32
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Dataset
# ============================================================

class JsonlTextDataset(Dataset):
    """读取 jsonl，每行包含至少 text、label"""

    def __init__(self, path: str):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(obj)
        print(f"Loaded {len(self.samples)} samples from {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """返回 DataLoader batch，同时保留原始对象"""
    return {
        "text": [item["text"] for item in batch],
        "label": [item.get("label") for item in batch],
        "__orig__": batch,
    }


# ============================================================
# 打分主逻辑
# ============================================================

def score_dataset():

    # 1. 加载 tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()

    # 2. DataLoader
    dataset = JsonlTextDataset(INPUT_FILE)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    output = []

    # 3. 主循环：对每个 batch 评分
    for batch in dataloader:

        texts = batch["text"]

        # Tokenize
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**enc).logits         # (batch, 2)
            probs = F.softmax(logits, dim=-1)    # softmax

        # stereotype probability = P(label=1)
        scores = probs[:, 1].cpu().tolist()

        # 保存结果：text + 原标签 + score
        for orig, s in zip(batch["__orig__"], scores):
            output.append({
                "text": orig["text"],
                "label": orig.get("label"),
                "stereo_score": float(s)
            })

    # 4. 写出 jsonl
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for obj in output:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"\nSaved scored dataset to: {OUTPUT_FILE}")

    scores_only = [o["stereo_score"] for o in output]
    print(f"Score range: {min(scores_only):.4f}  ~  {max(scores_only):.4f}")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    score_dataset()
