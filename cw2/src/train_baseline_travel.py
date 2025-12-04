import json
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn.functional as F


# ============================================================
# 1. Load travel-bias JSONL dataset
# ============================================================

jsonl_path = "Travelbias_dataset/travel_bias_hard_v2.jsonl"

print("Loading travel bias dataset...")

data_list = []
with open(jsonl_path, "r") as f:
    for line in f:
        if line.strip():
            data_list.append(json.loads(line))

dataset = Dataset.from_list(data_list)

print("Example:", dataset[0])
print(f"Total samples: {len(dataset)}")


# ============================================================
# 2. Load EMGSD baseline (你的最佳 checkpoint)
# ============================================================

EMGSD_MODEL_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src/results/emgsd_baseline_albert/checkpoint-11440"

print("\nLoading EMGSD baseline model...")
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = AutoModelForSequenceClassification.from_pretrained(EMGSD_MODEL_DIR)
model.eval()
model.cuda()


# ============================================================
# 3. EMGSD 的 label 映射（必须与训练时一致）
# ============================================================

# 你的 EMGSD baseline 有 13 个 label，例如：
#   neutral_gender, neutral_race, ...
#   stereotype_gender, stereotype_race, ...
# 这里只要根据名称合并为 0/1 即可

id2label = model.config.id2label

NEUTRAL_LABELS = [l for l in id2label.values() if l.startswith("neutral")]
STEREO_LABELS  = [l for l in id2label.values() if l.startswith("stereotype")]

stereo_ids = set([k for k, v in id2label.items() if v in STEREO_LABELS])

print("\n=== Label Groups ===")
print("Neutral labels:", NEUTRAL_LABELS)
print("Stereotype labels:", STEREO_LABELS)
print("Stereo label IDs:", stereo_ids)


# ============================================================
# 4. Run evaluation — No training, pure prediction
# ============================================================

texts = dataset["text"]
true_labels = dataset["label"]

pred_labels = []

BATCH = 32

print("\nRunning baseline EMGSD on TravelBias dataset...")

for i in range(0, len(texts), BATCH):

    batch_text = texts[i:i+BATCH]

    enc = tokenizer(
        batch_text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        logits = model(**enc).logits   # shape: (batch, 13)
        preds = torch.argmax(logits, dim=-1).cpu().tolist()

    # 将 13 类结果映射到 0/1
    for p in preds:
        if p in stereo_ids:
            pred_labels.append(1)
        else:
            pred_labels.append(0)


# ============================================================
# 5. Compute binary metrics
# ============================================================

acc = accuracy_score(true_labels, pred_labels)
macro_f1 = f1_score(true_labels, pred_labels, average="macro")

print("\n=== Baseline EMGSD Performance on TravelBias Dataset ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Macro-F1:  {macro_f1:.4f}")
print("\n(这将作为对比你的“改进模型”的 baseline)")


# ============================================================
# 6. Print examples (optional)
# ============================================================

# print("\nExample Predictions:")
# for i in range(5):
#     print("- Text:", texts[i][:80], "...")
#     print("  True label:", true_labels[i])
#     print("  Pred label:", pred_labels[i])
