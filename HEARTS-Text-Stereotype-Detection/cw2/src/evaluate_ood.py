# ============================================
# File: evaluate_ood_with_details.py
# ============================================

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

OOD_PATH = "data_travel_bias/travel_bias_OOD.jsonl"
MODEL_DIR = "results/improved_roberta/checkpoint-129"

# ================================
# device 自动检测
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================================
# Load model + tokenizer
# ================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# ================================
# Load OOD data
# ================================
data = [json.loads(l) for l in open(OOD_PATH)]
texts = [d["text"] for d in data]
labels = [d["label"] for d in data]

# ================================
# Tokenize
# """ (num_samples,) -> (num_samples, seq_len) """
# ================================
enc = tokenizer(
    texts,
    padding=True,          # pad to max batch length
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# Move tensors to device
enc = {k: v.to(device) for k, v in enc.items()}

# ================================
# Predict
# logits: """ (batch, 2) """
# preds:  """ (batch,) """
# ================================
with torch.no_grad():
    logits = model(**enc).logits
preds = torch.argmax(logits, dim=-1).cpu().tolist()

# ================================
# Print each sample result
# ================================
print("\n===== OOD Sample Predictions =====\n")

for i, (t, y_true, y_pred) in enumerate(zip(texts, labels, preds)):
    print(f"--- Sample {i+1} ---")
    print("Text:", t)
    print("True Label :", y_true)
    print("Pred Label :", y_pred)
    print()

# ================================
# Metrics
# ================================
acc = accuracy_score(labels, preds)
macro_f1 = f1_score(labels, preds, average="macro")

print("\n===== OOD Evaluation =====")
print("Accuracy:", acc)
print("Macro-F1:", macro_f1)
