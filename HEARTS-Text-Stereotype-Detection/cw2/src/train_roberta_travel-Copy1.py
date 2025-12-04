import json
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ============================================================
# 1. Load TravelBias Dataset
# ============================================================

DATA_PATH = "Travelbias_dataset/merged_dataset.jsonl"

data_list = []
with open(DATA_PATH, "r") as f:
    for line in f:
        if line.strip():
            data_list.append(json.loads(line))

dataset = Dataset.from_list(data_list)

dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_val = dataset["train"].train_test_split(test_size=0.2, seed=42)

train_ds = train_val["train"]
val_ds   = train_val["test"]
test_ds  = dataset["test"]


# ============================================================
# 2. Tokenizer
# ============================================================

MODEL_NAME = "roberta-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    enc["labels"] = batch["label"]
    return enc

train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(preprocess, remove_columns=val_ds.column_names)
test_ds  = test_ds.map(preprocess, remove_columns=test_ds.column_names)

train_ds.set_format("torch")
val_ds.set_format("torch")
test_ds.set_format("torch")


# ============================================================
# 3. Model （自动保存 config.json）
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# ============================================================
# 4. Metrics
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


# ============================================================
# 5. TrainingArguments
# ============================================================

args = TrainingArguments(
    output_dir="results/improved_roberta_merge",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_safetensors=True,
    save_total_limit=1,
    load_best_model_at_end=True,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    report_to="none"
)


# ============================================================
# 6. Trainer
# ============================================================

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# ============================================================
# 7. Train & Evaluate
# ============================================================

trainer.train()

print("\n================ Final Test ================")
print(trainer.evaluate(test_ds))

# Save metrics
import os
OUTDIR = "results/improved_roberta_merge"
os.makedirs(OUTDIR, exist_ok=True)

metrics = trainer.evaluate(test_ds)
with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved:", os.path.join(OUTDIR, "metrics.json"))



