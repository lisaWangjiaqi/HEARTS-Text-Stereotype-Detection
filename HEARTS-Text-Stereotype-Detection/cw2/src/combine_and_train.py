import os
import json
import pandas as pd
import random
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

BASE = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src/data_travel_bias"

AUTO_FILE = os.path.join(BASE, "travel_bias_dataset_clean.jsonl")
HUMAN_FILE = os.path.join(BASE, "travel_bias_for_annotation.csv")


# ============================================================
# 1. Load auto-generated data (LLM)
# ============================================================
auto_data = []
with open(AUTO_FILE, "r", encoding="utf-8") as f:
    for line in f:
        auto_data.append(json.loads(line))

print("Auto samples:", len(auto_data))


# ============================================================
# 2. Load human-labeled data (ground truth)
# ============================================================
df = pd.read_csv(HUMAN_FILE)
df = df[df["human_label"].isin([0, 1])]

human_data = [
    {"text": row["text"], "label": int(row["human_label"])}
    for _, row in df.iterrows()
]

print("Human samples:", len(human_data))


# ============================================================
# 3. 去重（分别对 auto 和 human 去重，不能混着去）
# ============================================================
def dedup_list(data):
    uniq = {}
    for item in data:
        uniq[item["text"]] = item
    return list(uniq.values())


auto_data = dedup_list(auto_data)
human_data = dedup_list(human_data)

print(f"After dedup: auto={len(auto_data)}, human={len(human_data)}")


# ============================================================
# 4. Train / Val = auto_data, Test = human_data
# ============================================================

# Shuffle auto_data
random.shuffle(auto_data)

N = len(auto_data)
train_end = int(N * 0.85)   # 85% train, 15% val

train = auto_data[:train_end]
val   = auto_data[train_end:]
test  = human_data          # 全量人工数据当测试集（重点）

print(f"Train={len(train)}, Val={len(val)}, Test={len(test)}")


# ============================================================
# 5. Build HF Dataset
# ============================================================
train_ds = Dataset.from_list(train)
val_ds   = Dataset.from_list(val)
test_ds  = Dataset.from_list(test)


# ============================================================
# 6. Tokenizer
# ============================================================
MODEL = "albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess(batch):
    e = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    e["labels"] = batch["label"]
    return e

train_ds = train_ds.map(preprocess, batched=True)
val_ds   = val_ds.map(preprocess, batched=True)
test_ds  = test_ds.map(preprocess, batched=True)


# ============================================================
# 7. Model
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=2
)


# ============================================================
# 8. Metrics
# ============================================================
def metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labs = p.label_ids
    return {
        "accuracy": accuracy_score(labs, preds),
        "macro_f1": f1_score(labs, preds, average="macro"),
    }


# ============================================================
# 9. Trainer
# ============================================================
OUT = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src/results/travel_albert_crosssource"

args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    learning_rate=3e-5,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=metrics,
)

trainer.train()


# ============================================================
# 10. Final evaluation on human-labeled TEST data
# ============================================================
print("\n===== TEST ON HUMAN DATA =====")
print(trainer.evaluate(test_ds))
