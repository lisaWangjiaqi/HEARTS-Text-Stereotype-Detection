import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# 1. Load JSONL dataset
# ============================================================

jsonl_path = "data_travel_bias/travel_bias_dataset.jsonl"

print("Loading travel bias dataset...")

data_list = []
with open(jsonl_path, "r") as f:
    for line in f:
        if line.strip():
            data_list.append(json.loads(line))

# Convert to HF Dataset
dataset = Dataset.from_list(data_list)

print(dataset[0])
print(f"Total samples: {len(dataset)}")

# ============================================================
# 2. Split dataset（train/test = 80/20）
# ============================================================

train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = train_test["train"]
test_ds = train_test["test"]

# ============================================================
# 3. Tokenizer
# ============================================================

model_name = "albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    enc = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    enc["labels"] = examples["label"]
    return enc

train_ds = train_ds.map(preprocess)
test_ds = test_ds.map(preprocess)

# ============================================================
# 4. Load baseline model
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3  # 无偏、隐性偏见、显性偏见
)

# ============================================================
# 5. Trainer
# ============================================================

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

args = TrainingArguments(
    output_dir="results/travel_baseline",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ============================================================
# 6. Train
# ============================================================

trainer.train()

trainer.save_model("results/travel_baseline/final_model")
print("\nTraining complete!")
