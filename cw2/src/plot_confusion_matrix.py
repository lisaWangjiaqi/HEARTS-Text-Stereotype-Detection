
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)


# 1. 
# DATA_PATH = "data_travel_bias/travel_bias_hard_v2.jsonl"
DATA_PATH = "Travelbias_dataset/merged_dataset.jsonl"
data_list = []
with open(DATA_PATH, "r") as f:
    for line in f:
        if line.strip():
            data_list.append(json.loads(line))

dataset = Dataset.from_list(data_list)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_val = dataset["train"].train_test_split(test_size=0.2, seed=42)

test_ds = dataset["test"]  


# 2. Tokenizer & preprocess

# MODEL_DIR = "results/improved_roberta/checkpoint-129"  
MODEL_DIR = "results/improved_roberta_merge/checkpoint-118" # merged
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

test_ds = test_ds.map(preprocess, remove_columns=test_ds.column_names)
test_ds.set_format("torch")


# 3.
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# 4.  Trainer

args = TrainingArguments(
    output_dir="tmp_confusion",
    per_device_eval_batch_size=8,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
)

# 5. Predict → preds / labels
pred_output = trainer.predict(test_ds)
logits = pred_output.predictions
labels = pred_output.label_ids
preds = np.argmax(logits, axis=1)

print("Labels shape:", labels.shape)
print("Preds shape:", preds.shape)


# 6. Plot Confusion Matrix

class_names = ["neutral", "biased"] 
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

plt.figure(figsize=(6, 6))
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - TravelBias Model")
plt.savefig("plot_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n[Saved] plot_confusion_matrix.png")

