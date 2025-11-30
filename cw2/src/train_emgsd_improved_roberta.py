# 文件名：train_emgsd_roberta.py

import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================================
# 1. Focal Loss 定义
# ==========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        if self.alpha is not None:
            self.alpha = self.alpha.to(logits.device)

        ce_loss = F.cross_entropy(
            logits,
            labels,
            weight=self.alpha,
            reduction="none"
        )

        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


# ==========================================================
# 2. 加载数据集 + label 映射
# ==========================================================
print("Loading EMGSD dataset...")
ds = load_dataset("holistic-ai/EMGSD")

label2id = {
    "neutral_gender": 0,
    "neutral_lgbtq+": 1,
    "neutral_nationality": 2,
    "neutral_profession": 3,
    "neutral_race": 4,
    "neutral_religion": 5,
    "stereotype_gender": 6,
    "stereotype_lgbtq+": 7,
    "stereotype_nationality": 8,
    "stereotype_profession": 9,
    "stereotype_race": 10,
    "stereotype_religion": 11,
    "unrelated": 12
}

ds = ds.map(lambda x: {"label": label2id[x["label"]]})
num_classes = len(label2id)
print(f"\nNumber of classes = {num_classes}")


# ==========================================================
# 3. 计算 class weights（平衡类别）
# ==========================================================
def compute_class_weights(labels):
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float)

train_labels = np.array(ds["train"]["label"])
class_weights = compute_class_weights(train_labels)

print("\n[Class Weights]")
for i, w in enumerate(class_weights):
    print(f" class {i}: weight={w.item():.4f}")


# ==========================================================
# 4. Tokenizer + preprocess
# ==========================================================
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": batch["label"]
    }

print("\nTokenizing dataset...")
ds = ds.map(preprocess, batched=True)
ds = ds.remove_columns(["text", "category", "text_with_marker", "data_source"])


# ==========================================================
# 5. 加载模型（roberta-base）
# ==========================================================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
)


# ==========================================================
# 6. Trainer + FocalLoss
# ==========================================================
class CustomTrainer(Trainer):
    def __init__(self, focal_alpha, focal_gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.criterion(logits, labels)

        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir="./results/emgsd_roberta",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50,
    metric_for_best_model="eval_macro_f1"
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1
    }


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    focal_alpha=class_weights,
    focal_gamma=2.0
)


# ==========================================================
# 7. 开始训练
# ==========================================================
trainer.train()


# ==========================================================
# 8. 保存最佳模型
# ==========================================================
save_path = "./results/emgsd_roberta/final_model"
trainer.save_model(save_path)
print(f"\nModel saved → {save_path}")

