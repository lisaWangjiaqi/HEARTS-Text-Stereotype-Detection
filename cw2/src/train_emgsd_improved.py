# 文件名：train_emgsd_improved.py

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
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --------------------------
# 1. Focal Loss 定义
# --------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        # 确保 alpha（class weights）在同一个 device
        if self.alpha is not None:
            self.alpha = self.alpha.to(logits.device)

        ce_loss = F.cross_entropy(
            logits,
            labels,
            weight=self.alpha,   # 必须和 logits 在同一个 device
            reduction="none"
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# --------------------------
# 2. 加载数据集 + label 映射
# --------------------------
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

# 替换 label 字段为数字
ds = ds.map(lambda x: {"label": label2id[x["label"]]})

num_classes = len(label2id)
print(f"Number of classes = {num_classes}")

# --------------------------
# 3. 计算 class weights
# --------------------------
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


# --------------------------
# 4. Tokenizer + preprocess
# --------------------------
model_name = "albert-base-v2"
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
        "token_type_ids": enc["token_type_ids"],
        "labels": batch["label"]
    }

print("Tokenizing dataset...")
ds = ds.map(preprocess, batched=True)

# 删除无用列
remove_cols = ["text", "category", "text_with_marker", "data_source"]
ds = ds.remove_columns(remove_cols)


# --------------------------
# 5. 加载模型（ALBERT）
# --------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
)


# --------------------------
# 6. Trainer + FocalLoss
# --------------------------
class CustomTrainer(Trainer):
    def __init__(self, focal_alpha, focal_gamma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")

        # 标准 forward
        outputs = model(**inputs)
        logits = outputs.logits

        # focal loss + class weights
        loss = self.criterion(logits, labels)

        if return_outputs:
            return loss, outputs
        else:
            return loss


training_args = TrainingArguments(
    output_dir="./results/emgsd_improved",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1"
)


from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    return {"accuracy": acc, "macro_f1": macro_f1}


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

# --------------------------
# 7. 开始训练
# --------------------------
trainer.train()

# --------------------------
# 8. 保存模型
# --------------------------
trainer.save_model("./results/emgsd_improved/final_model")
print("\nTraining complete!")
