

import json
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer
)

# ============================================================
# 1. Load TravelBias Dataset
# ============================================================

DATA_PATH = "data_travel_bias/travel_bias_hard_v1.jsonl"

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

MODEL_NAME = "roberta-large"   # 🚀 比 EMGSD stronger

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
# 3. Improved Model
#    (RoBERTa encoder + SoftPrompt + Focal Loss + LLRD)
# ============================================================

class ImprovedModel(nn.Module):


    def __init__(self, model_name="roberta-large", num_labels=2):
        super().__init__()
        self.num_labels = num_labels

        # Base encoder
        self.encoder = AutoModel.from_pretrained(model_name)

        hidden = self.encoder.config.hidden_size

        # ------ SoftPrompt (10 continuous prompt tokens) ------
        self.prompt_len = 10
        self.prompt = nn.Parameter(
            torch.randn(self.prompt_len, hidden) * 0.01
        )

        # ------ Classifier ------
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_labels)
        )

        # ------ Focal Loss ------
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)

    def forward(self, input_ids, attention_mask, labels=None):
  

        batch_size = input_ids.size(0)
        device = input_ids.device

        # --------------------------------------------------------
        # 1) Get original embeddings: """ (batch, seq_len, hidden) """
        # --------------------------------------------------------
        inputs_embeds = self.encoder.embeddings(input_ids)

        # --------------------------------------------------------
        # 2) Add soft prompt embeddings at the front
        #    prompt: """ (prompt_len, hidden) """
        #    expanded_prompt: """ (batch, prompt_len, hidden) """
        # --------------------------------------------------------
        expanded_prompt = self.prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # concat prompt + original_embed
        concat_embeds = torch.cat([expanded_prompt, inputs_embeds], dim=1)
        # shape: """ (batch, prompt_len + seq_len, hidden) """

        # attention mask padding
        prompt_mask = torch.ones(batch_size, self.prompt_len).to(device)
        concat_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        # shape: """ (batch, prompt_len + seq_len) """

        # --------------------------------------------------------
        # 3) Encode with prompt
        # --------------------------------------------------------
        outputs = self.encoder(
            inputs_embeds=concat_embeds,
            attention_mask=concat_mask
        )

        # last_hidden_state: """ (batch, total_len, hidden) """
        cls_vec = outputs.last_hidden_state[:, 0]  # use first token

        logits = self.classifier(cls_vec)  # """ (batch, num_labels) """

        # --------------------------------------------------------
        # 4) Loss or Inference
        # --------------------------------------------------------
        if labels is not None:
            loss = self.focal(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}


# ============================================================
# 4. Focal Loss
# ============================================================

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, labels):
        ce_loss = self.ce(logits, labels)       # """ (batch,) """
        pt = torch.exp(-ce_loss)                # """ (batch,) """
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


# ============================================================
# 5. Training Pipeline
# ============================================================

model = ImprovedModel(MODEL_NAME, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }

args = TrainingArguments(
    output_dir="results/improved_roberta",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_safetensors=True,
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_steps=20,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

print("\n================ Final Test ================")
print(trainer.evaluate(test_ds))

# ===== Save metrics to file =====
import json, os

metrics = trainer.evaluate(test_ds)
print(metrics)

OUTDIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src/results/improved_roberta"
os.makedirs(OUTDIR, exist_ok=True)

with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved to:", os.path.join(OUTDIR, "metrics.json"))



# import os
# import json
# import random

# import numpy as np
# import torch
# from datasets import Dataset
# from transformers import (
#     BertTokenizerFast,
#     BertForSequenceClassification,
#     TrainingArguments,
#     Trainer
# )
# from sklearn.metrics import accuracy_score, f1_score

# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)


# # ============================================================
# # 1) Load TravelBias-HARD dataset
# # ============================================================

# DATA_PATH = "data_travel_bias/travel_bias_hard_v2.jsonl"

# data_list = []
# with open(DATA_PATH, "r") as f:
#     for line in f:
#         if line.strip():
#             data_list.append(json.loads(line))

# dataset = Dataset.from_list(data_list)

# # Train/Val/Test split
# dataset = dataset.train_test_split(test_size=0.2, seed=SEED)
# train_val = dataset["train"].train_test_split(test_size=0.2, seed=SEED)

# train_ds = train_val["train"]
# val_ds   = train_val["test"]
# test_ds  = dataset["test"]

# print("Train:", len(train_ds), "Val:", len(val_ds), "Test:", len(test_ds))


# # ============================================================
# # 2) Model & Tokenizer
# # ============================================================

# MODEL_NAME = "bert-base-uncased"

# tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

# def preprocess(batch):
#     enc = tokenizer(
#         batch["text"],
#         truncation=True,
#         padding="max_length",
#         max_length=128
#     )
#     enc["labels"] = batch["label"]
#     return enc

# train_ds = train_ds.map(preprocess)
# val_ds   = val_ds.map(preprocess)
# test_ds  = test_ds.map(preprocess)

# train_ds.set_format("torch")
# val_ds.set_format("torch")
# test_ds.set_format("torch")

# model = BertForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=2
# )


# # ============================================================
# # 3) Metrics
# # ============================================================

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=-1)
#     return {
#         "accuracy": accuracy_score(labels, preds),
#         "macro_f1": f1_score(labels, preds, average="macro")
#     }


# # ============================================================
# # 4) Training
# # ============================================================

# OUT_DIR = "results/travel_bert_baseline"

# args = TrainingArguments(
#     output_dir=OUT_DIR,
#     num_train_epochs=5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     learning_rate=3e-5,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_macro_f1",
#     seed=SEED,
#     logging_steps=10
# )

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     compute_metrics=compute_metrics,
#     tokenizer=tokenizer,
# )

# trainer.train()


# # ============================================================
# # 5) Final Evaluation
# # ============================================================

# print("\n===== Final Test Evaluation (BERT Baseline) =====")
# results = trainer.evaluate(test_ds)
# print(results)

# # Save metrics
# with open(os.path.join(OUT_DIR, "test_metrics.json"), "w") as f:
#     json.dump(results, f, indent=4)

# print("Metrics saved to:", OUT_DIR)
