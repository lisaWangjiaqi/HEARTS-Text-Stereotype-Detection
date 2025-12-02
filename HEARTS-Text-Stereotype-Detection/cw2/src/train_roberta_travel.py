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

DATA_PATH = "data_travel_bias/travel_bias_hard_v2.jsonl"

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
    output_dir="results/improved_roberta",
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
OUTDIR = "results/improved_roberta"
os.makedirs(OUTDIR, exist_ok=True)

metrics = trainer.evaluate(test_ds)
with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved:", os.path.join(OUTDIR, "metrics.json"))



# import json
# import numpy as np
# from datasets import Dataset
# from sklearn.metrics import accuracy_score, f1_score

# import torch
# import torch.nn as nn
# from transformers import (
#     AutoTokenizer,
#     AutoModel,
#     TrainingArguments,
#     Trainer
# )

# # ============================================================
# # 1. Load TravelBias Dataset
# # ============================================================

# DATA_PATH = "data_travel_bias/travel_bias_hard_v2.jsonl"
# # DATA_PATH = "data_travel_bias/travel_bias_clean_aug.jsonl"

# data_list = []
# with open(DATA_PATH, "r") as f:
#     for line in f:
#         if line.strip():
#             data_list.append(json.loads(line))

# dataset = Dataset.from_list(data_list)

# dataset = dataset.train_test_split(test_size=0.2, seed=42)
# train_val = dataset["train"].train_test_split(test_size=0.2, seed=42)

# train_ds = train_val["train"]
# val_ds   = train_val["test"]
# test_ds  = dataset["test"]


# # ============================================================
# # 2. Tokenizer
# # ============================================================

# MODEL_NAME = "roberta-large"
# # MODEL_NAME = "google/electra-large-discriminator"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# def preprocess(batch):
#     enc = tokenizer(
#         batch["text"],
#         truncation=True,
#         padding="max_length",
#         max_length=128
#     )
#     enc["labels"] = batch["label"]
#     return enc

# train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
# val_ds   = val_ds.map(preprocess, remove_columns=val_ds.column_names)
# test_ds  = test_ds.map(preprocess, remove_columns=test_ds.column_names)

# train_ds.set_format("torch")
# val_ds.set_format("torch")
# test_ds.set_format("torch")


# # ============================================================
# # 3. Improved Model (SoftPrompt + FocalLoss)
# # ============================================================

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=0.5):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.ce = nn.CrossEntropyLoss(reduction="none")

#     def forward(self, logits, labels):
#         ce_loss = self.ce(logits, labels)       # """ (batch,) """
#         pt = torch.exp(-ce_loss)                # """ (batch,) """
#         focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
#         return focal.mean()


# class ImprovedModel(nn.Module):

#     def __init__(self, model_name="roberta-large", num_labels=2):
#         super().__init__()

#         self.encoder = AutoModel.from_pretrained(model_name)
#         hidden = self.encoder.config.hidden_size

#         # SoftPrompt: reduced to 4 tokens
#         self.prompt_len = 4
#         self.prompt = nn.Parameter(torch.zeros(self.prompt_len, hidden))

#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden, num_labels)
#         )

#         self.focal = FocalLoss(alpha=0.25, gamma=0.5)

#     def forward(self, input_ids, attention_mask, labels=None):

#         batch_size = input_ids.size(0)
#         device = input_ids.device

#         # embeddings """ (batch, seq_len, hidden) """
#         inputs_embeds = self.encoder.embeddings(input_ids)

#         # expand prompt """ (batch, prompt_len, hidden) """
#         expanded_prompt = self.prompt.unsqueeze(0).expand(batch_size, -1, -1)

#         # concat
#         concat_embeds = torch.cat([expanded_prompt, inputs_embeds], dim=1)
#         prompt_mask = torch.ones(batch_size, self.prompt_len).to(device)
#         concat_mask = torch.cat([prompt_mask, attention_mask], dim=1)

#         outputs = self.encoder(
#             inputs_embeds=concat_embeds,
#             attention_mask=concat_mask
#         )

#         cls_vec = outputs.last_hidden_state[:, 0]  # """ (batch, hidden) """
#         logits = self.classifier(cls_vec)

#         if labels is not None:
#             loss = self.focal(logits, labels)
#             return {"loss": loss, "logits": logits}
#         else:
#             return {"logits": logits}


# model = ImprovedModel(MODEL_NAME, num_labels=2)
# #model.encoder.config.save_pretrained("results/improved_roberta")    


# # ============================================================
# # 4. Metrics
# # ============================================================

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=-1)
#     return {
#         "accuracy": accuracy_score(labels, preds),
#         "macro_f1": f1_score(labels, preds, average="macro")
#     }


# # ============================================================
# # 5. LLRD Optimizer (Layer-wise Learning Rate Decay)
# # ============================================================

# def get_optimizer_with_llrd(model, base_lr=2e-5, weight_decay=0.01, decay_rate=0.95):

#     optimizer_grouped_parameters = []

#     # Roberta-large has 24 layers
#     layers = model.encoder.encoder.layer

#     for i, layer in enumerate(layers):
#         lr = base_lr * (decay_rate ** (23 - i))
#         optimizer_grouped_parameters.append({
#             "params": layer.parameters(),
#             "lr": lr,
#             "weight_decay": weight_decay
#         })

#     # Embeddings - lowest LR
#     optimizer_grouped_parameters.append({
#         "params": model.encoder.embeddings.parameters(),
#         "lr": base_lr * (decay_rate ** 24),
#         "weight_decay": weight_decay
#     })

#     # Soft prompt + classifier: highest LR
#     optimizer_grouped_parameters.append({
#         "params": model.prompt,
#         "lr": base_lr,
#         "weight_decay": 0.0
#     })
#     optimizer_grouped_parameters.append({
#         "params": model.classifier.parameters(),
#         "lr": base_lr,
#         "weight_decay": weight_decay
#     })

#     optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=base_lr)
#     return optimizer


# optimizer = get_optimizer_with_llrd(model)


# # ============================================================
# # 6. TrainingArguments + Trainer
# # ============================================================

# args = TrainingArguments(
#     output_dir="results/improved_roberta",
#     num_train_epochs=4,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     save_safetensors=True,
#     save_total_limit=1,
#     load_best_model_at_end=True,
#     logging_steps=20,
#     learning_rate=2e-5,
#     warmup_ratio=0.1,
#     report_to="none"
# )

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
#     #optimizers=(optimizer, None)   # ★ Inject LLRD optimizer
# )


# # ============================================================
# # 7. Train & Evaluate
# # ============================================================

# trainer.train()

# print("\n================ Final Test ================")
# print(trainer.evaluate(test_ds))

# import os
# OUTDIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src/results/improved_roberta"
# os.makedirs(OUTDIR, exist_ok=True)

# metrics = trainer.evaluate(test_ds)
# with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
#     json.dump(metrics, f, indent=4)

# print("Metrics saved:", os.path.join(OUTDIR, "metrics.json"))



# #gai-----
# # =======================
# # Save final model
# # =======================
# # FINAL_DIR = "results/improved_roberta/final_model"

# # model_to_save = trainer.model
# # model_to_save.save_pretrained(FINAL_DIR)
# # tokenizer.save_pretrained(FINAL_DIR)

# # print("Final model saved to:", FINAL_DIR)

