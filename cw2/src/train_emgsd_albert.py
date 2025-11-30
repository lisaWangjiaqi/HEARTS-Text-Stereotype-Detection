# 文件名：cw2/src/train_emgsd_albert.py

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import os
os.environ["CODECARBON_LOGGER_ENABLED"] = "false"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training on EMGSD."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="albert-base-v2")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    return parser.parse_args()


def compute_metrics(eval_pred):
    """(logits: (N, C), labels: (N,)) -> dict"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load EMGSD dataset
    ds = load_dataset("holistic-ai/EMGSD")  # train / test

    # 2. Build label <-> id mapping from the whole dataset
    label_list = sorted(list(set(ds["train"]["label"])))
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}

    def encode_labels(example: dict) -> dict:
        """{'label': str} -> {'label': int}"""
        example["label"] = label2id[example["label"]]
        return example

    ds = ds.map(encode_labels)

    # 3. Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(example: dict) -> dict:
        """{'text': str} -> token ids etc."""
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    ds = ds.map(tokenize_fn, batched=True)

    # HF Trainer 需要这些列名
    ds = ds.rename_column("label", "labels")
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    # 4. TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
    )

    # 6. Train & evaluate
    trainer.train()
    metrics = trainer.evaluate()

    # 7. Save metrics to JSON (供 notebook 读取)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(str(metrics))

    print("Final evaluation metrics:", metrics)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
