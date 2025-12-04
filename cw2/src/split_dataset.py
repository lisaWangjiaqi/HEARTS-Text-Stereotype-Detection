import pandas as pd
import json
import random
import os

DATA_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src/Travelbias_dataset"
CSV = os.path.join(DATA_DIR, "travel_bias_for_annotation.csv")

TRAIN = os.path.join(DATA_DIR, "train.jsonl")
VAL = os.path.join(DATA_DIR, "val.jsonl")
TEST = os.path.join(DATA_DIR, "test.jsonl")


df = pd.read_csv(CSV)

df = df[df["human_label"].isin([0, 1])]

print("Loaded", len(df), "labeled samples")
print(df["human_label"].value_counts())

data = [
    {"text": row["text"], "label": int(row["human_label"])}
    for _, row in df.iterrows()
]

random.seed(42)
random.shuffle(data)

N = len(data)
n_train = int(0.7 * N)
n_val = int(0.15 * N)

train = data[:n_train]
val = data[n_train:n_train+n_val]
test = data[n_train+n_val:]

def write_jsonl(path, arr):
    with open(path, "w", encoding="utf-8") as f:
        for item in arr:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

write_jsonl(TRAIN, train)
write_jsonl(VAL, val)
write_jsonl(TEST, test)

print("\nDataset split completed:")
print(f"  Train: {len(train)} → {TRAIN}")
print(f"  Val:   {len(val)} → {VAL}")
print(f"  Test:  {len(test)} → {TEST}")
