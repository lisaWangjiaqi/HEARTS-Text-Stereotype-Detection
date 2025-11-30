import json
import random
import os

BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias")
INPUT_FILE = os.path.join(DATA_DIR, "travel_bias_dataset.jsonl")

# 输出文件
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
DEV_FILE = os.path.join(DATA_DIR, "dev.jsonl")
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl")

# 读取数据
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f]

# 打乱
random.shuffle(lines)

# Split (70/15/15)
n = len(lines)
train, dev, test = lines[:70], lines[70:85], lines[85:]

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

write_jsonl(TRAIN_FILE, train)
write_jsonl(DEV_FILE, dev)
write_jsonl(TEST_FILE, test)

print("Dataset split completed:")
print(f"  Train: {len(train)} samples → {TRAIN_FILE}")
print(f"  Dev:   {len(dev)} samples → {DEV_FILE}")
print(f"  Test:  {len(test)} samples → {TEST_FILE}")

