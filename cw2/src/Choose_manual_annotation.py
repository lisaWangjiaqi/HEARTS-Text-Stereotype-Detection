import json
import os
import csv

BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias") #data_travel_bias
SCORED_FILE = os.path.join(DATA_DIR, "travel_bias_scored.jsonl")

# print(">>> SCORED_FILE =", SCORED_FILE)


# 你想要人工标注的数量（建议 Top40 + Bottom40）
TOP_N = 40
BOTTOM_N = 40

# 输出的 CSV 路径
OUTPUT_CSV = os.path.join(DATA_DIR, "travel_bias_for_annotation.csv")

# 1. 读取 scored jsonl
samples = []
with open(SCORED_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "text" in obj and "stereo_score" in obj:
            samples.append(obj)

print(f"Loaded {len(samples)} scored samples")

if not samples:
    raise SystemExit("No samples found, check SCORED_FILE path or content.")

# 2. 按 emgsd_score 排序
samples.sort(key=lambda x: x["stereo_score"])

# 防止样本数不足
bottom_k = min(BOTTOM_N, len(samples))
top_k = min(TOP_N, len(samples))

bottom = samples[:bottom_k]
top = samples[-top_k:]

print(f"Select bottom {bottom_k} and top {top_k} samples for export.")

# 3. 示例检查
print("\nBottom example:")
for s in bottom[:5]:
    print(s["stereo_score"], s["text"][:80].replace("\n", " ") + "...")

print("\nTop example:")
for s in top[-5:]:
    print(s["stereo_score"], s["text"][:80].replace("\n", " ") + "...")

# 4. 合并写入 CSV
rows = []

for s in bottom:
    rows.append({
        "text": s["text"],
        "stereo_score": s["stereo_score"],
        "split": "bottom",
        "human_label": "",
        "notes": "",
    })

for s in top:
    rows.append({
        "text": s["text"],
        "stereo_score": s["stereo_score"],
        "split": "top",
        "human_label": "",
        "notes": "",
    })

fieldnames = ["text", "stereo_score", "split", "human_label", "notes"]

with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"\n✅ Exported {len(rows)} samples to: {OUTPUT_CSV}")
print("你可以在 Excel 里填写 human_label=0/1（0=中性，1=有刻板印象）")

