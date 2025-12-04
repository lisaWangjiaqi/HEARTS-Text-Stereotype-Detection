
#1.download--------------
# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="holistic-ai/EMGSD",
#     repo_type="dataset",
#     local_dir="./Travelbias_dataset/EMGSD_raw",
#     local_dir_use_symlinks=False
# )


#------------------------------------------
#查看分布
# import pandas as pd

# df = pd.read_csv("Travelbias_dataset/EMGSD_raw/train.csv")

# print(df['label'].value_counts())

#------------------------------------------


#2. keep stereotype / unrelated
# import pandas as pd
# from datasets import Dataset

# RAW = "Travelbias_dataset/EMGSD_raw/train.csv"

# # 用 pandas 读取
# df = pd.read_csv(RAW)

# print("Original rows:", len(df))
# print(df.head())


# # ===============================
# # 规则：
# # stereotype_* → label = 1
# # unrelated & neutral_* → label = 0
# # ===============================

# def convert_label(label):
#     if str(label).startswith("stereotype"):
#         return 1
#     else:
#         return 0

# # 创建新标签列
# df["label_binary"] = df["label"].apply(convert_label)

# print(df["label_binary"].value_counts())


# # 只保留 text + label_binary
# df = df[["text", "label_binary"]]
# df = df.rename(columns={"label_binary": "label"})

# print("Cleaned rows:", len(df))
# print(df.head())


# # 转成 HF Dataset
# ds = Dataset.from_pandas(df)

# # 保存 JSONL
# OUT = "Travelbias_dataset/EMGSD_binary.jsonl"
# ds.to_json(OUT, lines=True, force_ascii=False)

# print("Saved:", OUT)
# print(ds[:5])


#------------------------------------------
# #3. save 200 data sample
# import json
# import random

# INPUT = "Travelbias_dataset/EMGSD_binary.jsonl"
# OUTPUT = "Travelbias_dataset/EMGSD_binary_clean_200.jsonl"

# # 读取数据
# label0 = []
# label1 = []

# with open(INPUT, "r") as fin:
#     for line in fin:
#         if not line.strip():
#             continue
#         obj = json.loads(line)

#         if obj["label"] == 0:
#             label0.append(obj)
#         elif obj["label"] == 1:
#             label1.append(obj)

# print("Label 0 count:", len(label0))
# print("Label 1 count:", len(label1))


# # ------ 抽样目标 ------
# sample_n = 100   # each class sample size


# # 如果某类不足 100，则全部取完
# sample0 = label0 if len(label0) <= sample_n else random.sample(label0, sample_n)
# sample1 = label1 if len(label1) <= sample_n else random.sample(label1, sample_n)


# # 合并并 shuffle
# sampled = sample0 + sample1
# random.shuffle(sampled)

# print("Total sampled:", len(sampled))


# # 写入输出文件
# with open(OUTPUT, "w") as fout:
#     for item in sampled:
#         fout.write(json.dumps(item, ensure_ascii=False) + "\n")

# print("Done. Saved balanced 200 samples to:", OUTPUT)

#------------------------------------------
#4. merge
import json
import random

EMGSD = "Travelbias_dataset/EMGSD_binary_clean_200.jsonl"
TRAVEL = "Travelbias_dataset/travel_bias_hard_v2.jsonl"
OUTPUT = "Travelbias_dataset/merged_dataset.jsonl"

all_items = []

# ---- 读取 EMGSD ----
with open(EMGSD, "r") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            all_items.append(obj)

# ---- 读取 travel_bias 数据集 ----
with open(TRAVEL, "r") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            all_items.append(obj)

print("Total merged samples:", len(all_items))

# ---- shuffle 打乱顺序 ----
random.shuffle(all_items)

# ---- 输出 ----
with open(OUTPUT, "w") as fout:
    for item in all_items:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done. Merged dataset saved to:", OUTPUT)

