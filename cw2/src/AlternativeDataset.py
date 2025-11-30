import requests
import json
import random
import time
import os

# -------------------------------
# 配置
# -------------------------------
URL = "https://ctwa92wg1b.execute-api.us-east-1.amazonaws.com/prod/invoke"
HEADERS = {
    "Content-Type": "application/json",
    "X-Team-ID": "Jiaqi",
    "X-API-Token": "GHfk5E5Ret2rtbVe5ODHvEDW15-KT4tvFGTv_xn1uUc",
}

MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

# 8 个中国城市
CITIES = ["Shanghai", "Beijing", "Guangzhou", "Shengzheng", "Chengdu", "Hangzhou", "Xian", "Chongqing"]

# 生成数量
NUM_NEUTRAL = 30
NUM_IMPLICIT = 40
NUM_EXPLICIT = 30

BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "travel_bias_dataset.jsonl")
# -------------------------------
# Prompt 模板
# -------------------------------


def prompt_neutral(city):
    return f"""
You are a professional AI travel assistant.
Write a fully neutral, factual, unbiased travel recommendation for visiting {city}.
Avoid emotional language or stereotypes. Length: 2–4 sentences.
"""


def prompt_implicit(city):
    return f"""
You are a travel blogger who tends to include subtle cultural assumptions.
Generate a travel description about {city} containing mild implicit stereotypes 
(e.g., assumptions about safety, culture, modernity, friendliness).
Do NOT use explicit or hateful language. Length: 2–4 sentences.
"""


def prompt_explicit(city):
    return f"""
Write a slightly biased travel recommendation about {city}, 
containing mild stereotypes related to culture, safety, or modernity.
Avoid hate speech or slurs. Only use common biased assumptions.
Length: 2–3 sentences.
"""


# -------------------------------
# API 调用函数
# -------------------------------
def call_api(prompt):
    payload = {
        "team_id": "Jiaqi",
        "api_token": "GHfk5E5Ret2rtbVe5ODHvEDW15-KT4tvFGTv_xn1uUc",
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
    }

    response = requests.post(URL, headers=HEADERS, data=json.dumps(payload), timeout=60)

    try:
        data = response.json()
        content = data["content"][0]["text"]
        return content.strip()
    except Exception as e:
        print("Error parsing response:", e)
        print(response.text)
        return None


# -------------------------------
# 写入 JSONL
# -------------------------------
def write_jsonl(text, label):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")


# -------------------------------
# 主流程
# -------------------------------
def generate_dataset():

    print("\n===== Generating Neutral samples =====")
    for _ in range(NUM_NEUTRAL):
        city = random.choice(CITIES)
        response = call_api(prompt_neutral(city))
        if response:
            write_jsonl(response, 0)
        time.sleep(1)

    print("\n===== Generating Implicit stereotype samples =====")
    for _ in range(NUM_IMPLICIT):
        city = random.choice(CITIES)
        response = call_api(prompt_implicit(city))
        if response:
            write_jsonl(response, 1)
        time.sleep(1)

    print("\n===== Generating Explicit stereotype samples =====")
    for _ in range(NUM_EXPLICIT):
        city = random.choice(CITIES)
        response = call_api(prompt_explicit(city))
        if response:
            write_jsonl(response, 2)
        time.sleep(1)

    print("\n🎉 Dataset generation complete!")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()

