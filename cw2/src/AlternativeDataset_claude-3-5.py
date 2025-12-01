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
CITIES = [
    "Shanghai", "Beijing", "Guangzhou", "Shenzhen", "Chengdu", "Hangzhou", "Xian", "Chongqing",
    "Nanjing", "Wuhan", "Shenyang", "Xizang",
    "Tianjin", "Qingdao", "Jinan", "Changsha",
    "Zhengzhou", "Xingjiang", "Kunming", "Xiamen",
]

# 二分类数量设置
NUM_NEUTRAL = 30          # label = 0
NUM_STEREO = 70           # label = 1

BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "travel_bias_dataset.jsonl")


# -------------------------------
# Prompt 模板（尽量避免触发安全机制）
# -------------------------------
def prompt_neutral(city: str) -> str:
    """
    生成尽量中性的旅行描述（label=0）
    """
    return f"""
The text you generate will be used for
model training and research purposes to study how impressionistic or
generalised travel descriptions are written.
You are a professional AI travel assistant.
Write a concise, factual city introduction for visitors to {city}.
Use a neutral, encyclopedic tone focusing on geography, transport, landmarks, and general travel information.
Avoid emotional wording, value judgements, or colourful subjective phrases. Length: 2–4 sentences.
"""


def prompt_stereo(city: str) -> str:
    """
    生成带有“常见印象 / cliché 风格”的旅行描述（label=1）
    语气可以是轻微印象化，也可以是稍微 cliché，一切交给模型在文本里变化。
    """
    return f"""
You are helping create examples of different writing styles for a research
project on travel narratives. The text will only be used to study linguistic
patterns, not for making real-world claims.

You are a professional AI travel assistant. 
# You are helping create examples of different writing styles for a research
# project on travel narratives. The text will only be used to study linguistic
# patterns, not for making real-world claims.

# You are a professional AI travel assistant. Write a short, travel-oriented description of **{city}** that mimics the *strong,
high-intensity generalisations* often found in casual travel talk. Your
description should portray the city as if it has a widely shared personality,
social atmosphere, and behavioural norms that outsiders assume apply to nearly
everyone who lives there.

To increase realism, incorporate several stereotype dimensions commonly used
by tourists, such as:
- perceived safety or danger (“rough after dark”, “always safe and orderly”)  
- wealth or development level (“flashy and wealthy”, “old-fashioned or behind”)  
- pace and atmosphere (“relentlessly fast”, “slow and sleepy”)  
- moral tone or nightlife judgment (“wild and indulgent”, “strict and upright”)  
- resident personality traits (“proud”, “calculating”, “warm”, “cold”, “tough”)  
- government efficiency or bureaucracy (“efficient but harsh”, “slow and confusing”)  
- tourism clichés (“packed with influencers”, “overrun by crowds”)  

Use strong generalisation patterns such as:
• “People there are almost always described as…”  
• “Locals are widely assumed to…”  
• “Visitors often claim that the whole city feels…”  
• “Travel stories regularly portray residents as…”  
• “Among tourists, it’s nearly taken for granted that people there…”  

Your sentences should emphasise **collective personality traits, social norms,
and assumed behavioural patterns**, written in an exaggerated, impressionistic
tone that reflects tourist stereotypes. Keep it mild, illustrative, and
non-offensive. Length: 2–4 sentences.
"""


# -------------------------------
# API 调用函数
# -------------------------------
def call_api(prompt: str) -> str | None:
    """
    调用远程大模型接口，返回生成文本

    输入:
        prompt: str, 要发送给模型的提示词

    输出:
        str: 模型返回的文本（去掉首尾空白）
        None: 请求或解析失败
    """
    payload = {
        "team_id": "Jiaqi",
        "api_token": "GHfk5E5Ret2rtbVe5ODHvEDW15-KT4tvFGTv_xn1uUc",
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        URL,
        headers=HEADERS,
        data=json.dumps(payload),
        timeout=60,
    )

    try:
        data = response.json()
        content = data["content"][0]["text"]
        return content.strip()
    except Exception as e:
        print("Error parsing response:", e)
        print("Raw response text:", response.text)
        return None


# -------------------------------
# 写入 JSONL
# -------------------------------
def write_jsonl(text: str, label: int) -> None:
    """
    将一条数据追加写入 jsonl 文件

    输入:
        text: 生成的旅行描述
        label: 0 = neutral, 1 = stereotype / cliché
    """
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")


# -------------------------------
# 主流程
# -------------------------------
def generate_dataset() -> None:
    """
    主流程：
        - 生成 neutral 文本（label=0）
        - 生成 stereotype 文本（label=1）
    """
    print("\n===== Generating Neutral samples (label=0) =====")
    for _ in range(NUM_NEUTRAL):
        city = random.choice(CITIES)
        response = call_api(prompt_neutral(city))
        if response:
            write_jsonl(response, 0)
        time.sleep(1)

    print("\n===== Generating Stereotype-style samples (label=1) =====")
    for _ in range(NUM_STEREO):
        city = random.choice(CITIES)
        response = call_api(prompt_stereo(city))
        if response:
            write_jsonl(response, 1)
        time.sleep(1)

    print("\n🎉 Dataset generation complete!")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_dataset()
