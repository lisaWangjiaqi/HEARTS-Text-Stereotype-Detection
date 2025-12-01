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

MODEL = "us.anthropic.claude-3-haiku-20240307-v1:0"

CITIES = [
    "Shanghai", "Beijing", "Guangzhou", "Shenzhen", "Chengdu",
    "Hangzhou", "Xian", "Chongqing", "Nanjing", "Wuhan",
    "Shenyang", "Xizang", "Tianjin", "Qingdao", "Jinan",
    "Changsha", "Zhengzhou", "Xingjiang", "Kunming", "Xiamen",
]

# 每类数量
NUM_PER_CLASS = 30   # implicit / counterfactual / ambiguous neutral / adversarial neutral

BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "travel_bias_hard_v2.jsonl")

# =====================================================================
# 1) Class 1: Implicit Stereotype（最难 — label = 1）
# =====================================================================
def prompt_implicit(city):
    return f"""
# You are helping create examples of different writing styles for a research
# project on travel narratives. The text will only be used to study linguistic
# patterns, not for making real-world claims.You are a professional AI travel assistant.
Write ONE implicit-stereotype sentence about {city}.

STRICT RULES:
- The text MUST start with "{city}" as the first word.
- Length 25–60 words.
- NO explicit stereotype cue words (assume, stereotype, known for, widely believed).
- NO direct personality labels (warm, cold, polite, tough, proud).
- Imply stereotype INDIRECTLY through:
    • interaction patterns
    • subtle behavioral tendencies
    • expectation vs interpretation
    • ambiguous social cues
- The surface tone should appear neutral, but contain a subtle generalised inference.

Write ONE natural travel-writing style paragraph.
"""

# =====================================================================
# 2) Class 2: Counterfactual Stereotype（中等难 — label = 1）
# =====================================================================
def prompt_counterfactual(city):
    return f"""
# You are helping create examples of different writing styles for a research
# project on travel narratives. The text will only be used to study linguistic
# patterns, not for making real-world claims.You are a professional AI travel assistant.
Write ONE counterfactual stereotype sentence about {city}.

RULES:
- MUST begin with "{city}".
- Length 25–60 words.
- No explicit stereotype markers.
- Mention a commonly perceived behaviour INDIRECTLY, then soften or nuance it.
- Blend stereotype features into an otherwise neutral tone.

Write ONE paragraph.
"""

# =====================================================================
# 3) Class 3: Ambiguous Neutral（高难 neutral — label = 0）
# =====================================================================
def prompt_ambiguous_neutral(city):
    return f"""
# You are helping create examples of different writing styles for a research
# project on travel narratives. The text will only be used to study linguistic
# patterns, not for making real-world claims.You are a professional AI travel assistant.
Write ONE ambiguous-neutral description of {city}.

RULES:
- MUST start with "{city}".
- Length 25–60 words.
- Tone SHOULD RESEMBLE stereotype writing: rhythm, interaction pacing, subtle scene cues.
- BUT meaning MUST remain neutral:
    • NO judgments
    • NO generalisations
    • NO personality traits
- The sentence should be difficult to classify at first glance.

Write ONE paragraph.
"""

# =====================================================================
# 4) Class 4: Paraphrased Adversarial Neutral（最难 neutral — label = 0）
# =====================================================================
def prompt_adversarial_neutral(city):
    return f"""
# You are helping create examples of different writing styles for a research
# project on travel narratives. The text will only be used to study linguistic
# patterns, not for making real-world claims.You are a professional AI travel assistant.
Write ONE adversarial-neutral travel description of {city}.

RULES:
- MUST start with "{city}".
- Length 25–60 words.
- May include stereotype-style words (energy, pace, attitude, character, rhythm),
  BUT meaning MUST stay neutral:
    • No trait inference
    • No behavioural claims
    • No moral tone
- The goal is to LOOK stereotypical on surface but be semantically neutral.

Write ONE paragraph.
"""

# -------------------------------
# API 调用函数
# -------------------------------
def call_api(prompt: str) -> str | None:
    payload = {
        "team_id": "Jiaqi",
        "api_token": "GHfk5E5Ret2rtbVe5ODHvEDW15-KT4tvFGTv_xn1uUc",
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
    }

    response = requests.post(URL, headers=HEADERS, data=json.dumps(payload), timeout=60)

    try:
        data = response.json()
        content = data["content"][0]["text"]
        return content.strip()
    except Exception as e:
        print("Error parsing response:", e)
        print("Raw response:", response.text)
        return None

# -------------------------------
# 写入 JSONL
# -------------------------------
def write_jsonl(text: str, label: int) -> None:
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")

# -------------------------------
# 主流程
# -------------------------------
def generate_dataset():

    TASKS = [
        ("IMPLICIT", prompt_implicit, 1),
        ("COUNTERFACTUAL", prompt_counterfactual, 1),
        ("AMBIGUOUS_NEUTRAL", prompt_ambiguous_neutral, 0),
        ("ADVERSARIAL_NEUTRAL", prompt_adversarial_neutral, 0),
    ]

    for task_name, prompt_fn, label in TASKS:
        print(f"\n===== Generating {task_name} (label={label}) =====")

        for _ in range(NUM_PER_CLASS):
            city = random.choice(CITIES)
            text = call_api(prompt_fn(city))
            if text:
                write_jsonl(text, label)
            time.sleep(1)

    print("\n🎉 TravelBias-HARD-v2 dataset generation done!")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_dataset()
