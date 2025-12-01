import os
import json
import random
import requests

# ============================================================
# API CONFIG
# ============================================================

URL = "https://ctwa92wg1b.execute-api.us-east-1.amazonaws.com/prod/invoke"
HEADERS = {
    "Content-Type": "application/json",
    "X-Team-ID": "Jiaqi",
    "X-API-Token": "GHfk5E5Ret2rtbVe5ODHvEDW15-KT4tvFGTv_xn1uUc",
}

MODEL = "us.anthropic.claude-3-haiku-20240307-v1:0"

def call_llm(prompt):
    payload = {
        "team_id": "Jiaqi",
        "api_token": "GHfk5E5Ret2rtbVe5ODHvEDW15-KT4tvFGTv_xn1uUc",
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 250
    }

    try:
        r = requests.post(URL, headers=HEADERS, data=json.dumps(payload), timeout=60)
        return r.json()["content"][0]["text"].strip()
    except:
        return None


# ============================================================
# CITIES (train OOD by using new cities)
# ============================================================

OOD_CITIES = [
    "Shanghai","Beijing","Guangzhou","Shenzhen","Chengdu",
    "Hangzhou","Xian","Chongqing","Nanjing","Wuhan"
]

BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "travel_bias_OOD.jsonl")


# ============================================================
# OOD TEMPLATE 1 — Human-style implicit stereotype
# ============================================================

IMPLICIT_OOD_HUMAN = """
Write a one-sentence IMPLICIT stereotype about {city} in a natural HUMAN writing style.

Rules:
- MUST start with '{city}'.
- Length 25–60 words.
- It must feel like written by a human (casual, varied rhythm, non-synthetic tone).
- It should subtly imply a behavioural tendency without naming traits.
- NO explicit markers like “stereotype”, “assume”, “known for”.
- Avoid robotic or formulaic patterns.
- Do NOT imitate AI writing.
"""


# ============================================================
# OOD TEMPLATE 2 — Realistic rewritten neutral
# ============================================================

NEUTRAL_OOD_REAL = """
Rewrite the following REAL TRAVEL DESCRIPTION into a one-sentence NEUTRAL description of {city}.

Base text:
"{base}"

Rules:
- MUST start with '{city}'.
- Length 25–60 words.
- Human-natural tone, descriptive but not synthetic.
- Must remain purely neutral (NO trait inference, NO bias).
"""


REAL_TRAVEL_SNIPPETS = [
    "The city blends historic districts with busy commercial areas, creating a layered urban landscape.",
    "Visitors often begin exploring from its central districts, where modern architecture mixes with traditional streets.",
    "The local transportation system makes it easy to reach cultural landmarks and riverside viewpoints.",
    "The city attracts travelers with its mixture of food markets, public squares, and waterfront scenery.",
]


# ============================================================
# VALIDATE
# ============================================================

def validate(text, city):
    if not text:
        return False
    if not text.startswith(city):
        return False
    wc = len(text.split())
    return 20 <= wc <= 70


# ============================================================
# WRITE
# ============================================================

def write(text, label):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")


# ============================================================
# GENERATE ONE SAMPLE
# ============================================================

def generate_one(city, prompt, label):
    for _ in range(6):
        text = call_llm(prompt)
        if validate(text, city):
            write(text, label)
            return True
    print("❌ Failed:", city, "(label:", label, ")")
    return False


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n===== Generating OOD Testset =====")

    # 10 implicit, 10 neutral (20 total OOD samples)
    for city in OOD_CITIES:
        print("IMPLICIT-HUMAN:", city)
        imp_prompt = IMPLICIT_OOD_HUMAN.format(city=city)
        generate_one(city, imp_prompt, 1)

        print("NEUTRAL-REAL:", city)
        base = random.choice(REAL_TRAVEL_SNIPPETS)
        neu_prompt = NEUTRAL_OOD_REAL.format(city=city, base=base)
        generate_one(city, neu_prompt, 0)

    print("\n🎉 DONE! OOD saved to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()

