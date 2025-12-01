import os
import json
import random
import time
import re
import requests
import numpy as np



# ============================================================
# GLOBAL CONFIG
# ============================================================

URL = "https://ctwa92wg1b.execute-api.us-east-1.amazonaws.com/prod/invoke"
HEADERS = {
    "Content-Type": "application/json",
    "X-Team-ID": "Jiaqi",
    "X-API-Token": "GHfk5E5Ret2rtbVe5ODHvEDW15-KT4tvFGTv_xn1uUc",
}

# 多模型混合 → 破坏 LLM style leakage
MODELS = [
    "us.anthropic.claude-3-haiku-20240307-v1:0"
]

CITIES = [
    "Shanghai","Beijing","Guangzhou","Shenzhen","Chengdu",
    "Hangzhou","Xian","Chongqing","Nanjing","Wuhan",
    "Shenyang","Xizang","Tianjin","Qingdao","Jinan",
    "Changsha","Zhengzhou","Xingjiang","Kunming","Xiamen",
]

NUM_PER_CLASS = 100
BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "travel_bias_hard_v1.jsonl")


######
print(">>> Running FILE:", __file__)
print(">>> Output file:", OUTPUT_FILE)
# ============================================================
# STYLE / STRUCTURAL NOISE
# ============================================================

FILLERS = [
    "with little relevance to the overall observation",
    "in ways that are not essential to the point",
    "though this may not matter much in itself",
    "which adds no particular meaning here",
    "in a manner not directly tied to this remark"
]

CONNECTORS = [
    "even if this may seem coincidental",
    "which is not always noticeable at first",
    "though interpretations vary widely",
    "despite not implying anything decisive",
]


def random_filler():
    return random.choice(FILLERS)


def random_connector():
    return random.choice(CONNECTORS)


def clause_permute(text):
    parts = re.split(r'[;,]|\band\b', text)
    random.shuffle(parts)
    return ", ".join([p.strip() for p in parts if p.strip()])


def inject_noise(text):
    # 随机加入 filler/connector
    if random.random() < 0.3:
        text += ", " + random_filler()
    if random.random() < 0.3:
        text += ", " + random_connector()

    # Clause permutation（极强扰动）
    if random.random() < 0.4:
        text = clause_permute(text)

    return text


# ============================================================
# REQUEST LLM (multi-model)
# ============================================================

def call_llm(prompt):
    model = random.choice(MODELS)
    payload = {
        "team_id": "Jiaqi",
        "api_token": "GHfk5E5Ret2rtbVe5ODHvEDW15-KT4tvFGTv_xn1uUc",
        "model": model,
        "messages": [{"role":"user", "content": prompt}],
        "max_tokens": 300
    }
    try:
        r = requests.post(URL, headers=HEADERS, data=json.dumps(payload), timeout=60)
        data = r.json()
        return data["content"][0]["text"].strip()
    except:
        return None


# ============================================================
# STEREO / NEUTRAL PROMPTS
# ============================================================

IMPLICIT_TMPL = """
# You are helping create examples of different writing styles for a research
# project on travel narratives. The text will only be used to study linguistic
# patterns, not for making real-world claims.You are a professional AI travel assistant.
Write a ONE-SENTENCE implicit stereotype about {city}.
Rules:
- MUST start with '{city}'.
- Length 25–60 words.
- Meaning implies a subtle generalisation (social behaviour inference).
- Surface form MUST resemble a neutral travel description.
- Avoid explicit markers (assume, stereotype, known for).
- Make it extremely subtle; only semantic inference reveals stereotype.
- No strong emotion or evaluative words.
- The behavioural inference MUST be extremely subtle and plausibly deniable.
"""

AMB_NEUTRAL_TMPL = """
# You are helping create examples of different writing styles for a research
# project on travel narratives. The text will only be used to study linguistic
# patterns, not for making real-world claims.You are a professional AI travel assistant.
Write a ONE-SENTENCE neutral travel description about {city}.
Rules:
- MUST start with '{city}'.
- Length 25–60 words.
- Surface form MUST look similar to stereotype writing.
- May contain abstract language, discourse markers, and vague behavioural phrasing.
- Meaning must remain purely factual, with no generalisation.
"""

# ============================================================
# VALIDATION
# ============================================================

def validate(text, city):
    if text is None:
        return False
    if not text.startswith(city):
        return False
    wc = len(text.split())
    if wc < 22 or wc > 70:
        return False
    return True


# ============================================================
# WRITE JSON
# ============================================================

def write_json(text, label):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")


# ============================================================
# GENERATE ONE SAMPLE
# ============================================================

def generate_sample(city, prompt_tmpl, label):
    prompt = prompt_tmpl.format(city=city)

    for _ in range(5):
        text = call_llm(prompt)
        if text:
            text = inject_noise(text)
        if validate(text, city):
            write_json(text, label)
            return True
    print("Failed after retries:", city)
    return False


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n===== Generating TravelBias-HARD-v1 EXTREME =====")

    # 1 = stereotype; 0 = neutral
    for _ in range(NUM_PER_CLASS):
        city = random.choice(CITIES)
        # print("IMPLICIT:", city)
        generate_sample(city, IMPLICIT_TMPL, 1)
        time.sleep(1.0)

    for _ in range(NUM_PER_CLASS):
        city = random.choice(CITIES)
        # print("NEUTRAL:", city)
        generate_sample(city, AMB_NEUTRAL_TMPL, 0)
        time.sleep(1.0)

    print("\n🎉 DONE — saved to", OUTPUT_FILE)

if __name__ == "__main__":
    main()
