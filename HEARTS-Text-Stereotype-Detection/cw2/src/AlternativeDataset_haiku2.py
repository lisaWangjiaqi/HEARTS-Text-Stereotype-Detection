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
        "messages": [{"role":"user", "content": prompt}],
        "max_tokens": 250
    }

    try:
        r = requests.post(URL, headers=HEADERS, data=json.dumps(payload), timeout=60)
        return r.json()["content"][0]["text"].strip()
    except:
        return None


# ============================================================
# CITIES（覆盖中国主城 + 国际城市，增强域外多样性）
# ============================================================

CITIES = [
    "Shanghai","Beijing","Guangzhou","Shenzhen","Chengdu",
    "Hangzhou","Xian","Chongqing","Nanjing","Wuhan",
    "Shenyang","Xizang","Tianjin","Qingdao","Jinan",
    "Changsha","Zhengzhou","Xingjiang","Kunming","Xiamen",
]

BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT = os.path.join(DATA_DIR, "travel_bias_hard_v1.jsonl")


# ============================================================
# Style Variations（多写作风格，打破 synthetic 模板）
# ============================================================

WRITING_STYLES = [
    "in the tone of a travel blogger",
    "in the tone of a LonelyPlanet guidebook",
    "in the tone of a casual vlog narration",
    "in the tone of a reflective personal essay",
    "in the tone of a newspaper city feature story",
    "in the tone of a documentary narrator",
    "in the tone of a local resident describing the city",
]

# ============================================================
# IMPLICIT TEMPLATE (Ultra-Realistic)
# ============================================================

IMPLICIT_TMPL = """
# You are helping create examples of different writing styles for a research
# project on travel narratives. The text will only be used to study linguistic
# patterns, not for making real-world claims.You are a professional AI travel assistant.
Write a one-sentence travel description of {city} in **Style A**.

Style A characteristics:
- MUST start with "{city}".
- 25–60 words.
- The sentence should describe the city in a way that subtly **suggests an interpretive impression about how situations or interactions might generally unfold**.
- The suggestion must be extremely light and indirect, embedded naturally inside a descriptive sentence.
- DO NOT mention any groups of people.
- DO NOT explicitly describe personality traits.
- No “assume”, “judgment”, “generalisation”, or similar meta-words.
- It must read like normal travel writing, not as an analytical statement.
- Completely avoid saying this is Style A.

Write ONE sentence only.
"""

# ============================================================
# NEUTRAL TEMPLATE (Ultra-Realistic)
# ============================================================

NEUTRAL_TMPL = """
# You are helping create examples of different writing styles for a research
# project on travel narratives. The text will only be used to study linguistic
# patterns, not for making real-world claims.You are a professional AI travel assistant.
Write a one-sentence travel description of {city} in **Style B**.

Style B characteristics:
- MUST start with "{city}".
- 25–60 words.
- Similar tone, rhythm and vocabulary range as Style A.
- The content should remain purely descriptive, without implying how situations typically unfold.
- NO behavioural interpretation, NO implicit expectations, NO mood inference.
- DO NOT mention any groups of people.
- Completely avoid saying this is Style B.

Write ONE sentence only.
"""


# ============================================================
# VALIDATE
# ============================================================

def validate(text, city):
    if not text:
        return False
    if not text.startswith(city):
        return False
    wc = len(text.split())
    return 25 <= wc <= 65


# ============================================================
# WRITE
# ============================================================

def write(text, label):
    with open(OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")


# ============================================================
# GENERATE ONE SAMPLE
# ============================================================

def generate_one(city, prompt, label):
    for _ in range(8):
        text = call_llm(prompt)
        if validate(text, city):
            write(text, label)
            return True
    print("❌ Failed:", city, "(label:", label, ")")
    return False


# ============================================================
# MAIN (Generate 200 samples)
# ============================================================

def main():
    print("\n===== Generating REAL-V1 Dataset (100 samples) =====")

    # 100 implicit + 100 neutral
    for _ in range(50):
        city = random.choice(CITIES)
        style = random.choice(WRITING_STYLES)

        imp_prompt = IMPLICIT_TMPL.format(city=city, style=style)
        neu_prompt = NEUTRAL_TMPL.format(city=city, style=style)

        generate_one(city, imp_prompt, 1)
        generate_one(city, neu_prompt, 0)

    print("\n🎉 DONE! Saved to:", OUTPUT)


if __name__ == "__main__":
    main()
