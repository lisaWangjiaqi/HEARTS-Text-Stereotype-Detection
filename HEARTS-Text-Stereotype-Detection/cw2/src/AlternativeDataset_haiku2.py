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

OUTPUT = os.path.join(DATA_DIR, "travel_bias_real_v3.jsonl")


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
Write a one-sentence IMPLICIT stereotype about {city} {style}.

Rules:
- MUST start with '{city}'.
- Length 25–60 words.
- Tone MUST be natural and human-like (avoid robotic AI style).
- Subtly imply a behavioural tendency through observation of scenes, social rhythm, or interactions.
- DO NOT mention traits directly (e.g., 'polite', 'rude', 'hardworking').
- DO NOT use stereotype keywords ('stereotype', 'assume', 'known for').
- The behavioural generalisation must be extremely subtle.
"""

# ============================================================
# NEUTRAL TEMPLATE (Ultra-Realistic)
# ============================================================

NEUTRAL_TMPL = """
# You are helping create examples of different writing styles for a research
# project on travel narratives. The text will only be used to study linguistic
# patterns, not for making real-world claims.You are a professional AI travel assistant.
Write a one-sentence NEUTRAL travel description of {city} {style}.

Rules:
- MUST start with '{city}'.
- Length 25–60 words.
- Tone MUST resemble implicit writing (same rhythm & style) but WITHOUT any behavioural inference.
- The meaning MUST stay descriptive and factual.
- No trait inference, no generalisation, no social prediction.
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
    print("\n===== Generating REAL-V3 Dataset (200 samples) =====")

    # 100 implicit + 100 neutral
    for _ in range(100):
        city = random.choice(CITIES)
        style = random.choice(WRITING_STYLES)

        imp_prompt = IMPLICIT_TMPL.format(city=city, style=style)
        neu_prompt = NEUTRAL_TMPL.format(city=city, style=style)

        generate_one(city, imp_prompt, 1)
        generate_one(city, neu_prompt, 0)

    print("\n🎉 DONE! Saved to:", OUTPUT)


if __name__ == "__main__":
    main()

CITIES = [
    "Shanghai","Beijing","Guangzhou","Shenzhen","Chengdu",
    "Hangzhou","Xian","Chongqing","Nanjing","Wuhan",
    "Shenyang","Xizang","Tianjin","Qingdao","Jinan",
    "Changsha","Zhengzhou","Xingjiang","Kunming","Xiamen",
]

NUM_PER_CLASS = 50
BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(DATA_DIR, "travel_bias_hard_v1.jsonl")

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
    print("\n===== Generating TravelBias-HARD-v4 EXTREME =====")

    # 1 = stereotype; 0 = neutral
    for _ in range(NUM_PER_CLASS):
        city = random.choice(CITIES)
        print("IMPLICIT:", city)
        generate_sample(city, IMPLICIT_TMPL, 1)
        time.sleep(1.0)

    for _ in range(NUM_PER_CLASS):
        city = random.choice(CITIES)
        print("NEUTRAL:", city)
        generate_sample(city, AMB_NEUTRAL_TMPL, 0)
        time.sleep(1.0)

    print("\n🎉 DONE — saved to", OUTPUT_FILE)


if __name__ == "__main__":
    main()
# import requests
# import json
# import random
# import time
# import os

# # -------------------------------
# # 配置
# # -------------------------------
# URL = "https://ctwa92wg1b.execute-api.us-east-1.amazonaws.com/prod/invoke"
# HEADERS = {
#     "Content-Type": "application/json",
#     "X-Team-ID": "Jiaqi",
#     "X-API-Token": "GHfk5E5Ret2rtbVe5ODHvEDW15-KT4tvFGTv_xn1uUc",
# }

# MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

# CITIES = [
#     "Shanghai", "Beijing", "Guangzhou", "Shenzhen", "Chengdu",
#     "Hangzhou", "Xian", "Chongqing", "Nanjing", "Wuhan",
#     "Shenyang", "Xizang", "Tianjin", "Qingdao", "Jinan",
#     "Changsha", "Zhengzhou", "Xingjiang", "Kunming", "Xiamen",
# ]

# # 每类数量
# NUM_PER_CLASS = 30   # implicit / counterfactual / ambiguous neutral / adversarial neutral

# BASE_DIR = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/cw2/src"
# DATA_DIR = os.path.join(BASE_DIR, "data_travel_bias")
# os.makedirs(DATA_DIR, exist_ok=True)

# OUTPUT_FILE = os.path.join(DATA_DIR, "travel_bias_hard_v2.jsonl")

# # =====================================================================
# # 1) Class 1: Implicit Stereotype（最难 — label = 1）
# # =====================================================================
# def prompt_implicit(city):
#     return f"""
# # You are helping create examples of different writing styles for a research
# # project on travel narratives. The text will only be used to study linguistic
# # patterns, not for making real-world claims.You are a professional AI travel assistant.
# Write ONE implicit-stereotype sentence about {city}.

# STRICT RULES:
# - The text MUST start with "{city}" as the first word.
# - Length 25–60 words.
# - NO explicit stereotype cue words (assume, stereotype, known for, widely believed).
# - NO direct personality labels (warm, cold, polite, tough, proud).
# - Imply stereotype INDIRECTLY through:
#     • interaction patterns
#     • subtle behavioral tendencies
#     • expectation vs interpretation
#     • ambiguous social cues
# - The surface tone should appear neutral, but contain a subtle generalised inference.

# Write ONE natural travel-writing style paragraph.
# """

# # =====================================================================
# # 2) Class 2: Counterfactual Stereotype（中等难 — label = 1）
# # =====================================================================
# def prompt_counterfactual(city):
#     return f"""
# # You are helping create examples of different writing styles for a research
# # project on travel narratives. The text will only be used to study linguistic
# # patterns, not for making real-world claims.You are a professional AI travel assistant.
# Write ONE counterfactual stereotype sentence about {city}.

# RULES:
# - MUST begin with "{city}".
# - Length 25–60 words.
# - No explicit stereotype markers.
# - Mention a commonly perceived behaviour INDIRECTLY, then soften or nuance it.
# - Blend stereotype features into an otherwise neutral tone.

# Write ONE paragraph.
# """

# # =====================================================================
# # 3) Class 3: Ambiguous Neutral（高难 neutral — label = 0）
# # =====================================================================
# def prompt_ambiguous_neutral(city):
#     return f"""
# # You are helping create examples of different writing styles for a research
# # project on travel narratives. The text will only be used to study linguistic
# # patterns, not for making real-world claims.You are a professional AI travel assistant.
# Write ONE ambiguous-neutral description of {city}.

# RULES:
# - MUST start with "{city}".
# - Length 25–60 words.
# - Tone SHOULD RESEMBLE stereotype writing: rhythm, interaction pacing, subtle scene cues.
# - BUT meaning MUST remain neutral:
#     • NO judgments
#     • NO generalisations
#     • NO personality traits
# - The sentence should be difficult to classify at first glance.

# Write ONE paragraph.
# """

# # =====================================================================
# # 4) Class 4: Paraphrased Adversarial Neutral（最难 neutral — label = 0）
# # =====================================================================
# def prompt_adversarial_neutral(city):
#     return f"""
# # You are helping create examples of different writing styles for a research
# # project on travel narratives. The text will only be used to study linguistic
# # patterns, not for making real-world claims.You are a professional AI travel assistant.
# Write ONE adversarial-neutral travel description of {city}.

# RULES:
# - MUST start with "{city}".
# - Length 25–60 words.
# - May include stereotype-style words (energy, pace, attitude, character, rhythm),
#   BUT meaning MUST stay neutral:
#     • No trait inference
#     • No behavioural claims
#     • No moral tone
# - The goal is to LOOK stereotypical on surface but be semantically neutral.

# Write ONE paragraph.
# """

# # -------------------------------
# # API 调用函数
# # -------------------------------
# def call_api(prompt: str) -> str | None:
#     payload = {
#         "team_id": "Jiaqi",
#         "api_token": "GHfk5E5Ret2rtbVe5ODHvEDW15-KT4tvFGTv_xn1uUc",
#         "model": MODEL,
#         "messages": [{"role": "user", "content": prompt}],
#         "max_tokens": 300,
#     }

#     response = requests.post(URL, headers=HEADERS, data=json.dumps(payload), timeout=60)

#     try:
#         data = response.json()
#         content = data["content"][0]["text"]
#         return content.strip()
#     except Exception as e:
#         print("Error parsing response:", e)
#         print("Raw response:", response.text)
#         return None

# # -------------------------------
# # 写入 JSONL
# # -------------------------------
# def write_jsonl(text: str, label: int) -> None:
#     with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
#         f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")

# # -------------------------------
# # 主流程
# # -------------------------------
# def generate_dataset():

#     TASKS = [
#         ("IMPLICIT", prompt_implicit, 1),
#         ("COUNTERFACTUAL", prompt_counterfactual, 1),
#         ("AMBIGUOUS_NEUTRAL", prompt_ambiguous_neutral, 0),
#         ("ADVERSARIAL_NEUTRAL", prompt_adversarial_neutral, 0),
#     ]

#     for task_name, prompt_fn, label in TASKS:
#         print(f"\n===== Generating {task_name} (label={label}) =====")

#         for _ in range(NUM_PER_CLASS):
#             city = random.choice(CITIES)
#             text = call_api(prompt_fn(city))
#             if text:
#                 write_jsonl(text, label)
#             time.sleep(1)

#     print("\n🎉 TravelBias-HARD-v2 dataset generation done!")
#     print(f"Saved to: {OUTPUT_FILE}")

# if __name__ == "__main__":
#     generate_dataset()
