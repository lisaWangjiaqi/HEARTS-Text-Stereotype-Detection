from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch

model_path = "/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/results/emgsd/checkpoint-8580"

tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 加载 EMGSD test 数据
ds = load_dataset("holistic-ai/EMGSD", split="test")

# ⚠ 自动收集所有字符串标签
labels = sorted(list(set(ds["label"])))
print("Found labels:", labels)

# 创建 string → int 映射
name_to_id = {label: i for i, label in enumerate(labels)}

y_true = []
y_pred = []

for item in ds:
    # string → int
    y_true.append(name_to_id[item["label"]])

    text = item["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=-1).item()
        y_pred.append(pred)

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix — ALBERT on EMGSD")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
