from sklearn.metrics import f1_score
import json

# Load evaluation metrics
with open("/home/ec2-user/SageMaker/HEARTS-Text-Stereotype-Detection/results/emgsd/metrics.json") as f:
    metrics = json.load(f)

macro_f1 = metrics["eval_macro_f1"]
print("Macro-F1 =", macro_f1)

# Compare with baseline
baseline_f1 = 0.78
error = abs(macro_f1 - baseline_f1)

print("Absolute Error =", error)
if error <= 0.05:
    print("Successfully reproduced the baseline (error within ±5%).")
else:
    print("Baseline NOT reproduced (error exceeds ±5%).")

