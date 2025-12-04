# COMP0173 Coursework 2 – HEARTS for Auditing AI Travel Assistants

> Student: Jiaqi Wang  
> UCL ID: 25032303 
> Course: COMP0173 – AI for Sustainable Development 
> Project: HEARTS for Auditing AI Travel Assistants

---

## 1. Overview

This folder (`cw2/`) contains my implementation for **Coursework 2**, building on the original  
paper:

> *HEARTS: A Holistic Framework for Explainable, Sustainable and Robust Text Stereotype Detection* (2024)

and its official codebase (this repository).

The coursework has two main goals:

1. **Replicate** the baseline HEARTS methodology on the EMGSD dataset (ALBERT-based stereotype detector).  
2. **Adapt** the model to a new context:  
   auditing **AI travel assistants** for **China/city stereotypes** in English travel descriptions.

The work is organised to directly support the five technical requirements of the coursework:  
baseline replication, new context definition, alternative dataset, model adaptation, and evaluation.

---

## 2. Relationship to the Original Repository

This repository is a **fork** of  
[`holistic-ai/HEARTS-Text-Stereotype-Detection`](https://github.com/holistic-ai/HEARTS-Text-Stereotype-Detection).

- All original code and notebooks remain under the existing folders:
  - `Exploratory Data Analysis/`
  - `Model Training and Evaluation/`
  - `Model Explainability/`
  - `LLM Bias Evaluation Exercise/`

- All **coursework-specific** code, notebooks and documentation are contained in the `cw2/` directory:
  - This makes it clear which parts are my contribution.

---

## 3. Repository Structure (Coursework Part)

```text
HEARTS-Text-Stereotype-Detection/
│
├── cw2/                                  
│   │
│   ├── src/                              # All codes + datasets + training results
│   │   │
│   │   ├── Travelbias_dataset/           
│   │   │   ├── EMGSD_raw/
│   │   │   ├── EMGSD_binary_clean_200.jsonl
│   │   │   ├── EMGSD_binary.jsonl
│   │   │   ├── merged_dataset.jsonl             # Final dataset (Travelbias_dataset + EMGSD)
│   │   │   ├── removed_v2.jsonl
│   │   │   ├── travel_bias_clean_aug.jsonl
│   │   │   ├── travel_bias_hard_v1.jsonl
│   │   │   ├── travel_bias_hard_v1-Copy1.jsonl
│   │   │   ├── travel_bias_hard_v2.jsonl        # Travelbias_dataset
│   │   │   ├── travel_bias_OOD.jsonl
│   │   │   └── travel_bias_OOD-Copy1.jsonl
│   │   │
│   │   ├── results/                      
│   │   │   ├── emgsd_baseline_albert/    # EMGSD baseline (ALBERT)
│   │   │   ├── improved_roberta/         # RoBERTa (Travelbias_dataset)
│   │   │   ├── improved_roberta_merge/   # RoBERTa (Travelbias_dataset + EMGSD)
│   │   │   └── travel_baseline/          
│   │   │
│   │   ├── AlternativeDataset_haiku.py 
│   │   ├── build_final_dataset.py        # dataset
│   │   ├── clean_dataset.py              # dataset
│   │   ├── emgsd_score_dataset.py
│   │   ├── evaluate_ood.py               # evaluate model
│   │   ├── plot_confusion_matrix.png
│   │   ├── split_dataset.py
│   │   ├── test_prompt.py
│   │   ├── train_baseline_travel.py
│   │   ├── train_emgsd_albert.py          # Baseline Reproduction
│   │   ├── train_roberta_travel.py        # RoBERTa (Travelbias_dataset)
│   │   └── train_roberta_travel_merged.py # RoBERTa (Travelbias_dataset + EMGSD) 
│   │
│   │
│   └── poster/                           # Poster 
│       └── cw2_poster.pdf
│
├── Exploratory Data Analysis/            
├── LLM Bias Evaluation Exercise/         
├── Model Explainability/                 
├── Model Training and Evaluation/        
│
├── LICENSE
├── README.md
└── requirements.txt
```

## 4. Baseline Reproduction (EMGSD → ALBERT)
The baseline model from the HEARTS paper was reproduced using the EMGSD dataset.
train_emgsd_albert.py

Command used:
```bash
python train_emgsd_albert.py \
  --model_name albert-base-v2 \
  --output_dir results/emgsd_baseline_albert \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --num_epochs 4 \
  --seed 42
```


## 5. Dataset
| Source                                              | Count   |
| --------------------------------------------------- | ------- |
| Original EMGSD dataset                              | 200     |
| Travel stereotype dataset (LLM generated + cleaned) | 539     |
| **Total(Merged Dataset)**                           | **739** |

Out-of-distribution test set: 

```bash
travel_bias_OOD.jsonl
```

Dataset format:
```jsonl
{"text": "People in {city} always…", "label": 1}
{"text": "{city} is a vibrant city with...", "label": 0}
```
1 = stereotype
0 = neutral text


## 6. 🔥 Improved Model
RoBERTa: 
```bash
python train_roberta_travel_merge.py
```

Model and results are stored under:
```bash
cw2/src/results/
```

## 7. 📊 Evaluation
Accuracy
Macro-F1
Confusion Matrix
```bash
python evaluate_ood.py
```


## 8. Reproducing the Entire Project

Prepare datasets
```bash
cw2/src/Travelbias_dataset/
```

Train travel stereotype models
```bash
python cw2/src/train_roberta_travel_merged.py
```

Run evaluation
```bash
python cw2/src/evaluate_ood.py
```






  
