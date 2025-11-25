# COMPxxxx Coursework 2 – HEARTS for Auditing AI Travel Assistants

> Student: Jiaqi Wang  
> UCL ID: 25032303 
> Course: COMP0173 – AI for Sustainable Development 
> Project: Auditing AI Travel Assistants for National Stereotypes with HEARTS

---

## 1. Overview

This folder (`cw2/`) contains my implementation for **Coursework 2**, building on the original  
paper:

> *HEARTS: A Holistic Framework for Explainable, Sustainable and Robust Text Stereotype Detection* (2024)

and its official codebase (this repository).

The coursework has two main goals:

1. **Replicate** the baseline HEARTS methodology on the EMGSD dataset (ALBERT-based stereotype detector).  
2. **Adapt** the model to a new context:  
   auditing **AI travel assistants** for **country/city stereotypes** in English travel descriptions.

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
cw2/
├─ README_cw2.md                 # This file – coursework documentation
├─ notebooks/
│   ├─ 01_emgsd_baseline_reproduce.ipynb
│   ├─ 02_travel_dataset_building.ipynb
│   └─ 03_travel_model_evaluation.ipynb
│
├─ src/
│   ├─ train_emgsd_albert.py
│   ├─ build_travel_dataset.py
│   ├─ train_travel_stereotype.py
│   └─ explain_travel_examples.py
│
├─ configs/
│   ├─ emgsd_albert_config.json
│   └─ travel_albert_config.json
│
├─ results/
│   ├─ emgsd_baseline/
│   ├─ travel_baseline/
│   └─ travel_finetune/
│
└─ poster/
    └─ cw2_poster.pdf
