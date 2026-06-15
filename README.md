# ECHR Court Decision Analysis with Legal-BERT

Fine-tuning `nlpaueb/legal-bert-base-uncased` on 11,478 European Court of Human Rights judgments to predict violation outcomes and classify violated Convention articles.

## Results

| Task | Test Macro F1 | Accuracy |
|------|:------------:|:--------:|
| Task 1: Violation vs No Violation | **0.84** | **0.87** |
| Task 2: Article Classification (7 classes) | **0.62** | **0.70** |

## Overview

This project addresses two classification tasks using ECHR judgment texts:

**Task 1 — Binary Classification:** Predict whether a judgment results in a violation of the European Convention on Human Rights.

**Task 2 — Multi-class Classification:** Identify which Convention article was violated (Art. 2, 3, 5, 6, 8, 10, or Protocol 1 Art. 1).

## Dataset

[glnmario/ECHR](https://huggingface.co/datasets/glnmario/ECHR) — 11,478 English ECHR judgments (1960–2018), pre-split into train/val/test partitions.

| Split | Task 1 | Task 2 |
|-------|-------:|-------:|
| Train | 7,100 | 3,293 |
| Val | 1,380 | 650 |
| Test | 2,998 | 1,822 |

## Key Findings

**Task 1:**
- Violation recall of **0.99** — the model captures almost all actual violations
- No Violation recall of **0.63** — harder to classify due to heterogeneous inadmissible and mixed cases

**Task 2:**
- Art. 6 (Fair Trial, F1: 0.79) and Art. 3 (Prohibition of Torture, F1: 0.74) perform best, benefiting from larger training sets
- Art. 1 (Protection of Property, F1: 0.26) suffers from severe data imbalance (only 54 training examples)
- Most common misclassification: **Art. 5 → Art. 3** (121 cases) — detention and ill-treatment cases share overlapping legal language
- Country-level analysis shows Montenegro, Portugal, and Serbia achieve the highest accuracy; Lithuania and Croatia perform below average

## Methods & Tools

**Language:** Python  
**Model:** `nlpaueb/legal-bert-base-uncased` (Legal-BERT)  
**Key packages:** `transformers`, `datasets`, `torch`, `scikit-learn`, `pandas`, `seaborn`  
**Training:** AdamW optimizer, linear warmup scheduler, gradient clipping (max norm 1.0), early stopping based on val F1  
**Hardware:** Tesla T4 GPU (Google Colab)

## Model Configuration

```python
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LEN    = 512   # BERT max token length
BATCH_SIZE = 16
LR         = 2e-5
EPOCHS_T1  = 3     # Binary classification
EPOCHS_T2  = 4     # Article classification
```

## Repository Structure

```
├── echr_analysis_final.ipynb   # Full analysis notebook with embedded results
└── README.md
```

## How to Run

```bash
pip install transformers datasets torch scikit-learn pandas seaborn
```

Open `echr_analysis_final.ipynb` in Google Colab with a GPU runtime (Runtime → Change runtime type → T4 GPU) and run all cells. The dataset is loaded automatically from Hugging Face.

> ⚠️ Training takes approximately 1–2 hours on a T4 GPU. Pre-computed results and visualizations are embedded in the notebook.

## Authors

Pelin Kımız · Anıl Yazıcı  
Constructor University Bremen — Data Science for Society and Business
