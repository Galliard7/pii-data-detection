# PII Data Detection — Kaggle Competition

## Overview

The [PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data) competition (2024) challenged participants to identify and classify Personally Identifiable Information (PII) in student essays. The task was token-level Named Entity Recognition (NER) with 7 PII entity types: `NAME_STUDENT`, `EMAIL`, `USERNAME`, `ID_NUM`, `PHONE_NUM`, `URL_PERSONAL`, and `STREET_ADDRESS`.

**Result: Public LB 0.970 / Private LB 0.956**

Kaggle profile: [illidan7](https://www.kaggle.com/illidan7)

## Approach

### 1. Synthetic Data Generation (The Differentiator)

The competition provided ~6,800 training essays but PII instances were sparse. The key insight was that generating high-quality synthetic data could significantly boost model performance. Built a 3-stage pipeline:

- **PII Generation**: Used Mistral 7B (4-bit quantized) + Faker + LangChain to generate realistic PII entities — names, emails, URLs, phone numbers, etc. The public version of this notebook received 14 upvotes.
- **Essay Generation**: Used Mistral 7B + LangChain prompts to generate full student essays. Also explored Gemma 7B on TPU (Keras 3 + JAX with SPMD across 8 TPUs) as an alternative.
- **Format Conversion**: SpaCy tokenization + BIO label alignment to convert generated essays into competition-compatible JSON format. Heavily iterated (24 versions).

This pipeline produced 3,000+ synthetic essays that augmented the training set.

### 2. Model Training (DeBERTa)

- Started from Valentin Werner's public DeBERTa-v3-base baseline (0.915 LB)
- Added synthetic data + external datasets (nbroad)
- Experimented with stride-based training for handling long essays
- Trained both DeBERTa-v3-base and DeBERTa-v3-large checkpoints
- Tracked experiments with Weights & Biases

### 3. Key Inference Insight: truncation=False

Discovered that disabling truncation entirely (letting the model process full documents) outperformed the prevalent stride-based sliding window approach by +0.02 on the leaderboard. This contradicted common competition wisdom that long documents needed chunking.

### 4. Ensemble Strategy

Evolved through several ensemble approaches:

1. **Weighted Softmax Ensemble** (67 iterations) — weighted average of model predictions with threshold tuning
2. **Slice Ensemble** — per-PII-type specialist models: different DeBERTa-v3-large checkpoints assigned to different entity types based on per-type F1 scores
3. **ONNX-Accelerated Ensemble** — adopted Lavrikov's ONNX framework to fit 8-9 models within Kaggle's time/memory constraints. BFloat16 quantization halved memory with minimal accuracy loss.
4. **Combined Lavrikov + Slice** — merged global weighted ensemble with per-entity-type specialists
5. **Grid Search** — systematic weight optimization across model combinations

### 5. Postprocessing

Regex-based detection for well-structured PII types (emails, phone numbers, URLs, street addresses) as a safety net alongside DeBERTa predictions.

## Repository Structure

```
├── eda/                          # Exploratory Data Analysis
│   └── data-exploration.ipynb    # Label distributions, SpaCy NER visualization
├── data-generation/              # Synthetic Data Pipeline
│   ├── mistral-pii-generation.ipynb        # ★ PII entity generation (14 upvotes)
│   ├── mistral-dataset-generation.ipynb    # Essay generation with Mistral 7B
│   ├── gemma-tpu-essay-generation.ipynb    # Gemma 7B on TPU (alternative approach)
│   └── competition-format-converter.ipynb  # SpaCy tokenization + BIO label conversion
├── training/                     # Model Training
│   ├── deberta3base-training.ipynb         # DeBERTa-v3-base with synthetic data
│   └── deberta3base-stride-training.ipynb  # Stride-based training experiment
├── inference/                    # Single-Model Inference
│   ├── deberta3base-inference.ipynb        # Main inference (14 versions)
│   └── deberta3base-truncation-false.ipynb # ★ Key insight: truncation=False > stride
├── ensemble/                     # Ensemble Methods
│   ├── weighted-ensemble.ipynb             # Weighted softmax ensemble (67 versions)
│   ├── slice-ensemble.ipynb                # ★ Per-PII-type specialist models
│   ├── onnx-ensemble-lavrikov.ipynb        # ONNX-accelerated multi-model ensemble
│   └── final-submission-0970.ipynb         # Final submission (Pub: 0.970 / Priv: 0.956)
└── utils/                        # Utilities
    ├── onnx-convert.ipynb                  # DeBERTa → ONNX conversion
    └── data-prep.ipynb                     # Data preparation + CV splits
```


## Kaggle Notebooks

Key notebooks published on Kaggle ([illidan7](https://www.kaggle.com/illidan7)):

- [PII-Detect-Mistral-PII-Generation](https://www.kaggle.com/code/illidan7/pii-detect-mistral-pii-generation) (14 upvotes)
- [PII-Detect-Mistral-Dataset-Generation](https://www.kaggle.com/code/illidan7/pii-detect-mistral-dataset-generation) (2 upvotes)
- [Public-Model-Ensemble [Pub: 0.970; Priv: 0.956]](https://www.kaggle.com/code/illidan7/public-model-ensemble-pub-0-970-priv-0-956) (2 upvotes)
- [Deberta3base Inference - TOA](https://www.kaggle.com/code/illidan7/deberta3base-inference-toa)
- [Deberta3base Train - TOA](https://www.kaggle.com/code/illidan7/deberta3base-train-toa)
- [Deberta3base Training - valentin](https://www.kaggle.com/code/illidan7/deberta3base-training-valentin)
- [PII-Detect-Data-Exploration](https://www.kaggle.com/code/illidan7/pii-detect-data-exploration)
- [PII-Detect - Deberta3base Inference](https://www.kaggle.com/code/illidan7/pii-detect-deberta3base-inference)
- [PII-Detect-Ensemble-Inference-Lavrikov](https://www.kaggle.com/code/illidan7/pii-detect-ensemble-inference-lavrikov)
- [PII-detect-Ensemble-Inference](https://www.kaggle.com/code/illidan7/pii-detect-ensemble-inference)
- [PII-Detect-Lavri+Slice-Inference](https://www.kaggle.com/code/illidan7/pii-detect-lavri-slice-inference)
- [PII-Detect-Lavri+Slice-ONNX-Inference](https://www.kaggle.com/code/illidan7/pii-detect-lavri-slice-onnx-inference)
- [PII-Detect-LLM-Data-Generation](https://www.kaggle.com/code/illidan7/pii-detect-llm-data-generation)
- [PII-Detect-Mistral7b-Data-Generation](https://www.kaggle.com/code/illidan7/pii-detect-mistral7b-data-generation)
- [PII-Detect-Mistral7b-Dataset-Generation](https://www.kaggle.com/code/illidan7/pii-detect-mistral7b-dataset-generation)

*Plus 4 more notebooks — see [full profile](https://www.kaggle.com/illidan7/code)*
## Tech Stack

- **Models**: DeBERTa-v3-base, DeBERTa-v3-large (HuggingFace Transformers)
- **Data Generation**: Mistral 7B (BitsAndBytes 4-bit), Gemma 7B (Keras 3 / JAX TPU), Faker, LangChain
- **NLP**: SpaCy (tokenization, NER visualization)
- **Optimization**: ONNX Runtime (GPU), BFloat16 quantization
- **Experiment Tracking**: Weights & Biases
- **Infrastructure**: Kaggle Notebooks (GPU T4/P100, TPU v3-8)

## Competition

- **Name**: [PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)
- **Type**: Token-level NER
- **Metric**: Micro-averaged F5 score (recall-weighted F-beta)
- **Timeline**: January — April 2024
