# 🕵️ FidesAI: Fake News Detection — Dual-Engine NLP System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**A production-grade NLP system that runs two independent model architectures in parallel — a fine-tuned RoBERTa transformer and a TF-IDF + Logistic Regression baseline — to classify news articles as real or fake with explainable confidence scores.**

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Results & Performance](#-results--performance)
- [Deep ML Insights — Model Divergence Phenomenon](#-deep-ml-insights--model-divergence-phenomenon)
- [Project Structure](#-project-structure)
- [Installation & Quickstart](#-installation--quickstart)
- [Next Steps](#-next-steps)

---

## 🧭 Project Overview

This project was built as a full-stack NLP portfolio piece that goes well beyond a single notebook. The core idea: **no single model architecture is trustworthy enough on its own** when dealing with adversarially crafted misinformation. The solution is a dual-engine pipeline that runs both a statistical and a deep learning classifier independently and surfaces disagreements to the user as a signal of uncertainty.

**Key engineering highlights:**
- Fine-tuned `roberta-base` on a combined WELFake + ISOT dataset achieving **94–96% F1-Score**
- TF-IDF + Logistic Regression classical baseline achieving **87–89% accuracy** for rapid comparison
- Resilient inference engine with **isolated component failure boundaries** — a crash in one model never takes down the other
- Streamlit interface with half-precision model loading (`torch.float16`) to keep RAM usage under 1GB
- Full experiment tracking across all training runs

---

## 🏗️ Architecture

```
                  ┌──► TF-IDF Vectorizer ──► Logistic Regression ──► P(fake) [Statistical]
                  │
[Raw News Text] ──┤──► Text Cleaner (utils.py)
                  │
                  └──► RoBERTa Tokenizer ──► Fine-tuned Transformer ──► P(fake) [Semantic]

Both outputs feed into a unified inference response:
  → Label (REAL: 1, FAKE: 0)
  → Confidence score per model
  → Divergence flag if models disagree
```

The two pipelines are **fully decoupled at the state level** inside `src/predict.py` using isolated boolean flags (`self.has_classical`, `self.has_transformer`). A failure in the classical pipeline — whether a missing `.joblib` file, a disk I/O lag, or a version mismatch exception — does not propagate to the transformer engine, and vice versa.

---

## 📂 Dataset

Training and evaluation used a **combined corpus of three public datasets**, merged and deduplicated:

| Dataset                    | Size            | Download Link                                                                           |
|----------------------------|-----------------|-----------------------------------------------------------------------------------------|
| **WELFake**                | 72,134 articles | [Dataset Link](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| **ISOT Fake/True News**    | 44,898 articles | [Dataset Link](https://www.kaggle.com/datasets/rahulogoel/isot-fake-news-dataset)       |
| **Combined (after dedup)** | 117,032 samples | —                                                                                       |

- **Split:** 70% train / 15% validation / 15% test (stratified)
- **Test set is strictly unseen** — the `processed_test_df.csv` and `sample_suggestions.csv` sandbox were never touched during any training run
- **Class balance:** approximately 51% fake, 49% real after deduplication

---

## 📈 Results & Performance

Evaluation on the held-out test split across all architectures:

| Model | Accuracy   | F1-Score   | AUC-ROC     | Precision  | Recall     |
|---|------------|------------|-------------|------------|------------|
| TF-IDF + Logistic Regression | 88.1%      | 87.6%      | 0.924       | 86.9%      | 88.3%      |
| TF-IDF + LinearSVC | 89.3%      | 88.9%      | 0.931       | 88.1%      | 89.7%      |
| **Fine-tuned RoBERTa** | **99.85%** | **99.86%** | **~0.9985** | **99.76%** | **99.96%** |
---

## 🧠 Deep ML Insights — Model Divergence Phenomenon

During evaluation using the Quick-Test Suggestions Sandbox (`sample_suggestions.csv` — strictly unseen data), a fascinating and repeatable phenomenon was identified:

> **Both models evaluated the same headline with 100% confidence but arrived at completely opposite classifications.**

This was named **Model Divergence (Semantic Dissociation)**. It has two distinct root causes:

---

### Phenomenon A — Feature Space Misalignment (The Baseline Trap)

The TF-IDF pipeline evaluates text as a **sparse frequency matrix of token probabilities**. It has no awareness of sentence structure, negation, or context — only token co-occurrence statistics from training.

If a fabricated article contains an ultra-high concentration of authoritative vocabulary that appeared predominantly in genuine articles during training, the Logistic Regression model maps the token distribution to the positive class with high statistical certainty. It cannot distinguish between:

- *"Scientists confirm vaccine is effective"* (real)
- *"Scientists confirm vaccine is a bioweapon"* (fake with identical authoritative token profile)

The word `Scientists` + `confirm` appears overwhelmingly in real articles in the training set, so the baseline is blind to the rest of the sentence.

---

### Phenomenon B — Adversarial Framing & Calibration Drift (The Transformer Trap)

RoBERTa is susceptible to **overconfidence caused by the mathematical behavior of the Softmax function**:

$$P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

When logits are large and well-separated, Softmax outputs approach 1.0 regardless of whether the model is actually correct. This is **calibration drift** — confidence does not equal accuracy.

If a bad actor writes a fake story using **flawless, formal journalistic prose** — impeccable punctuation, typical mainstream capitalization, professional sentence structure — RoBERTa's self-attention layers flag the *style* as characteristic of real journalism and outputs a high-confidence "Real" label.

The model is evaluating the **stylistic fingerprint of professional journalism**, not the factual accuracy of the claims.

---

### Why the Dual-Engine Architecture Matters

This divergence behavior is actually **useful as a signal**, not a failure:

| Scenario | Classical | Transformer | Meaning |
|---|---|---|---|
| Both agree → REAL | REAL | REAL | High confidence genuine |
| Both agree → FAKE | FAKE | FAKE | High confidence fake |
| **Divergence** | **REAL** | **FAKE** | ⚠️ Possible adversarial framing |
| **Divergence** | **FAKE** | **REAL** | ⚠️ Keyword-heavy but stylistically clean |

When the two engines disagree, the system surfaces a **divergence flag** to the user rather than forcing a single verdict. This is a deliberate design choice — uncertainty should be communicated, not hidden.

---

## 📂 Project Structure

```
📁 Fake_News_Detection/
│
├── 📂 artifacts/
│   ├── 📂 roberta_model/checkpoint-8232  # Fine-tuned RoBERTa transformer checkpoints
│   ├── 📄 lr_model.joblib                # Optimized Logistic Regression baseline binary
│   ├── 📄 svc_model.joblib               # Trained Support Vector Classifier model binary
│   ├── 📄 tfidf_vectorizer.joblib        # Fitted TF-IDF vectorizer configuration metadata
│   └── 🖼️ top_baseline_confusion_matrix.png # Baseline model performance evaluation matrix
│
├── 📂 configs/
│   └── 📄 roberta_config.yaml            # Hyperparameter configurations for transformer training
│
├── 📂 data/
│   ├── 📂 processed/
│   │   ├── 📄 processed_test_df.csv      # Stratified, vaulted testing split (strictly unseen)
│   │   ├── 📄 processed_train_df.csv     # Training partition data frame
│   │   ├── 📄 processed_val_df.csv       # Evaluation split partition data frame
│   │   └── 📄 sample_suggestions.csv     # Unified matrix for the sandbox UI component
│   └── 📂 raw/
│       ├── 📄 ISOT_Fake.csv              # Source raw benchmark data: ISOT False subset
│       ├── 📄 ISOT_True.csv              # Source raw benchmark data: ISOT True subset
│       └── 📄 WELFake.csv                # Source raw benchmark data: WELFake corpus
│
├── 📂 notebooks/
│   ├── 📂 google_collab/                 # Cloud-based training environments
│   │   ├── 📓 TextCleaner_collab_notebook.ipynb
│   │   └── 📓 Trainer_roberta_collab_notebook.ipynb
│   ├── 📓 data_loader.ipynb
│   ├── 📓 evaluate.ipynb
│   ├── 📓 evalution_transformer.ipynb    # Validation metrics extraction notebook
│   ├── 📓 features.ipynb                 # Feature engineering playground
│   ├── 📓 preprocess.ipynb
│   ├── 📓 pridict.ipynb                 # Local prediction exploration notebook
│   ├── 📓 sample_df.ipynb
│   ├── 📓 train.ipynb                    # Baseline modeling workbook
│   └── 📓 train_transformer.ipynb        # Local transformer iteration testing script
│
├── 📂 src/
│   ├── 📄 app.py                         # Multi-threaded Streamlit user interface entrypoint
│   ├── 📄 data_loader.py                 # Structured pipeline ingestion module
│   ├── 📄 evaluate.py
│   ├── 📄 evalution_transformer.py
│   ├── 📄 features.py                    # Production tokenization and vectorization scripts
│   ├── 📄 predict.py                     # Parallel dual-engine asynchronous inference architecture
│   ├── 📄 preprocess.py                  # Core text standardization and cleaning pipeline
│   ├── 📄 train.py
│   ├── 📄 train_transformer.py
│   └── 📄 utils.py                       # General helper subroutines
│
├── 📝 README.md                          # Structured technical portfolio overview document
└── 📋 requirements.txt                   # Explicitly locked environment dependency index
```

---

## 🚀 Installation & Quickstart

**1. Clone the repository**
```bash
git clone https://github.com/codebywolf/Data-Science.git
cd 'Machine learning Projects/FidesAI_Fake_News_Detection'
```

**2. Create a virtual environment**
```bash
python -m venv spacy_env
source fidesai_env/bin/activate        # macOS/Linux
fidesai_env\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download model artifacts**

Place the fine-tuned RoBERTa checkpoint folder and `.joblib` files inside `artifacts/` as per the directory structure above. Pre-trained weights are available at: `[amansshrrma/RoBERTa_pre_trained]`

**5. Run the Streamlit app**
```bash
streamlit run src/app.py
```
---

## 🗺️ Next Steps

- [ ] **Dynamic Model Calibration** — Implement Platt Scaling or temperature tuning on RoBERTa's raw logit outputs to reduce overconfident false positives and bring confidence scores closer to true probabilities
- [ ] **SHAP Explainability Layer** — Inject SHAP word-level visualizers into the Streamlit interface to highlight exactly which tokens are driving each model's prediction in real time


---
<div align="center">

Built with curiosity, debugged with patience.

*Real engineering is measured not by the features you ship, but by the failures you document.*

</div>
