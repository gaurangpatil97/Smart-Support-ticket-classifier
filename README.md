# Smart Support Ticket Classifier

A production-oriented end-to-end NLP system that classifies IT support tickets into 6 helpdesk categories using six parallel models — served via a Flask API with a Next.js demo frontend.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-13-black)](https://nextjs.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-orange)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)

---

## 🚀 Overview

Smart Support Ticket Classifier automates ticket triage by predicting which of six categories an incoming IT support ticket belongs to. It runs six models in parallel (classical, embedding-based, LoRA-fine-tuned, and LLMs) and returns predictions with confidence scores through a simple REST API.

Real-world use case: automated helpdesk ticket routing — route tickets to the right team (Fileservice, O365, Active Directory, etc.) without manual review.

---

## 📚 Dataset

- **Source:** Zenodo IT Support Tickets dataset (2,229 raw tickets; 7 original classes)
- **Data cleaning:**
  - 610 duplicate records detected and removed
  - EOL class dropped — 44/45 templates were identical, artificially inflating performance
- **Final training set:** 962 clean samples across 6 classes
- **Language breakdown:**

| Language | Count |
|---|---|
| English | 721 |
| Portuguese | 94 |
| German | 60 |
| Unknown | 50 |
| Spanish | 11 |

---

## 🧾 The 6 Classes

`Fileservice` `Support general` `Software` `O365` `Active Directory` `Computer-Services`

---

## 📈 Models & Results

| Model | Macro F1 | Accuracy | Real Tickets F1 | Custom 50 |
|---|---:|---:|---:|---:|
| Keras (dirty data) | 0.72 | 0.79 | — | — |
| TF-IDF + Logistic Regression | 0.69 | 0.75 | — | — |
| **Sentence Embeddings + LR ⭐** | **0.74** | **0.78** | **0.83** | 0.59 |
| GPT-4o mini zero-shot | 0.64 | 0.67 | 0.71 | 0.80 |
| GPT-4o mini few-shot | 0.71 | 0.72 | 0.68 | 0.86 |
| DistilBERT + LoRA A | 0.70 | 0.78 | 0.69 | 0.48 |
| DistilBERT + LoRA B (failed) | 0.11 | 0.30 | — | — |
| Embeddings + LR + Synthetic | 0.72 | 0.78 | 0.73 | 0.84 |

---

## 🔍 Key Findings

- **Dirty data silently inflated metrics** — EOL class produced F1 = 1.00 on 44 identical templates
- **Embeddings outperform TF-IDF** — +0.05 macro F1 and +0.09 on real ticket subsets
- **GPT excels on clean text, fails on noise** — strong on custom set, weak on real helpdesk language
- **Synthetic augmentation caused distribution shift** — real-ticket F1 dropped from 0.83 → 0.73
- **LoRA collapsed with noisy samples** — 24 EOL samples destabilized the entire fine-tune
- **Active Directory is the hardest class** — strong semantic overlap with Support general across all models

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Backend | Python, Flask, Flask-CORS, python-dotenv |
| ML | scikit-learn, sentence-transformers, transformers, PEFT, PyTorch |
| LLM | OpenAI API (GPT-4o mini) |
| Frontend | Next.js, Tailwind CSS, Recharts |

**Base models:**
- `paraphrase-multilingual-MiniLM-L12-v2` — sentence embeddings
- `distilbert-base-multilingual-cased` — LoRA fine-tuning
- `gpt-4o-mini` — zero-shot and few-shot classification

---

## 📁 Project Structure

```
Smart Support Ticket Classifier/
├── backend/
│   ├── app.py                        ← Flask API entrypoint
│   ├── models/
│   │   ├── tfidf_pipeline.pkl
│   │   ├── emb_classifier.pkl
│   │   ├── emb_augmented_classifier.pkl
│   │   └── lora model a/
│   │       ├── adapter_model.safetensors
│   │       ├── adapter_config.json
│   │       └── label_config.json
│   ├── static/images/                ← confusion matrices + training curves
│   ├── data/
│   └── notebooks/
├── frontend/
│   ├── app/
│   └── components/
└── it-support-ticket-report.html     ← full interactive analysis report
```

---

## ⚙️ Setup & Installation

### Backend

```bash
cd backend
pip install flask flask-cors python-dotenv scikit-learn sentence-transformers transformers peft torch openai numpy
```

Create a `.env` file in the backend folder:
```
OPENAI_API_KEY=sk-...
```

Run the server:
```bash
python app.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

> Requires Python 3.10+. Flask runs on port 5000, Next.js on port 3000.

---

## 🧭 API

### `POST /predict`

**Request:**
```json
{ "text": "I can't access my shared drive since this morning" }
```

**Response:**
```json
{
  "tfidf": { "label": "Fileservice", "confidence": 0.87 },
  "embeddings": { "label": "Fileservice", "confidence": 0.91 },
  "embeddings_augmented": { "label": "Fileservice", "confidence": 0.88 },
  "gpt_zeroshot": { "label": "Fileservice", "confidence": 0.95 },
  "gpt_fewshot": { "label": "Fileservice", "confidence": 0.95 },
  "lora": { "label": "Fileservice", "confidence": 0.83 }
}
```

### `GET /static/images/`
Serves confusion matrices and training curve images for all models.

---

## 🔁 Pipeline

```
Raw Ticket
    → Clean & Deduplicate
    → 6 parallel models:
         TF-IDF + LR
         Sentence Embeddings + LR  ⭐ winner
         Embeddings + LR (Synthetic)
         GPT-4o mini zero-shot
         GPT-4o mini few-shot
         DistilBERT + LoRA
    → Predicted Class + Confidence Score
    → Frontend / Routing
```

---

## ✨ Next Steps

- Collect more real helpdesk tickets to expand beyond the current 962 clean samples
- Retry LoRA fine-tuning on a larger, cleaner corpus — it showed promise but needs more data
- Fix Active Directory classification — add more diverse AD-specific tickets to reduce overlap with Support general
- Balance the multilingual distribution — English currently dominates at 77%
- Generate noisier synthetic data that better matches real helpdesk language patterns
- Add confidence calibration layer for LLM outputs
- Deploy with Docker for scalable API serving

## 📊 Presentation

[View Project Presentation (PDF)](./Smart_Support_Ticket_Classifier.pdf)

## 🧾 Acknowledgements

- Dataset: [Zenodo IT Support Tickets](https://zenodo.org/)
- Base models: [HuggingFace Transformers](https://huggingface.co/), [sentence-transformers](https://www.sbert.net/), [PEFT](https://github.com/huggingface/peft)
- LLM API: [OpenAI](https://openai.com/)