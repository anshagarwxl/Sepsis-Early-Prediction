# 🩺 Sepsis Early Warning RAG Assistant

A lightweight AI-powered assistant that combines **clinical risk scoring** and **retrieval-augmented generation (RAG)** to support frontline healthcare workers in the **early detection and management of sepsis**.

---

## 🚨 Problem Statement

- Sepsis is a **life-threatening syndrome** caused by the body’s extreme response to infection.  
- Every **hour of delayed treatment increases mortality by ~8%**.  
- Globally, **1 in 3 patients with sepsis die**.  
- In India, there is **only 1 doctor per 1000 people**, leading to overcrowding and diagnostic delays.  
- Manual tools (**NEWS2, qSOFA, SIRS**) are rarely used in busy hospitals.  

👉 **Result:** Many patients slip into **septic shock** before receiving timely care.

---

## 💡 Solution

The **Sepsis Early Warning RAG Assistant** is a **Streamlit-based app** that enables clinicians to:

- ✅ Enter patient vitals (temperature, HR, BP, SpO₂, consciousness, WBC).  
- ✅ Instantly calculate sepsis risk scores: **NEWS2, qSOFA, SIRS**.  
- ✅ Cluster patients into risk profiles using **KMeans**.  
- ✅ Ask clinical questions → receive **guideline-grounded answers with citations** (via RAG).  

**Workflow:**  
`Vitals ➝ Scores + Cluster ➝ Clinical Question ➝ Evidence-based Recommendation`

---

## 🛠 Tech Stack

- **Frontend:** Streamlit (Python UI)  
- **Risk Scoring:** Pure Python implementations of NEWS2, qSOFA, SIRS  
- **ML Profiling:** KMeans clustering + StandardScaler  
- **RAG Pipeline:**  
  - Sentence Transformers (`all-MiniLM-L6-v2`) for embeddings  
  - FAISS vector database for fast retrieval  
  - OpenAI GPT-3.5 for generation (grounded in guidelines)  
- **Data Sources:** WHO, Surviving Sepsis Campaign, CDC guideline PDFs  

---

## 📂 Project Structure

```bash
sepsis-rag-assistant/
├── app.py                # Streamlit app (main entry)
├── scoring.py            # NEWS2, qSOFA, SIRS scoring functions
├── rag_system.py         # RAG implementation (FAISS + OpenAI)
├── data_prep.py          # Preprocess guidelines → embeddings + FAISS
├── requirements.txt      # Dependencies
├── config/
│   └── settings.py       # Config loader (.env for OpenAI key)
├── data/
│   ├── guidelines/       # Raw guideline PDFs
│   └── processed/        # FAISS index + chunks/sources
└── README.md             # Project documentation
```
## How to clone this Repository? Just copy the command below

[git clone https://github.com/<your-username>/sepsis-rag-assistant.git](https://github.com/anshagarwxl/Sepsis-Early-Prediction)
