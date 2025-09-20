# 🩺 Sepsis Early Warning RAG Assistant

A lightweight AI-powered clinical decision support tool that combines risk scoring and retrieval-augmented generation (RAG) to assist healthcare workers in the early detection and management of **sepsis**.

<br>

⚠️ **Status:** This is a **Work in Progress (WIP)** project and is in its final development phase. Most core features are functional, with deployment and UI refinements in progress.

---

## 🚨 Problem Statement

Sepsis is a life-threatening syndrome caused by the body’s extreme response to infection.

* Every hour of delayed treatment increases mortality by **~8%**.
* Globally, **1 in 3** patients with sepsis die.
* In India, there is only **1 doctor per 1000 people**, leading to overcrowding and diagnostic delays.
* Manual scoring tools (NEWS2, qSOFA, SIRS) are **underutilized** in busy hospitals, leading to late detection.

**Result:** Many patients slip into septic shock before timely care is provided, increasing mortality and ICU burden.

---

## 💡 Solution

The Sepsis Early Warning RAG Assistant provides an end-to-end digital solution for clinicians:

✅ **Enter patient vitals** (Temp, HR, BP, SpO₂, consciousness, WBC)  
✅ **Automatically calculate risk scores** — NEWS2, qSOFA, SIRS  
✅ **Cluster patients into risk categories** using ML (KMeans)  
✅ **Ask clinical questions** — get guideline-grounded answers with citations (via RAG)

**Workflow:** Vitals ➝ Risk Scores + Clustering ➝ Clinical Question ➝ Evidence-based Recommendation

This helps clinicians prioritize high-risk patients, reduce time to diagnosis, and follow evidence-based treatment guidelines.

---

## 🛠 Tech Stack

| Layer | Tools / Libraries |
| :--- | :--- |
| **Frontend** | Streamlit (Python-based UI) |
| **Risk Scoring** | Custom Python implementations of NEWS2, qSOFA, SIRS |
| **Machine Learning** | KMeans clustering + StandardScaler for patient profiling |
| **RAG Pipeline** | SentenceTransformers (all-MiniLM-L6-v2), FAISS vector DB, OpenAI GPT-3.5 |
| **Data Sources** | WHO, Surviving Sepsis Campaign, CDC Guidelines |
| **Config & Secrets** | .env file for API keys and configs |

---

## 📂 Project Structure

To fix the project structure image to a proper and readable format, you need to use a tree-like hierarchy that's easily understood and commonly used in README files. The provided image shows a flattened, linear structure that's difficult to parse.

## 📂 Corrected Project Structure

Here is the corrected, well-formatted project structure that you can copy and paste into your GitHub README file. It uses proper indentation to show the hierarchy of files and folders.

```
sepsis-rag-assistant/
├── app.py                # Main Streamlit app entry point
├── scoring.py            # NEWS2, qSOFA, SIRS scoring functions
├── rag_system.py         # RAG implementation (FAISS + OpenAI)
├── data_prep.py          # Preprocess guidelines → embeddings + FAISS index
├── requirements.txt      # Dependencies
├── config/
│   └── settings.py       # Config loader & OpenAI key handling
├── data/
│   ├── guidelines/       # Raw guideline PDFs
│   └── processed/        # FAISS index, chunks, and metadata
└── README.md             # Project documentation
```

---
## 👥 Team Members

<p align="center">
    <span>Ansh Agarwal</span> &nbsp; &bull; &nbsp; <span>Adithya S</span> &nbsp; &bull; &nbsp; <span>Kanishk Jaiswal</span> &nbsp; &bull; &nbsp; <span>Sayan Basu</span>
</p>


📜 License

This project is licensed under the MIT License — you are free to use, modify, and distribute with attribution.
