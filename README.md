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

⚡ Quickstart (Local Setup)
1️⃣ Clone Repository
git clone https://github.com/<your-username>/sepsis-rag-assistant.git
cd sepsis-rag-assistant

2️⃣ Create Virtual Environment & Install Dependencies
python3 -m venv venv
source venv/bin/activate      # For Mac/Linux
venv\Scripts\activate         # For Windows

pip install -r requirements.txt

3️⃣ Set Environment Variables

Create a .env file in the root folder and add:

OPENAI_API_KEY=your_openai_api_key_here

4️⃣ Preprocess Guidelines (Build FAISS Index)
python data_prep.py

5️⃣ Run the App
streamlit run app.py


The app will open in your browser at http://localhost:8501.

🎯 Key Features

Risk Scoring: Calculates NEWS2, qSOFA, and SIRS instantly.

Patient Profiling: Clusters patients into Low/Medium/High risk using ML.

RAG-Powered Q&A: Ask clinical questions, get AI-generated responses grounded in official guidelines with source links.

Lightweight & Fast: Runs locally with minimal setup, suitable for low-resource environments.

WIP: UI enhancements and cloud deployment are currently under development.

🖼 Example Use Case
Input (Vitals)	Output
Temp: 39°C, HR: 120, BP: 85/50, SpO₂: 90%, WBC: 18k	NEWS2 = High Risk (8), qSOFA = 2, SIRS = Positive → High Risk Alert
Q: "What is the first-line antibiotic for septic shock?"	AI: "According to the Surviving Sepsis Campaign, initiate broad-spectrum IV antibiotics within the first hour..." (with source citation)

🗺 Roadmap

 Implement risk scoring functions

 Build FAISS-based RAG pipeline

 Integrate OpenAI GPT for answer generation

 Add alert dashboard for multiple patients

 Enable offline mode with local LLM

 Deploy to cloud (Streamlit Community Cloud / Hugging Face Spaces)

👥 Team Members

This project is built with ❤️ by:

Ansh Agarwal 

Adithya S 

Kanishk Jaiswal  

Sayan Basu 


📜 License

This project is licensed under the MIT License — you are free to use, modify, and distribute with attribution.
