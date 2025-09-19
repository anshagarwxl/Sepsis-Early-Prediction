# 🩺 Sepsis Early Warning RAG Assistant

A lightweight AI-powered assistant that combines **clinical risk scoring** and **retrieval-augmented generation (RAG)** to support frontline healthcare workers in the **early detection and management of sepsis**.

---

## 🚨 Problem Statement

- Sepsis is a **life-threatening syndrome** caused by the body’s extreme response to infection.  
- Every **hour of delayed treatment increases mortality by ~8%**.  
- Globally, **1 in 3 patients with sepsis die**.  
- In India, there is **only 1 doctor per 1000 people**, leading to overcrowding and diagnostic delays.  
- Manual tools (NEWS2, qSOFA, SIRS) are rarely used in busy hospitals.  

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
- **Data:** WHO, Surviving Sepsis Campaign, CDC guideline PDFs  

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

🚀 Getting Started

Follow these steps to set up and run the project locally:

1. Clone the repository
git clone https://github.com/<your-username>/sepsis-rag-assistant.git
cd sepsis-rag-assistant

2. Create a virtual environment (Python 3.10 recommended)
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows

3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

4. Add your OpenAI API key

Create a .env file in the project root:

OPENAI_API_KEY=sk-xxxx

5. Prepare guideline data

Place guideline PDFs in data/guidelines/ and run:

python data_prep.py

6. Run the app
streamlit run app.py --server.port 8501

✨ Features

⚡ Real-time calculation of NEWS2, qSOFA, SIRS

📊 KMeans clustering for patient risk profiling

📚 Evidence-based recommendations with citations from medical guidelines

💬 Interactive Q&A with RAG assistant

🧪 Demo scenarios for High / Medium / Low risk patients

🌍 Impact

Doctors save minutes → Patients gain hours.

Potential to save 100k–200k lives annually if deployed at scale.

Trusted, affordable, and globally scalable.

🧩 Challenges

Aligning FAISS, LangChain, OpenAI SDK in Python environments.

Preprocessing messy medical PDFs.

Balancing clinical accuracy with a working demo in 36 hours.

Keeping Streamlit lightweight for low-resource hospitals.

🔮 Future Work

Expand to other conditions: pneumonia, dengue, trauma, obstetrics.

Mobile-first design for rural clinics.

EMR integration for real-time vitals.

Multilingual support for non-English healthcare settings.

Paid SaaS model with analytics + compliance for hospitals.

👨‍💻 Team 4Bytes

Ansh Agarwal 

Adithya S 

Sayan Basu

Kanishk Jaiswal 
