# Sepsis Early Warning RAG Assistant

A lightweight AI-powered assistant that combines clinical risk scoring and retrieval-augmented generation (RAG) to support frontline healthcare workers in the early detection and management of sepsis.

# Problem Statement

Sepsis is a life-threatening syndrome caused by the body’s extreme response to infection.

Every hour of delayed treatment increases mortality by 8%.

Globally, 1 in 3 patients with sepsis die.

In India, there is only 1 doctor per 1000 people, leading to overcrowding and diagnostic delays.

Manual tools like NEWS2, qSOFA, SIRS are rarely used in busy hospitals.

Result: Many patients slip into septic shock before getting timely care.

# Solution

The Sepsis Early Warning RAG Assistant is a Streamlit-based app that enables clinicians to:

Enter patient vitals (temperature, HR, BP, SpO₂, consciousness, WBC).

Calculate sepsis risk scores instantly: NEWS2, qSOFA, SIRS.

Cluster patients into risk profiles using a KMeans model.

Ask clinical questions → get guideline-grounded answers with citations (via RAG).

Workflow:
Vitals in ➝ Scores + Cluster ➝ Question ➝ Evidence-based recommendation.

# Tech Stack

Frontend: Streamlit (Python-based UI)

Risk Scoring: Pure Python implementations of NEWS2, qSOFA, SIRS

ML Profiling: KMeans clustering + StandardScaler

RAG Pipeline:

Sentence Transformers (all-MiniLM-L6-v2) for embeddings

FAISS vector database for fast retrieval

OpenAI GPT-3.5 for generation, grounded in guidelines

Data: Medical guideline PDFs (WHO, Surviving Sepsis Campaign, CDC)

# Project Structure
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

# Getting Started
1. Clone the repo
git clone https://github.com/<your-username>/sepsis-rag-assistant.git
cd sepsis-rag-assistant

2. Create virtual environment (Python 3.10 recommended)
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows

3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

4. Add API key

Create a .env file:

OPENAI_API_KEY=sk-xxxx

5. Prepare guideline data

Place PDF files in data/guidelines/ and run:

python data_prep.py

6. Run the app
streamlit run app.py --server.port 8501

# Features

Real-time calculation of NEWS2, qSOFA, SIRS

KMeans clustering for patient risk profiling

Evidence-based recommendations with citations from guidelines

Interactive Q&A with RAG assistant

Demo scenarios for High / Medium / Low risk

# Impact

Doctors save minutes → Patients gain hours.

Potential to save 100k–200k lives annually if deployed at scale.

Trusted, affordable, and globally scalable.

# Challenges Faced

Aligning FAISS, LangChain, OpenAI SDK with Python environment.

Preprocessing messy medical PDFs.

Balancing clinical accuracy with a working demo in 36 hours.

Making Streamlit lightweight enough for low-resource settings.

# Future Work

Expand to other conditions (pneumonia, dengue, trauma, obstetrics).

Mobile-first design for rural clinics.

EMR integration for real-time vitals.

Multilingual support.

Paid SaaS model with analytics + compliance for hospitals.

# Team 4Bytes

Ansh Agarwal[RA2411003011202]

Adithya S[RA2411003011210]

Sayan Basu[RA2411003011220]

Kanishk Jaiswal[RA2411003011180]
