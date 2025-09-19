import os, pickle
import streamlit as st
from scoring import (
    calculate_news2, calculate_qsofa, calculate_sirs, interpret_risk
)
from rag_system import SepsisRAG

st.set_page_config(page_title="Sepsis Early Warning RAG Assistant", layout="wide")

# ---------- Cached loaders ----------
@st.cache_resource
def load_rag_system():
    return SepsisRAG()  # loads FAISS, chunks, model, and OpenAI key via config .env

@st.cache_resource
def load_kmeans_and_scaler():
    try:
        with open("models/kmeans_model.pkl", "rb") as f:
            kmeans = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return kmeans, scaler
    except Exception:
        return None, None

def predict_cluster(vitals, kmeans, scaler):
    """Return cluster label or 'N/A' if models unavailable."""
    if not kmeans or not scaler:
        return "N/A"
    # Order matters: use the exact feature order you trained on
    features = [
        vitals["temperature"],
        vitals["heart_rate"],
        vitals["respiratory_rate"],
        vitals["systolic_bp"],
        vitals["spo2"],
        # Optional: WBC if present in training; else omit
        vitals.get("wbc", 0)
    ]
    X = scaler.transform([features])
    return int(kmeans.predict(X)[0])

# ---------- App ----------
st.title("🩺 Sepsis Early Warning RAG Assistant")
st.caption("Demo/education only — not a medical device.")

# Load systems
rag = None
with st.spinner("Loading RAG system..."):
    try:
        rag = load_rag_system()
        rag_ok = True
    except Exception as e:
        rag_ok = False
        st.error(f"RAG unavailable: {e}. Run data preparation first.")

kmeans, scaler = load_kmeans_and_scaler()

# Sidebar inputs
st.sidebar.header("Patient Vitals")
vitals = {
    "temperature": st.sidebar.number_input("Temperature (°C)", 30.0, 45.0, 37.0, 0.1),
    "heart_rate": st.sidebar.number_input("Heart Rate (bpm)", 30, 200, 80, 1),
    "respiratory_rate": st.sidebar.number_input("Respiratory Rate (/min)", 5, 50, 16, 1),
    "systolic_bp": st.sidebar.number_input("Systolic BP (mmHg)", 60, 250, 120, 1),
    "spo2": st.sidebar.number_input("SpO₂ (%)", 70, 100, 98, 1),
    "consciousness": st.sidebar.selectbox("Consciousness (AVPU)", ["Alert", "Voice", "Pain", "Unresponsive"]),
    "wbc": st.sidebar.number_input("WBC (/μL) (optional)", 0, 50000, 0, 100),
}

# Optional demo presets
st.sidebar.subheader("Demo Scenarios")
col_demo1, col_demo2, col_demo3 = st.sidebar.columns(3)
if col_demo1.button("High"):
    vitals.update({"temperature": 39.2, "heart_rate": 125, "respiratory_rate": 28, "systolic_bp": 88, "spo2": 89, "consciousness": "Voice"})
if col_demo2.button("Med"):
    vitals.update({"temperature": 38.5, "heart_rate": 105, "respiratory_rate": 22, "systolic_bp": 95, "spo2": 94, "consciousness": "Alert"})
if col_demo3.button("Low"):
    vitals.update({"temperature": 37.1, "heart_rate": 85, "respiratory_rate": 18, "systolic_bp": 115, "spo2": 97, "consciousness": "Alert"})

# Validation (minimal)
errors = []
if vitals["temperature"] < 30 or vitals["temperature"] > 45: errors.append("Temperature out of range")
if vitals["heart_rate"] < 30 or vitals["heart_rate"] > 200: errors.append("Heart rate out of range")
if errors:
    st.sidebar.error(" • ".join(errors))

# Compute
if st.sidebar.button("⚙️  Calculate Risk Scores", disabled=bool(errors)):
    news2_score, news2_risk = calculate_news2(vitals)
    qsofa_score, qsofa_risk = calculate_qsofa(vitals)
    sirs_score, sirs_risk = calculate_sirs(vitals)
    cluster = predict_cluster(vitals, kmeans, scaler)

    st.session_state["scores"] = {
        "news2": (news2_score, news2_risk),
        "qsofa": (qsofa_score, qsofa_risk),
        "sirs": (sirs_score, sirs_risk),
        "cluster": cluster,
    }

# Results area
scores = st.session_state.get("scores")
if scores:
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        st.metric("NEWS2", scores["news2"][0])
        st.write(f"**{scores['news2'][1]}**")
    with c2:
        st.metric("qSOFA", scores["qsofa"][0])
        st.write(f"**{scores['qsofa'][1]}**")
    with c3:
        st.metric("SIRS", scores["sirs"][0])
        st.write(f"**{scores['sirs'][1]}**")
    with c4:
        st.metric("Profile", scores["cluster"])
        st.caption("K-Means cluster")

    st.subheader("Clinical Interpretation")
    for msg in interpret_risk(scores["news2"][0], scores["qsofa"][0], scores["sirs"][0]):
        st.warning(msg)

# RAG chat
if rag:
    st.subheader("🧠 Clinical Guidance Assistant")
    cc1, cc2, cc3 = st.columns(3)
    if cc1.button("💊 Antibiotics"): st.session_state["query"] = "What are recommended empiric antibiotics for suspected sepsis?"
    if cc2.button("💧 Fluid Resuscitation"): st.session_state["query"] = "How much crystalloid should I give initially?"
    if cc3.button("⚡ Emergency Actions"): st.session_state["query"] = "Immediate steps for high sepsis risk?"

    user_query = st.text_input("Ask a clinical question", value=st.session_state.get("query", ""))
    if user_query:
        with st.spinner("Searching clinical guidelines..."):
            patient_scores = st.session_state.get("scores", None)
            answer, sources = rag.query(user_query, patient_scores=patient_scores)
        st.subheader("Answer")
        st.write(answer)
        with st.expander("Sources & Guidelines"):
            for i, s in enumerate(sorted(set(sources)), 1):
                st.write(f"{i}. {s}")
rag = SepsisRAG()
question = st.text_input("Ask a sepsis-related question:")
if question:
    answer, sources = rag.query(question)
    st.write(answer)
    st.write("Sources:", sources)
