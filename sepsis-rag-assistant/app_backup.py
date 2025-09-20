# app.py
# Sepsis RAG Assistant — Dark Glassmorphism PRO with RAG Integration
# Advanced medical AI assistant with seamless RAG system integration
# Features: Real-time scoring, ML predictions, RAG chat, data analytics

from __future__ import annotations

import os
import io
import json
import math
import time
import pickle
import numbers
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ------------------------- Page & Theme ------------------------- #
st.set_page_config(
    page_title="Sepsis RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced Dark Glassmorphism CSS
GLASS_CSS = """
<style>
:root{
  --bg-a: #0a0f1b;
  --bg-b: #0b1220;
  --bg-c: #0e1530;
  --panel: rgba(255,255,255,.06);
  --panel-2: rgba(255,255,255,.08);
  --panel-3: rgba(255,255,255,.05);
  --border: rgba(255,255,255,.12);
  --ink: rgba(255,255,255,.98);
  --muted: rgba(255,255,255,.70);
  --green: #00d1a7;
  --amber: #ffcc66;
  --red: #ff6b6b;
  --blue: #7fd3ff;
  --violet: #a78bfa;
  --pink: #ff9bd2;
}

html, body, .stApp {
  background: radial-gradient(80% 140% at 10% 10%, var(--bg-a) 0%, var(--bg-b) 55%, var(--bg-c) 100%);
  color: var(--ink);
}

[data-testid="stSidebar"] > div {
  background: var(--panel);
  backdrop-filter: blur(12px) saturate(140%);
  border-right: 1px solid var(--border);
}

.main .block-container {
  padding-top: 0.8rem;
  max-width: 1200px;
}

.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px;
  backdrop-filter: blur(14px) saturate(160%);
  box-shadow: 0 12px 40px rgba(0,0,0,.35);
}

.tile {
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
}

.soft {
  background: var(--panel-3);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 14px;
}

.hero{
  position: relative; overflow:hidden;
  background: linear-gradient(145deg, rgba(8,12,28,.6), rgba(30,35,60,.35));
  border:1px solid var(--border); border-radius:24px; padding:26px 22px;
}
.hero h1{ margin:0 0 8px; font-size:34px; }
.hero p{ margin:0; color:var(--muted); }

.kpi {
  font-weight: 800;
  font-size: 26px;
}
.kpi-sub {
  color: var(--muted);
  font-size: 12px;
  margin-top: 4px;
}

.hr { height:1px; background:var(--border); margin: 10px 0 16px 0; }

.pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-weight: 800;
  letter-spacing: .2px;
  font-size: 12px;
}
.pill.low  { background: rgba(0,209,167,.12); color: var(--green); }
.pill.med  { background: rgba(255,204,102,.14); color: var(--amber); }
.pill.high { background: rgba(255,107,107,.14); color: var(--red); }
.pill.ml   { background: rgba(127,211,255,.14); color: var(--blue); }

.cta{
  display:flex; gap:16px; flex-wrap:wrap; margin-top:18px;
}
.cta .tile{
  flex: 1 1 300px;
  padding:16px; border-radius:16px; border:1px solid var(--border);
  background: linear-gradient(180deg, rgba(25,32,60,.45), rgba(18,24,45,.35));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 12px 26px rgba(0,0,0,.30);
}
.cta .title{ font-weight:900; font-size:18px; margin-bottom:6px; }
.cta .sub{ color:var(--muted); font-size:.95rem; }

h1, h2, h3 { color: var(--ink); }
small, .muted { color: var(--muted); }

.stButton>button {
  border-radius: 12px;
  border: 1px solid var(--border);
  background: linear-gradient(135deg, var(--blue), var(--violet));
  color: #06131f !important;
  font-weight: 900;
}
.stButton>button:hover{ transform: translateY(-1px); box-shadow:0 12px 28px rgba(0,0,0,.30); }

.stTabs [role="tablist"] { gap:8px; }
.stTabs [role="tab"]{
  padding:6px 10px;
  background:rgba(255,255,255,.04);
  border:1px solid var(--border) !important;
  border-radius:10px !important;
}
.stTabs [aria-selected="true"]{
  background:#16203a !important; border-color:#223055 !important; color:#eaf0ff !important; font-weight:800;
}

a, a:visited { color: var(--blue) !important; text-decoration: none; }
</style>
"""
st.markdown(GLASS_CSS, unsafe_allow_html=True)

# Animated Hero Component
HERO = """
<div style="position:relative;border-radius:24px;overflow:hidden;">
  <div style="
    position:absolute; inset:0; z-index:0; opacity:.45; pointer-events:none;
    background:
      radial-gradient(60% 60% at 10% 10%, rgba(127,211,255,.20) 0%, rgba(127,211,255,0) 60%),
      radial-gradient(50% 50% at 90% 20%, rgba(167,139,250,.18) 0%, rgba(167,139,250,0) 60%),
      radial-gradient(40% 40% at 70% 80%, rgba(255,155,210,.16) 0%, rgba(255,155,210,0) 60%);
    filter: blur(24px);
  "></div>
  <div class="card" style="position:relative; z-index:1; border-radius:24px; padding:24px;">
    <div style="display:flex; align-items:center; gap:18px;">
      <div style="font-size:28px;">🩺</div>
      <div>
        <div style="font-weight:900; letter-spacing:.4px; font-size:26px;">Sepsis RAG Assistant</div>
        <div class="muted" style="margin-top:2px;">AI-Powered Clinical Decision Support • RAG • ML • Analytics</div>
      </div>
      <div style="margin-left:auto;">
        <span class="pill ml">Live Session</span>
      </div>
    </div>
  </div>
</div>
"""

# tiny SVG background (doesn't push content)
components.html("<div style='position:fixed;inset:0;z-index:-1;pointer-events:none'></div>", height=1)

PLOTLY_TEMPLATE = "plotly_dark"

# ------------------------- RAG System Integration ------------------------- #
try:
    from rag_system import SepsisRAG
    RAG_AVAILABLE = True
except Exception as e:
    SepsisRAG = None
    RAG_AVAILABLE = False

# Configure Gemini API
try:
    from config.settings import GEMINI_API_KEY
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Ensure API key is available for RAG system
if not GEMINI_API_KEY:
    # Try one more time to get from environment
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ------------------------- ML Model Integration ------------------------- #
try:
    import joblib
    ML_AVAILABLE = True
except Exception:
    joblib = None
    ML_AVAILABLE = False

# Paths for optional ML pipeline
MODEL_PATH = Path("models/knn_sepsis_model.joblib")
META_PATH = Path("models/meta.json")

# ------------------------- Session State & Schema ------------------------- #
SCHEMA = [
    "Timestamp","Temperature","HeartRate","RespiratoryRate",
    "SystolicBP","DiastolicBP","SpO2","WBC","GCS",
    "Consciousness","Notes",
    "NEWS2","NEWS2_Risk","qSOFA","qSOFA_Risk","SIRS","SIRS_Risk",
    "ML_Prediction","ML_Prob","ML_Risk"
]

def init_state():
    """Initialize session state with all required variables."""
    if "data" not in st.session_state:
        st.session_state["data"] = pd.DataFrame(columns=SCHEMA)
    
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    
    if "vitals" not in st.session_state:
        st.session_state["vitals"] = {
            "temperature": 37.0, "heart_rate": 80, "respiratory_rate": 16,
            "systolic_bp": 120, "diastolic_bp": 80, "spo2": 98, 
            "consciousness": "Alert", "wbc": 0, "gcs": 15
        }
    
    if "scores" not in st.session_state:
        st.session_state["scores"] = None
    
    if "timeline" not in st.session_state:
        st.session_state["timeline"] = []

init_state()

# ------------------------- Core System Loaders (Eager Loading) ------------------------- #
@st.cache_resource(show_spinner=False)
def load_rag_system():
    """Load RAG system with eager initialization."""
    if not RAG_AVAILABLE:
        return None
    try:
        if GEMINI_API_KEY:
            rag = SepsisRAG(gemini_api_key=GEMINI_API_KEY)
            return rag
        else:
            st.sidebar.warning("⚠️ GEMINI_API_KEY not found. RAG system disabled.")
            return None
    except Exception as e:
        st.sidebar.error(f"RAG system failed to load: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_ml_model():
    """Load ML model if available."""
    if not ML_AVAILABLE or not MODEL_PATH.exists() or not META_PATH.exists():
        return None, {}
    try:
        model = joblib.load(MODEL_PATH)
        meta = json.loads(META_PATH.read_text())
        return model, meta
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load ML model: {e}")
        return None, {}

# Initialize systems eagerly
RAG_SYSTEM = load_rag_system()
ML_MODEL, ML_META = load_ml_model()

# ------------------------- Clinical Scoring Functions ------------------------- #
def news2_score(temp: float, hr: float, rr: float, sbp: float, spo2: float, cons: str) -> Tuple[int, str]:
    """Calculate NEWS2 score and risk level."""
    s = 0
    # Respiratory Rate
    if rr <= 8: s += 3
    elif 9 <= rr <= 11: s += 1
    elif 12 <= rr <= 20: s += 0
    elif 21 <= rr <= 24: s += 2
    else: s += 3
    
    # SpO2 (scale 1)
    if spo2 <= 91: s += 3
    elif 92 <= spo2 <= 93: s += 2
    elif 94 <= spo2 <= 95: s += 1
    
    # Temperature
    if temp <= 35.0: s += 3
    elif 35.1 <= temp <= 36.0: s += 1
    elif 36.1 <= temp <= 38.0: s += 0
    elif 38.1 <= temp <= 39.0: s += 1
    else: s += 2
    
    # Systolic BP
    if sbp <= 90: s += 3
    elif 91 <= sbp <= 100: s += 2
    elif 101 <= sbp <= 110: s += 1
    
    # Heart Rate
    if hr <= 40 or hr >= 131: s += 3
    elif 111 <= hr <= 130: s += 2
    elif 91 <= hr <= 110: s += 1
    
    # Consciousness
    if cons.lower() != "alert": s += 3
    
    # Risk assessment
    if s >= 7:
        risk = "High"
    elif s >= 5:
        risk = "Medium"
    else:
        # Check for red flag criteria
        red_flag = any([
            rr <= 8 or rr >= 25,
            spo2 <= 91,
            temp <= 35.0 or temp >= 39.1,
            sbp <= 90,
            hr <= 40 or hr >= 131,
            cons.lower() != "alert",
        ])
        risk = "Medium" if red_flag else "Low"
    
    return s, risk

def qsofa_score(rr: float, sbp: float, gcs: Optional[float], cons: str) -> Tuple[int, str]:
    """Calculate qSOFA score and risk level."""
    s = 0
    if rr >= 22: s += 1
    if sbp <= 100: s += 1
    
    # Altered mental status
    altered = (gcs is not None and gcs < 15) or (cons.lower() != "alert")
    if altered: s += 1
    
    risk = "High" if s >= 2 else ("Medium" if s == 1 else "Low")
    return s, risk

def sirs_score(temp: float, hr: float, rr: float, wbc: Optional[float]) -> Tuple[int, str]:
    """Calculate SIRS score and risk level."""
    c = 0
    if temp > 38.0 or temp < 36.0: c += 1
    if hr > 90: c += 1
    if rr > 20: c += 1
    if wbc is not None and (wbc > 12 or wbc < 4): c += 1
    
    risk = "High" if c >= 3 else ("Medium" if c == 2 else "Low")
    return c, risk

def interpret_risk_level(news2: int, qsofa: int, sirs: int) -> Tuple[str, str, str]:
    """Interpret overall risk based on scores."""
    if news2 >= 7 or qsofa >= 2:
        return ("High", "#ff6b6b", "Urgent clinical escalation required. Consider sepsis bundle.")
    elif news2 >= 5:
        return ("Medium", "#ffcc66", "Urgent review by clinician. Monitor closely.")
    elif news2 >= 1 or sirs >= 2:
        return ("Low-Medium", "#7fd3ff", "Requires assessment and monitoring.")
    else:
        return ("Low", "#00d1a7", "Continue routine monitoring")

def compute_all_scores(vitals: Dict[str, Any]) -> Dict[str, Any]:
    """Compute all clinical scores for given vitals."""
    temp = float(vitals.get("temperature", 37.0))
    hr = float(vitals.get("heart_rate", 80))
    rr = float(vitals.get("respiratory_rate", 16))
    sbp = float(vitals.get("systolic_bp", 120))
    dbp = float(vitals.get("diastolic_bp", 80))
    spo2 = float(vitals.get("spo2", 98))
    wbc = vitals.get("wbc", None)
    wbc = float(wbc) if wbc not in (None, "", np.nan) else None
    gcs = vitals.get("gcs", None)
    gcs = float(gcs) if gcs not in (None, "", np.nan) else None
    cons = str(vitals.get("consciousness", "Alert"))
    
    # Calculate clinical scores
    n2, n2r = news2_score(temp, hr, rr, sbp, spo2, cons)
    qf, qfr = qsofa_score(rr, sbp, gcs, cons)
    si, sir = sirs_score(temp, hr, rr, wbc)
    
    # ML prediction if available
    ml_pred, ml_prob, ml_risk = None, None, None
    if ML_MODEL is not None:
        try:
            # Feature preparation for ML model
            features = ["Temperature", "HeartRate", "RespiratoryRate", "SystolicBP", "DiastolicBP", "SpO2", "WBC", "GCS"]
            row_data = {
                "Temperature": temp,
                "HeartRate": hr, 
                "RespiratoryRate": rr,
                "SystolicBP": sbp,
                "DiastolicBP": dbp,
                "SpO2": spo2,
                "WBC": wbc if wbc is not None else 8.0,  # Default WBC
                "GCS": gcs if gcs is not None else 15   # Default GCS
            }
            
            X = pd.DataFrame([row_data], columns=features)
            y = ML_MODEL.predict(X)[0]
            ml_pred = int(y) if isinstance(y, numbers.Number) else y
            
            if hasattr(ML_MODEL, "predict_proba"):
                proba = ML_MODEL.predict_proba(X)
                ml_prob = float(proba[0][1]) if proba.shape[1] == 2 else float(np.max(proba[0]))
            
            ml_risk = "High" if ml_pred == 1 else "Low"
        except Exception as e:
            st.sidebar.warning(f"ML inference failed: {e}")
    
    return {
        "NEWS2": n2, "NEWS2_Risk": n2r,
        "qSOFA": qf, "qSOFA_Risk": qfr, 
        "SIRS": si, "SIRS_Risk": sir,
        "ML_Prediction": ml_pred, "ML_Prob": ml_prob, "ML_Risk": ml_risk
    }

# ------------------------- UI Helper Functions ------------------------- #
def pill(text: str) -> str:
    """Create colored risk pill."""
    key = text.lower()
    cls = "low" if key == "low" else "med" if key == "medium" else "high" if key == "high" else "ml"
    return f'<span class="pill {cls}">{text}</span>'

def kpi_tile(title: str, value: str, subtitle: str = ""):
    """Create KPI tile component."""
    st.markdown(
        f'<div class="tile"><div class="kpi">{value}</div>'
        f'<div class="kpi-sub">{title}</div>'
        f'<div class="kpi-sub">{subtitle}</div></div>', 
        unsafe_allow_html=True
    )

def gauge_svg(label: str, value: float, max_val: float, color: str, size: int = 170):
    """Create circular gauge SVG."""
    pct = max(0.0, min(value / max_val, 1.0))
    angle = pct * 270.0
    r = 64
    cx = size // 2
    cy = size // 2 + 12
    start = -135
    end = start + angle
    sx = cx + r * math.cos(math.radians(start))
    sy = cy + r * math.sin(math.radians(start))
    ex = cx + r * math.cos(math.radians(end))
    ey = cy + r * math.sin(math.radians(end))
    large = 1 if angle > 180 else 0
    
    svg = f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <defs>
        <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="{color}" stop-opacity="0.9"/>
          <stop offset="100%" stop-color="#ffffff" stop-opacity="0.25"/>
        </linearGradient>
      </defs>
      <circle cx="{cx}" cy="{cy}" r="{r}" stroke="rgba(255,255,255,.08)" stroke-width="16" fill="none"/>
      <path d="M {sx:.2f} {sy:.2f} A {r} {r} 0 {large} 1 {ex:.2f} {ey:.2f}"
            stroke="url(#g1)" stroke-width="16" fill="none" stroke-linecap="round"/>
      <text x="{cx}" y="{cy-4}" text-anchor="middle" fill="white" font-size="30" font-weight="900">{int(value)}</text>
      <text x="{cx}" y="{cy+26}" text-anchor="middle" fill="rgba(255,255,255,.7)" font-size="13">{label}</text>
    </svg>
    """
    return svg

# ------------------------- Sidebar Status ------------------------- #
with st.sidebar:
    st.markdown("### System Status")
    st.markdown('<div class="soft">Real-time clinical decision support system</div>', unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    
    # RAG System Status
    if RAG_SYSTEM is not None:
        st.success("✅ RAG System: Active")
        st.caption("Gemini AI • Medical Knowledge Base")
    else:
        st.error("❌ RAG System: Offline")
        if not GEMINI_API_KEY:
            st.caption("Missing GEMINI_API_KEY")
        else:
            st.caption("System initialization failed")
    
    # ML Model Status  
    if ML_MODEL is not None:
        st.success("✅ ML Model: Loaded")
        st.caption("Predictive Risk Assessment")
    else:
        st.warning("⚠️ ML Model: Not Available")
        st.caption("Basic scoring only")
    
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    
    # Data Summary
    total_records = len(st.session_state["data"])
    st.metric("Records", total_records)
    
    if total_records > 0:
        last_score = st.session_state.get("scores")
        if last_score:
            st.metric("Latest NEWS2", last_score["NEWS2"])
        
        high_risk = (st.session_state["data"]["NEWS2_Risk"] == "High").sum()
        st.metric("High Risk", f"{high_risk}/{total_records}")

# ------------------------- Main Application ------------------------- #
# Hero Section
st.markdown(HERO, unsafe_allow_html=True)
st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# Main Tabs
tab_overview, tab_patient, tab_analytics, tab_assistant, tab_upload = st.tabs([
    "🏠 Overview", "🩺 Patient Entry", "📊 Analytics", "💬 AI Assistant", "📂 Data Upload"
])

# === OVERVIEW TAB ===
with tab_overview:
    st.markdown("#### Clinical Dashboard")
    
    df = st.session_state["data"]
    total = len(df)
    
    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_tile("Total Records", f"{total}")
    with col2:
        if total > 0:
            high_news = (df["NEWS2_Risk"] == "High").sum()
            kpi_tile("High NEWS2", f"{high_news}", f"{(100*high_news/total):.1f}%")
        else:
            kpi_tile("High NEWS2", "0", "0.0%")
    with col3:
        if total > 0 and "ML_Risk" in df.columns and df["ML_Risk"].notna().any():
            high_ml = (df["ML_Risk"] == "High").sum()
            kpi_tile("ML High Risk", f"{high_ml}", f"{(100*high_ml/total):.1f}%")
        else:
            kpi_tile("ML High Risk", "—", "Model Required")
    with col4:
        if total > 0:
            core_fields = ["Temperature", "HeartRate", "RespiratoryRate", "SystolicBP", "SpO2"]
            completeness = np.mean([df[c].notna().mean() if c in df else 0 for c in core_fields]) * 100
            kpi_tile("Completeness", f"{completeness:.0f}%", "Data Quality")
        else:
            kpi_tile("Completeness", "0%", "No Data")
    
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    
    # Latest Assessment Card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Latest Patient Assessment**")
    
    if total == 0:
        st.info("No patient data recorded yet. Use the **Patient Entry** tab to add vital signs and clinical data.")
    else:
        latest = df.iloc[-1]
        
        # Risk Gauges
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            color = "#ff6b6b" if latest["NEWS2_Risk"] == "High" else "#ffcc66" if latest["NEWS2_Risk"] == "Medium" else "#00d1a7"
            st.markdown(gauge_svg("NEWS2", float(latest["NEWS2"]), 20, color), unsafe_allow_html=True)
        
        with col_b:
            color = "#ff6b6b" if latest["qSOFA_Risk"] == "High" else "#ffcc66" if latest["qSOFA_Risk"] == "Medium" else "#00d1a7"
            st.markdown(gauge_svg("qSOFA", float(latest["qSOFA"]), 3, color), unsafe_allow_html=True)
        
        with col_c:
            color = "#ff6b6b" if latest["SIRS_Risk"] == "High" else "#ffcc66" if latest["SIRS_Risk"] == "Medium" else "#00d1a7"
            st.markdown(gauge_svg("SIRS", float(latest["SIRS"]), 4, color), unsafe_allow_html=True)
        
        with col_d:
            if pd.notna(latest["ML_Risk"]):
                color = "#ff6b6b" if latest["ML_Risk"] == "High" else "#7fd3ff"
                score = 100 * float(latest["ML_Prob"]) if pd.notna(latest["ML_Prob"]) else 50
                st.markdown(gauge_svg("ML Risk%", score, 100, color), unsafe_allow_html=True)
            else:
                st.markdown('<div class="soft">ML model not available</div>', unsafe_allow_html=True)
        
        # Risk Summary
        ml_text = ""
        if pd.notna(latest["ML_Risk"]):
            prob_text = f" (p={latest['ML_Prob']:.2f})" if pd.notna(latest["ML_Prob"]) else ""
            ml_text = f" &nbsp; ML: {pill(latest['ML_Risk'])}{prob_text}"
        
        st.markdown(
            f"**Risk Assessment:** NEWS2: **{int(latest['NEWS2'])}** {pill(latest['NEWS2_Risk'])} &nbsp; "
            f"qSOFA: **{int(latest['qSOFA'])}** {pill(latest['qSOFA_Risk'])} &nbsp; "
            f"SIRS: **{int(latest['SIRS'])}** {pill(latest['SIRS_Risk'])}{ml_text}",
            unsafe_allow_html=True
        )
        
        # Timestamp
        st.caption(f"Recorded: {latest['Timestamp']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# === PATIENT ENTRY TAB ===
with tab_patient:
    st.markdown("#### Patient Vital Signs Entry")
    
    # Quick presets for testing
    st.markdown('<div class="soft">Quick Presets for Testing</div>', unsafe_allow_html=True)
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    
    with preset_col1:
        if st.button("🟢 Normal Vitals"):
            st.session_state.vitals.update({
                "temperature": 37.0, "heart_rate": 75, "respiratory_rate": 16,
                "systolic_bp": 120, "diastolic_bp": 80, "spo2": 98,
                "consciousness": "Alert", "wbc": 7.5, "gcs": 15
            })
    
    with preset_col2:
        if st.button("🟡 Moderate Risk"):
            st.session_state.vitals.update({
                "temperature": 38.5, "heart_rate": 105, "respiratory_rate": 22,
                "systolic_bp": 95, "diastolic_bp": 65, "spo2": 94,
                "consciousness": "Alert", "wbc": 13.2, "gcs": 14
            })
    
    with preset_col3:
        if st.button("🔴 High Risk"):
            st.session_state.vitals.update({
                "temperature": 39.2, "heart_rate": 125, "respiratory_rate": 28,
                "systolic_bp": 88, "diastolic_bp": 55, "spo2": 89,
                "consciousness": "Voice", "wbc": 15.8, "gcs": 12
            })
    
    with preset_col4:
        if st.button("🧹 Clear Form"):
            st.session_state.vitals = {
                "temperature": 37.0, "heart_rate": 80, "respiratory_rate": 16,
                "systolic_bp": 120, "diastolic_bp": 80, "spo2": 98,
                "consciousness": "Alert", "wbc": 0, "gcs": 15
            }
    
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    
    # Vitals Entry Form
    with st.form("vitals_form", clear_on_submit=False):
        st.markdown("**Enter Patient Vital Signs**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Primary Vitals**")
            temperature = st.number_input(
                "Temperature (°C)", 30.0, 45.0, 
                st.session_state.vitals.get("temperature", 37.0), 
                step=0.1, help="Normal: 36.1-37.2°C"
            )
            heart_rate = st.number_input(
                "Heart Rate (bpm)", 30, 250, 
                int(st.session_state.vitals.get("heart_rate", 80)), 
                step=1, help="Normal: 60-100 bpm"
            )
            respiratory_rate = st.number_input(
                "Respiratory Rate (/min)", 5, 60, 
                int(st.session_state.vitals.get("respiratory_rate", 16)), 
                step=1, help="Normal: 12-20 /min"
            )
        
        with col2:
            st.markdown("**Blood Pressure & Oxygen**")
            systolic_bp = st.number_input(
                "Systolic BP (mmHg)", 60, 300, 
                int(st.session_state.vitals.get("systolic_bp", 120)), 
                step=1, help="Normal: 100-130 mmHg"
            )
            diastolic_bp = st.number_input(
                "Diastolic BP (mmHg)", 40, 200, 
                int(st.session_state.vitals.get("diastolic_bp", 80)), 
                step=1, help="Normal: 60-90 mmHg"
            )
            spo2 = st.number_input(
                "SpO₂ (%)", 70, 100, 
                int(st.session_state.vitals.get("spo2", 98)), 
                step=1, help="Normal: >95%"
            )
        
        with col3:
            st.markdown("**Additional Parameters**")
            wbc = st.number_input(
                "WBC (×10³/µL)", 0.0, 50.0, 
                float(st.session_state.vitals.get("wbc", 8.0)), 
                step=0.1, help="Normal: 4-12 ×10³/µL"
            )
            gcs = st.number_input(
                "Glasgow Coma Scale", 3, 15, 
                int(st.session_state.vitals.get("gcs", 15)), 
                step=1, help="Normal: 15"
            )
            consciousness = st.selectbox(
                "Consciousness (AVPU)", 
                ["Alert", "Voice", "Pain", "Unresponsive"],
                index=["Alert", "Voice", "Pain", "Unresponsive"].index(
                    st.session_state.vitals.get("consciousness", "Alert")
                )
            )
        
        notes = st.text_area(
            "Clinical Notes (optional)", 
            value="", 
            height=80, 
            placeholder="Additional observations, symptoms, or clinical context..."
        )
        
        submitted = st.form_submit_button("💾 Save & Calculate Scores", use_container_width=True)
    
    if submitted:
        # Validation
        errors = []
        if temperature < 30 or temperature > 45:
            errors.append("Temperature out of valid range (30-45°C)")
        if heart_rate <= 0:
            errors.append("Heart rate must be positive")
        if respiratory_rate <= 0:
            errors.append("Respiratory rate must be positive")
        if systolic_bp <= 0:
            errors.append("Systolic BP must be positive")
        if spo2 < 0 or spo2 > 100:
            errors.append("SpO₂ must be between 0-100%")
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            # Create record
            record = {
                "Timestamp": dt.datetime.now(),
                "Temperature": float(temperature),
                "HeartRate": float(heart_rate),
                "RespiratoryRate": float(respiratory_rate),
                "SystolicBP": float(systolic_bp),
                "DiastolicBP": float(diastolic_bp),
                "SpO2": float(spo2),
                "WBC": float(wbc) if wbc > 0 else None,
                "GCS": float(gcs),
                "Consciousness": str(consciousness),
                "Notes": str(notes).strip()
            }
            
            # Compute scores
            vitals_for_scoring = {
                "temperature": temperature,
                "heart_rate": heart_rate,
                "respiratory_rate": respiratory_rate,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "spo2": spo2,
                "wbc": wbc if wbc > 0 else None,
                "gcs": gcs,
                "consciousness": consciousness
            }
            
            scores = compute_all_scores(vitals_for_scoring)
            record.update(scores)
            
            # Add to session data
            new_df = pd.DataFrame([record], columns=SCHEMA)
            st.session_state["data"] = pd.concat([st.session_state["data"], new_df], ignore_index=True)
            
            # Update session state
            st.session_state.scores = scores
            st.session_state.vitals = vitals_for_scoring
            
            st.success("✅ Vital signs saved and clinical scores calculated!")
            
            # Show immediate results
            with st.expander("📊 View Calculated Scores", expanded=True):
                score_col1, score_col2, score_col3 = st.columns(3)
                
                with score_col1:
                    st.metric(
                        "NEWS2", 
                        scores["NEWS2"], 
                        help="National Early Warning Score 2"
                    )
                    st.markdown(pill(scores["NEWS2_Risk"]), unsafe_allow_html=True)
                
                with score_col2:
                    st.metric(
                        "qSOFA", 
                        scores["qSOFA"], 
                        help="Quick Sequential Organ Failure Assessment"
                    )
                    st.markdown(pill(scores["qSOFA_Risk"]), unsafe_allow_html=True)
                
                with score_col3:
                    st.metric(
                        "SIRS", 
                        scores["SIRS"], 
                        help="Systemic Inflammatory Response Syndrome"
                    )
                    st.markdown(pill(scores["SIRS_Risk"]), unsafe_allow_html=True)
                
                if scores["ML_Prediction"] is not None:
                    st.markdown("**ML Risk Assessment:**")
                    if scores["ML_Prob"] is not None:
                        st.metric("ML Probability", f"{scores['ML_Prob']:.1%}")
                    st.markdown(pill(scores["ML_Risk"]), unsafe_allow_html=True)

# === ANALYTICS TAB ===
with tab_analytics:
    st.markdown("#### Clinical Analytics & Trends")
    
    df = st.session_state["data"]
    
    if df.empty:
        st.info("📊 No data available for analysis. Add patient vitals in the **Patient Entry** tab to view analytics.")
    else:
        # Risk Distribution
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**NEWS2 Risk Distribution**")
            
            risk_counts = df["NEWS2_Risk"].value_counts()
            for category in ["Low", "Medium", "High"]:
                if category not in risk_counts:
                    risk_counts.loc[category] = 0
            risk_counts = risk_counts[["Low", "Medium", "High"]]
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=[f"{k} ({int(v)})" for k, v in risk_counts.items()],
                values=risk_counts.values,
                hole=0.4,
                marker_colors=["#00d1a7", "#ffcc66", "#ff6b6b"]
            )])
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Data Completeness**")
            
            fields = ["Temperature", "HeartRate", "RespiratoryRate", "SystolicBP", "DiastolicBP", "SpO2", "WBC", "GCS"]
            completeness = []
            
            for field in fields:
                if field in df.columns:
                    if df[field].dtype == object:
                        filled = df[field].astype(str).str.strip().ne("").mean() * 100.0
                    else:
                        filled = df[field].notna().mean() * 100.0
                    completeness.append(filled)
                else:
                    completeness.append(0.0)
            
            comp_df = pd.DataFrame({"Field": fields, "Completeness %": completeness})
            
            fig = px.bar(
                comp_df, 
                x="Field", 
                y="Completeness %",
                template=PLOTLY_TEMPLATE,
                color="Completeness %",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Trends Over Time
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Clinical Score Trends**")
        
        trend_df = df[["Timestamp", "NEWS2", "qSOFA", "SIRS"]].copy()
        if not pd.api.types.is_datetime64_any_dtype(trend_df["Timestamp"]):
            trend_df["Timestamp"] = pd.to_datetime(trend_df["Timestamp"], errors="coerce")
        
        trend_df = trend_df.sort_values("Timestamp")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend_df["Timestamp"], y=trend_df["NEWS2"], name="NEWS2", line=dict(color="#ff6b6b", width=3)))
        fig.add_trace(go.Scatter(x=trend_df["Timestamp"], y=trend_df["qSOFA"], name="qSOFA", line=dict(color="#ffcc66", width=3)))
        fig.add_trace(go.Scatter(x=trend_df["Timestamp"], y=trend_df["SIRS"], name="SIRS", line=dict(color="#7fd3ff", width=3)))
        
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Time",
            yaxis_title="Score",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export Options
        st.markdown('<div class="soft">', unsafe_allow_html=True)
        st.markdown("**Export Data**")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Download CSV",
                data=csv_data,
                file_name=f"sepsis_data_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_col2:
            json_data = df.to_json(orient="records", date_format="iso").encode("utf-8")
            st.download_button(
                "📋 Download JSON", 
                data=json_data,
                file_name=f"sepsis_data_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with export_col3:
            summary = {
                "total_records": len(df),
                "high_risk_count": (df["NEWS2_Risk"] == "High").sum(),
                "avg_news2": df["NEWS2"].mean(),
                "export_timestamp": dt.datetime.now().isoformat()
            }
            summary_json = json.dumps(summary, indent=2).encode("utf-8")
            st.download_button(
                "📊 Download Summary",
                data=summary_json,
                file_name=f"sepsis_summary_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed Data Table
        with st.expander("📋 View Detailed Data Table"):
            st.dataframe(df, use_container_width=True, height=400)

# === AI ASSISTANT TAB ===
with tab_assistant:
    st.markdown("#### RAG-Powered Clinical Assistant")
    
    # Chat Container
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="card" style="height: 400px; overflow-y: auto; padding: 1rem; background: rgba(20,20,25,0.6);">', unsafe_allow_html=True)
        
        if not st.session_state.chat:
            st.markdown("**Welcome to the Clinical Assistant!**")
            st.markdown("Ask questions about sepsis guidelines, treatments, or get help with clinical decisions.")
            st.markdown("---")
        
        for i, message in enumerate(st.session_state.chat):
            if message["role"] == "user":
                st.markdown(f"**🧑‍⚕️ You:** {message['content']}")
            else:
                st.markdown(f"**🤖 Assistant:** {message['content']}")
            
            if i < len(st.session_state.chat) - 1:
                st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Input Section
    st.markdown("**Ask a Question:**")
    
    # Quick action buttons
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    
    with quick_col1:
        if st.button("💊 Antibiotics"):
            question = "What are the recommended empiric antibiotics for suspected sepsis?"
            st.session_state.pending_question = question
    
    with quick_col2:
        if st.button("💧 Fluid Therapy"):
            question = "How much fluid resuscitation should I give for septic shock?"
            st.session_state.pending_question = question
    
    with quick_col3:
        if st.button("🔍 Diagnostics"):
            question = "What laboratory tests should I order for suspected sepsis?"
            st.session_state.pending_question = question
    
    with quick_col4:
        if st.button("⚠️ Warning Signs"):
            question = "What are the early warning signs of sepsis?"
            st.session_state.pending_question = question
    
    # Chat input
    question = st.text_area(
        "Type your question here:", 
        value=st.session_state.get("pending_question", ""),
        height=100,
        placeholder="e.g., 'What are the sepsis-3 criteria?' or 'When should I start vasopressors?'"
    )
    
    col_send, col_clear = st.columns([4, 1])
    
    with col_send:
        if st.button("🚀 Ask Assistant", use_container_width=True, type="primary"):
            if question.strip():
                # Add user question to chat
                st.session_state.chat.append({"role": "user", "content": question})
                
                try:
                    # Get RAG response
                    rag_system = load_rag_system()
                    
                    with st.spinner("🔍 Searching medical guidelines..."):
                        # Include current patient context if available
                        patient_context = None
                        if st.session_state.scores:
                            patient_context = {
                                "vitals": st.session_state.vitals,
                                "scores": st.session_state.scores
                            }
                        
                        answer, sources = rag_system.query(question, patient_scores=patient_context)
                    
                    # Format response with sources
                    response = answer
                    if sources:
                        unique_sources = list(dict.fromkeys(sources))[:5]  # Limit to 5 sources
                        response += "\n\n**📚 Sources:**\n"
                        for i, source in enumerate(unique_sources, 1):
                            response += f"{i}. {source}\n"
                    
                    # Add assistant response to chat
                    st.session_state.chat.append({"role": "assistant", "content": response})
                    
                    # Clear the input
                    st.session_state.pending_question = ""
                    
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.session_state.chat.append({"role": "assistant", "content": error_msg})
                
                st.rerun()
            else:
                st.warning("Please enter a question first.")
    
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat = []
            st.session_state.pending_question = ""
            st.rerun()
    
    # Context information
    with st.expander("ℹ️ About the Assistant"):
        st.markdown("""
        **This AI assistant is powered by:**
        - 🧠 **RAG (Retrieval-Augmented Generation)**: Searches medical guidelines and evidence
        - 📚 **Medical Knowledge Base**: Sepsis care protocols, evidence-based treatments
        - 🤖 **Google Gemini AI**: Advanced language understanding and generation
        - 🎯 **Patient Context**: Uses current vital signs and scores when available
        
        **Best for asking about:**
        - Treatment protocols and guidelines
        - Diagnostic criteria and scoring systems
        - Medication dosing and administration
        - Clinical decision support
        - Evidence-based recommendations
        """)

# === DATA UPLOAD TAB ===
with tab_upload:
    st.markdown("#### Data Management & Upload")
    
    # Current Data Overview
    st.markdown("**Current Session Data**")
    
    df = st.session_state["data"]
    
    if not df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
        
        with col2:
            high_risk = (df["NEWS2_Risk"] == "High").sum() if "NEWS2_Risk" in df.columns else 0
            st.metric("High Risk Cases", high_risk)
        
        with col3:
            latest = df.iloc[-1]["Timestamp"] if len(df) > 0 else "None"
            st.metric("Latest Entry", str(latest)[:19])
        
        # Data preview
        with st.expander("📋 Preview Current Data"):
            st.dataframe(df, use_container_width=True, height=300)
    
    else:
        st.info("No data in current session.")
    
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    
    # File Upload Section
    st.markdown("**📁 Upload Patient Data**")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload a CSV file with patient vital signs and clinical data"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded CSV
            upload_df = pd.read_csv(uploaded_file)
            
            st.markdown("**📊 File Preview:**")
            st.dataframe(upload_df.head(), use_container_width=True)
            
            # Validation
            required_cols = ["Temperature", "HeartRate", "RespiratoryRate", "SystolicBP", "SpO2"]
            missing_cols = [col for col in required_cols if col not in upload_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Required columns: Temperature, HeartRate, RespiratoryRate, SystolicBP, SpO2")
            
            else:
                st.success("✅ File format validated!")
                
                col_process, col_sample = st.columns(2)
                
                with col_process:
                    if st.button("🔄 Process & Add to Session", use_container_width=True):
                        processed_records = []
                        
                        progress_bar = st.progress(0)
                        total_rows = len(upload_df)
                        
                        for idx, row in upload_df.iterrows():
                            # Create vitals dict for scoring
                            vitals = {
                                "temperature": float(row.get("Temperature", 37.0)),
                                "heart_rate": float(row.get("HeartRate", 80)),
                                "respiratory_rate": float(row.get("RespiratoryRate", 16)),
                                "systolic_bp": float(row.get("SystolicBP", 120)),
                                "diastolic_bp": float(row.get("DiastolicBP", 80)),
                                "spo2": float(row.get("SpO2", 98)),
                                "wbc": float(row.get("WBC", 0)) if pd.notna(row.get("WBC")) else None,
                                "gcs": float(row.get("GCS", 15)),
                                "consciousness": str(row.get("Consciousness", "Alert"))
                            }
                            
                            # Compute scores
                            scores = compute_all_scores(vitals)
                            
                            # Create record
                            record = {
                                "Timestamp": pd.to_datetime(row.get("Timestamp", dt.datetime.now())),
                                "Temperature": vitals["temperature"],
                                "HeartRate": vitals["heart_rate"],
                                "RespiratoryRate": vitals["respiratory_rate"],
                                "SystolicBP": vitals["systolic_bp"],
                                "DiastolicBP": vitals["diastolic_bp"],
                                "SpO2": vitals["spo2"],
                                "WBC": vitals["wbc"],
                                "GCS": vitals["gcs"],
                                "Consciousness": vitals["consciousness"],
                                "Notes": str(row.get("Notes", "")).strip()
                            }
                            
                            # Add scores
                            record.update(scores)
                            processed_records.append(record)
                            
                            # Update progress
                            progress_bar.progress((idx + 1) / total_rows)
                        
                        # Add to session data
                        new_df = pd.DataFrame(processed_records, columns=SCHEMA)
                        st.session_state["data"] = pd.concat([st.session_state["data"], new_df], ignore_index=True)
                        
                        st.success(f"✅ Successfully processed and added {len(processed_records)} records!")
                        st.rerun()
                
                with col_sample:
                    if st.button("📥 Download Sample CSV", use_container_width=True):
                        sample_data = {
                            "Timestamp": [dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                            "Temperature": [37.2],
                            "HeartRate": [85],
                            "RespiratoryRate": [18],
                            "SystolicBP": [125],
                            "DiastolicBP": [75],
                            "SpO2": [97],
                            "WBC": [8.5],
                            "GCS": [15],
                            "Consciousness": ["Alert"],
                            "Notes": ["Sample patient data"]
                        }
                        
                        sample_df = pd.DataFrame(sample_data)
                        csv_data = sample_df.to_csv(index=False).encode("utf-8")
                        
                        st.download_button(
                            "📄 Download",
                            data=csv_data,
                            file_name="sample_patient_data.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    
    # Export Options
    st.markdown("**📤 Export Current Data**")
    
    if not df.empty:
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Export as CSV",
                data=csv_data,
                file_name=f"sepsis_data_export_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_col2:
            json_data = df.to_json(orient="records", date_format="iso").encode("utf-8")
            st.download_button(
                "📋 Export as JSON",
                data=json_data,
                file_name=f"sepsis_data_export_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with export_col3:
            # Create summary report
            summary_stats = {
                "export_info": {
                    "timestamp": dt.datetime.now().isoformat(),
                    "total_records": len(df),
                    "date_range": {
                        "start": df["Timestamp"].min().isoformat() if len(df) > 0 else None,
                        "end": df["Timestamp"].max().isoformat() if len(df) > 0 else None
                    }
                },
                "risk_distribution": {
                    "high_risk": int((df["NEWS2_Risk"] == "High").sum()),
                    "medium_risk": int((df["NEWS2_Risk"] == "Medium").sum()),
                    "low_risk": int((df["NEWS2_Risk"] == "Low").sum())
                },
                "clinical_averages": {
                    "avg_news2": float(df["NEWS2"].mean()),
                    "avg_qsofa": float(df["qSOFA"].mean()),
                    "avg_sirs": float(df["SIRS"].mean()),
                    "avg_temperature": float(df["Temperature"].mean()),
                    "avg_heart_rate": float(df["HeartRate"].mean())
                }
            }
            
            summary_json = json.dumps(summary_stats, indent=2).encode("utf-8")
            st.download_button(
                "📊 Export Summary",
                data=summary_json,
                file_name=f"sepsis_summary_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    else:
        st.info("No data available for export. Add patient data first.")
    
    # Data Management
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("**🗂️ Data Management**")
    
    manage_col1, manage_col2 = st.columns(2)
    
    with manage_col1:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            if st.session_state.get("confirm_clear"):
                st.session_state["data"] = pd.DataFrame(columns=SCHEMA)
                st.session_state.scores = None
                st.session_state.vitals = {
                    "temperature": 37.0, "heart_rate": 80, "respiratory_rate": 16,
                    "systolic_bp": 120, "diastolic_bp": 80, "spo2": 98,
                    "consciousness": "Alert", "wbc": 0, "gcs": 15
                }
                st.session_state.confirm_clear = False
                st.success("✅ All data cleared!")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm clearing all data")
    
    with manage_col2:
        if st.button("🔄 Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["auth_user"]:  # Keep authentication
                    del st.session_state[key]
            st.success("✅ Session reset!")
            st.rerun()

# ------------------------- Pages ------------------------- #
def goto(page_name):
    """Navigate to a specific page"""
    st.session_state.route = page_name
    st.rerun()

def page_auth():
    st.markdown("<div class='auth-wrap'><div class='gcard auth-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;margin:0 0 6px;'>Welcome to Sepsis RAG Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<div class='small' style='text-align:center;margin-bottom:10px'>Advanced Clinical Decision Support System</div>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Login", "Register"])
    
    with tabs[0]:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            if u and p:
                st.session_state.auth_user = u
                goto("MAIN")
            else:
                st.error("Enter both username and password.")
    
    with tabs[1]:
        u2 = st.text_input("Create username", key="reg_user")
        p2 = st.text_input("Create password", type="password", key="reg_pass")
        if st.button("Create account", key="reg_btn"):
            if u2 and p2:
                st.session_state.auth_user = u2
                goto("MAIN")
            else:
                st.error("Enter both username and password.")
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def nebula_svg():
    return """
<svg class="nebula" viewBox="0 0 1200 420" preserveAspectRatio="xMidYMid slice">
  <defs>
    <radialGradient id="g1" cx="30%" cy="30%">
      <stop offset="0%" stop-color="#7fd3ff" stop-opacity=".45"/>
      <stop offset="100%" stop-color="#7c3aed" stop-opacity=".0"/>
    </radialGradient>
    <radialGradient id="g2" cx="70%" cy="70%">
      <stop offset="0%" stop-color="#8b5cf6" stop-opacity=".35"/>
      <stop offset="100%" stop-color="#0ea5e9" stop-opacity=".0"/>
    </radialGradient>
    <filter id="blur"><feGaussianBlur stdDeviation="40"/></filter>
  </defs>
  <circle cx="280" cy="160" r="220" fill="url(#g1)" filter="url(#blur)"/>
  <circle cx="900" cy="300" r="260" fill="url(#g2)" filter="url(#blur)"/>
</svg>
    """

def page_home():
    # header row (slim)
    st.markdown(
        f"<div class='small' style='display:flex;justify-content:space-between;opacity:.9'>"
        f"<div>Signed in as: <b>{st.session_state.get('auth_user','—')}</b></div>"
        f"<div></div></div>", unsafe_allow_html=True)

    # HERO with nebula artwork
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.markdown(nebula_svg(), unsafe_allow_html=True)
    st.markdown("<h1>Welcome to Sepsis RAG Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p>AI-powered clinical support for early sepsis detection and management.</p>", unsafe_allow_html=True)

    # CTA tiles
    st.markdown("<div class='cta'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>🩺 Enter Patient Vitals</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub'>Temperature, HR, BP, SpO₂, etc.</div>", unsafe_allow_html=True)
        st.markdown("<div class='btn-wrap'>", unsafe_allow_html=True)
        if st.button("Open Dashboard", key="home_to_main"):
            goto("MAIN")
        st.markdown("</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>📊 View Risk Scores</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub'>NEWS2, qSOFA, SIRS, ML predictions.</div>", unsafe_allow_html=True)
        st.markdown("<div class='btn-wrap'>", unsafe_allow_html=True)
        if st.button("Go to Analytics", key="home_to_analytics"):
            goto("MAIN")
        st.markdown("</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>💬 Ask Clinical Questions</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub'>Get guideline-grounded answers.</div>", unsafe_allow_html=True)
        st.markdown("<div class='btn-wrap'>", unsafe_allow_html=True)
        if st.button("Open Assistant", key="home_to_assistant"):
            goto("MAIN")
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end cta
    st.markdown("</div>", unsafe_allow_html=True)  # end hero

    # Footer actions
    col_l, col_c, col_r = st.columns([1,6,1])
    with col_l:
        if st.button("Logout", key="logout_btn"):
            for key in list(st.session_state.keys()):
                if key != "route":
                    del st.session_state[key]
            goto("AUTH")

def page_vitals_tabs():
    topbar(show_back=True)

    # quick popover editor + risk compute
    tb_l, tb_r = st.columns([3,2])
    with tb_l:
        st.markdown("<div class='card small'>Use presets to simulate risk, then generate a digital report.</div>", unsafe_allow_html=True)
    with tb_r:
        with st.popover("✏️ Edit vitals"):
            v = st.session_state.vitals
            c1, c2, c3 = st.columns(3)
            with c1:
                v["temperature"] = st.number_input("Temp (°C)", 30.0, 45.0, v.get("temperature", 37.0), 0.1)
                v["spo2"] = st.number_input("SpO₂ (%)", 70, 100, v.get("spo2", 98), 1)
            with c2:
                v["heart_rate"] = st.number_input("Heart Rate (bpm)", 30, 200, v.get("heart_rate", 80), 1)
                v["systolic_bp"] = st.number_input("Systolic BP (mmHg)", 60, 250, v.get("systolic_bp", 120), 1)
            with c3:
                v["respiratory_rate"] = st.number_input("Resp Rate (/min)", 5, 50, v.get("respiratory_rate", 16), 1)
                v["wbc"] = st.number_input("WBC (/μL) (optional)", 0, 50000, v.get("wbc", 0), 100)
            v["consciousness"] = st.selectbox("AVPU", ["Alert","Voice","Pain","Unresponsive"],
                                              index=["Alert","Voice","Pain","Unresponsive"].index(v.get("consciousness","Alert")))
            st.caption("Tip: use presets below.")
        if st.button("⚙️ Calculate risk", key="calc_risk"):
            vt = st.session_state.vitals
            errs=[]
            if not (30 <= vt["temperature"] <= 45): errs.append("Temperature out of range")
            if not (30 <= vt["heart_rate"] <= 200): errs.append("Heart rate out of range")
            if errs: st.error(" • ".join(errs))
            else:
                compute_scores(vt)
                st.success("Scores updated.")

    tabs = st.tabs(["Dashboard", "Timeline", "Report"])

    # DASHBOARD
    with tabs[0]:
        pa, pb, pc, pd = st.columns(4)
        if pa.button("Preset: High"): st.session_state.vitals.update({"temperature":39.2,"heart_rate":125,"respiratory_rate":28,"systolic_bp":88,"spo2":89,"consciousness":"Voice"})
        if pb.button("Preset: Med"):  st.session_state.vitals.update({"temperature":38.5,"heart_rate":105,"respiratory_rate":22,"systolic_bp":95,"spo2":94,"consciousness":"Alert"})
        if pc.button("Preset: Low"):  st.session_state.vitals.update({"temperature":37.1,"heart_rate":85,"respiratory_rate":18,"systolic_bp":115,"spo2":97,"consciousness":"Alert"})
        if pd.button("➕ Snapshot"):
            st.session_state.timeline.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **st.session_state.vitals})
        st.write("")
        scores = st.session_state.scores
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown('<div class="card">**NEWS2**</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.6rem;font-weight:900'>{scores['news2'][0] if scores else '—'}</div>", unsafe_allow_html=True)
            if scores: st.markdown(risk_badge(scores["news2"][1]), unsafe_allow_html=True)
        with k2:
            st.markdown('<div class="card">**qSOFA**</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.6rem;font-weight:900'>{scores['qsofa'][0] if scores else '—'}</div>", unsafe_allow_html=True)
            if scores: st.markdown(risk_badge(scores["qsofa"][1]), unsafe_allow_html=True)
        with k3:
            st.markdown('<div class="card">**SIRS**</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.6rem;font-weight:900'>{scores['sirs'][0] if scores else '—'}</div>", unsafe_allow_html=True)
            if scores: st.markdown(risk_badge(scores["sirs"][1]), unsafe_allow_html=True)
        with k4:
            st.markdown('<div class="card">**Profile**</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.6rem;font-weight:900'>{scores['cluster'] if scores else '—'}</div>", unsafe_allow_html=True)
            st.caption("K-Means cluster")

        c1, c2 = st.columns([1.1,1])
        with c1:
            risk_numeric = 0
            if scores:
                risk_numeric += min(20, scores["news2"][0] * 2)
                risk_numeric += min(9, scores["qsofa"][0] * 3)
                risk_numeric += min(8, scores["sirs"][0] * 2)
                risk_numeric = max(0, min(100, int((risk_numeric / 37) * 100)))
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk_numeric, title={"text":"Overall Risk (heuristic)"},
                gauge={"axis":{"range":[0,100]},"bar":{"thickness":0.3},
                       "steps":[{"range":[0,35],"color":"#10b98140"},
                                {"range":[35,70],"color":"#f59e0b40"},
                                {"range":[70,100],"color":"#ef444440"}],
                       "threshold":{"line":{"color":"#ef4444","width":4},"thickness":0.75,"value":70}},
                number={"suffix":"%"}))
            gauge.update_layout(template=PLOTLY_TEMPLATE, height=260, margin=dict(l=18,r=18,t=30,b=10))
            st.plotly_chart(gauge, use_container_width=True)
        with c2:
            normals={"temperature":(36.1,37.2),"heart_rate":(60,100),"respiratory_rate":(12,20),"systolic_bp":(100,130),"spo2":(95,100)}
            current=scores["vitals"] if scores else st.session_state.vitals
            cats,vals=[],[]
            for k,(lo,hi) in normals.items():
                v=current[k]; cats.append(k)
                score=0.1 if lo<=v<=hi else min(1.0, abs(v-(lo if v<lo else hi))/max(1e-6,(hi-lo)))
                vals.append(score)
            radar=go.Figure()
            radar.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill='toself', name='Abnormality'))
            radar.update_layout(template=PLOTLY_TEMPLATE, polar=dict(radialaxis=dict(range=[0,1], showticklabels=False)),
                                showlegend=False, height=260, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(radar, use_container_width=True)

        st.markdown("#### 🔎 Interpretation")
        st.markdown('<div class="card" style="max-height:170px; overflow:auto">', unsafe_allow_html=True)
        if scores:
            lbl,col,adv = fb_interpret(scores["news2"][0], scores["qsofa"][0], scores["sirs"][0]) if not interpret_risk else interpret_risk(scores["news2"][0], scores["qsofa"][0], scores["sirs"][0])
            st.markdown(f"**Risk:** <span style='color:{col}; font-weight:800'>{lbl}</span><br>{adv}", unsafe_allow_html=True)
        else:
            st.info("Click **⚙️ Calculate risk** to populate this section.")
        st.markdown('</div>', unsafe_allow_html=True)

    # TIMELINE
    with tabs[1]:
        top = st.columns([1,1,1,1])
        with top[0]:
            if st.button("➕ Add snapshot (now)"):
                st.session_state.timeline.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **st.session_state.vitals})
        with top[1]:
            if st.button("🧹 Clear"):
                st.session_state.timeline = []
        with top[3]:
            st.caption("Snapshots capture current vitals for trends.")
        st.markdown('<div class="card" style="max-height:460px; overflow:auto">', unsafe_allow_html=True)
        if st.session_state.timeline:
            df = pd.DataFrame(st.session_state.timeline)
            df["timestamp"]=pd.to_datetime(df["timestamp"]); df=df.sort_values("timestamp")
            long = df.melt(id_vars=["timestamp","consciousness","wbc"],
                           value_vars=["heart_rate","respiratory_rate","systolic_bp","temperature","spo2"],
                           var_name="vital", value_name="value")
            line = px.line(long, x="timestamp", y="value", color="vital", markers=True, template=PLOTLY_TEMPLATE)
            line.update_layout(height=360, margin=dict(l=6,r=6,t=6,b=6), legend_title="")
            st.plotly_chart(line, use_container_width=True)
            st.dataframe(df.set_index("timestamp"), use_container_width=True, height=200)
        else:
            st.info("No snapshots yet. Add one to visualize trends.")
        st.markdown('</div>', unsafe_allow_html=True)

    # REPORT
    with tabs[2]:
        st.markdown("### 🧾 Create Digital Report")
        st.caption("Build a compact patient profile & export.")
        p = st.session_state.profile
        r1c1,r1c2,r1c3,r1c4,r1c5,r1c6 = st.columns([1.2,1.2,0.9,0.9,0.9,1.0])
        with r1c1: p["username"]=st.text_input("Username / Handle", p.get("username",""))
        with r1c2: p["patient_name"]=st.text_input("Patient name", p.get("patient_name",""))
        with r1c3: p["mrn"]=st.text_input("MRN / ID", p.get("mrn",""))
        with r1c4: p["age"]=st.number_input("Age", 0, 120, int(p.get("age",0)))
        with r1c5: p["sex"]=st.selectbox("Sex", ["Male","Female","Other"], index=["Male","Female","Other"].index(p.get("sex","Male")))
        with r1c6:
            col_h,col_w = st.columns(2)
            with col_h: p["height_cm"]=st.number_input("Ht (cm)", 0.0, 260.0, float(p.get("height_cm",0.0)), step=0.1)
            with col_w: p["weight_kg"]=st.number_input("Wt (kg)", 0.0, 400.0, float(p.get("weight_kg",0.0)), step=0.1)
        r2c1,r2c2,r2c3,r2c4 = st.columns([1.2,1.2,1.2,1.2])
        with r2c1: p["chief_complaint"]=st.text_input("Chief complaint", p.get("chief_complaint",""))
        with r2c2: p["infection_source"]=st.text_input("Suspected infection source", p.get("infection_source",""))
        with r2c3: p["allergies"]=st.text_input("Allergies", p.get("allergies",""))
        with r2c4: p["comorbidities"]=st.text_input("Comorbidities", p.get("comorbidities",""))
        r3c1,r3c2,r3c3 = st.columns([1.4,0.8,1.8])
        with r3c1: p["current_meds"]=st.text_input("Current meds", p.get("current_meds",""))
        with r3c2: p["code_status"]=st.selectbox("Code status", ["Full Code","DNR","DNI","Limited"],
                                                 index=["Full Code","DNR","DNI","Limited"].index(p.get("code_status","Full Code")))
        with r3c3: p["notes"]=st.text_input("Notes", p.get("notes",""))
        files = st.file_uploader("Attach medical report(s) (PDF/Image)", type=["pdf","png","jpg","jpeg"], accept_multiple_files=True)
        if files:
            for f in files:
                st.session_state.profile["attachments"].append({"name": f.name, "type": f.type})
            st.success(f"Added {len(files)} attachment(s).")
        a1,a2,a3,_ = st.columns([1,1,1,3])
        if a1.button("💾 Save Profile"): st.success("Profile saved for this session.")
        if a2.button("🧹 Clear Profile"):
            st.session_state.profile = {k: ("" if isinstance(v,str) else 0 if isinstance(v,(int,float)) else [] if isinstance(v,list) else v)
                                        for k,v in st.session_state.profile.items()}
            st.rerun()
        gen_clicked = a3.button("🧾 Generate Report")
        if gen_clicked:
            h_m = (p.get("height_cm",0.0) or 0)/100.0; w_kg = p.get("weight_kg",0.0) or 0
            bmi = round(w_kg/(h_m*h_m),1) if h_m and w_kg else None
            s = st.session_state.scores; v = st.session_state.vitals
            lines = ["# Patient Digital Report", f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
                     "## Identity",
                     f"- Username: {p.get('username','—')}",
                     f"- Name: {p.get('patient_name','—')}  |  MRN: {p.get('mrn','—')}",
                     f"- Age/Sex: {p.get('age','—')} / {p.get('sex','—')}",
                     f"- Height/Weight/BMI: {p.get('height_cm','—')} cm / {p.get('weight_kg','—')} kg" + (f" / BMI {bmi}" if bmi else ""),
                     "", "## Clinical Context",
                     f"- Chief complaint: {p.get('chief_complaint','—')}",
                     f"- Suspected infection source: {p.get('infection_source','—')}",
                     f"- Allergies: {p.get('allergies','—')}",
                     f"- Comorbidities: {p.get('comorbidities','—')}",
                     f"- Current meds: {p.get('current_meds','—')}",
                     f"- Code status: {p.get('code_status','—')}",
                     f"- Notes: {p.get('notes','—')}"]
            if p.get("attachments"):
                lines.append(f"- Attachments: {', '.join([a['name'] for a in p['attachments']])}")
            lines += ["", "## Latest Vitals",
                      f"- Temp: {v['temperature']} °C   | HR: {v['heart_rate']} bpm   | RR: {v['respiratory_rate']} /min",
                      f"- SBP: {v['systolic_bp']} mmHg  | SpO₂: {v['spo2']} %         | AVPU: {v['consciousness']}", "",
                      "## Risk Scores"]
            if s:
                lines += [f"- NEWS2: {s['news2'][0]} ({s['news2'][1]})",
                          f"- qSOFA: {s['qsofa'][0]} ({s['qsofa'][1]})",
                          f"- SIRS: {s['sirs'][0]} ({s['sirs'][1]})",
                          f"- Profile (K-Means): {s['cluster']}"]
                lbl,col,adv = fb_interpret(s['news2'][0], s['qsofa'][0], s['sirs'][0]) if not interpret_risk else interpret_risk(s['news2'][0], s['qsofa'][0], s['sirs'][0])
                lines += ["", "## Interpretation", f"- Risk: {lbl} — {adv}"]
            else:
                lines.append("_No scores yet. Click ‘⚙️ Calculate risk’ above._")
            report_md = "\n".join(lines)
            st.markdown("#### Preview")
            st.markdown('<div class="card" style="height:360px; overflow:auto">', unsafe_allow_html=True)
            st.markdown(report_md)
            st.markdown('</div>', unsafe_allow_html=True)
            d1,d2,d3 = st.columns(3)
            with d1:
                st.download_button("⬇️ Profile (JSON)", data=json.dumps(st.session_state.profile, indent=2),
                                   file_name="patient_profile.json", mime="application/json", use_container_width=True)
            with d2:
                if st.session_state.timeline:
                    df = pd.DataFrame(st.session_state.timeline); buf = io.StringIO(); df.to_csv(buf, index=False)
                    st.download_button("⬇️ Timeline (CSV)", data=buf.getvalue(), file_name="vitals_timeline.csv",
                                       mime="text/csv", use_container_width=True)
                else:
                    st.button("⬇️ Timeline (CSV)", disabled=True, use_container_width=True)
            with d3:
                st.download_button("⬇️ Report (TXT)", data=report_md, file_name="patient_report.txt",
                                   mime="text/plain", use_container_width=True)

def page_chat():
    topbar(show_back=True)
    st.markdown("<div class='card small'>Guideline-aware assistant with citations.</div>", unsafe_allow_html=True)

    chat_h = 520
    st.markdown(f'<div class="card" style="height:{chat_h}px; overflow:auto; padding:12px;">', unsafe_allow_html=True)
    if len(st.session_state.chat) == 0:
        st.markdown("**Ask about fluids, antibiotics, or immediate steps.**")
    else:
        for m in st.session_state.chat:
            speaker = "You" if m["role"]=="user" else "Assistant"
            st.markdown(f"**{speaker}:** {m['content']}")
    st.markdown('</div>', unsafe_allow_html=True)

    c_l, c_r, c_r2 = st.columns([6,1.2,1.2])
    with c_l:
        q = st.text_input("Ask a clinical question", value="", placeholder="e.g., Empiric antibiotics for suspected sepsis?")
    with c_r:
        ask = st.button("Ask 💬", use_container_width=True, key="ask_btn")
    with c_r2:
        if st.button("💧 Fluids", use_container_width=True): q = "How much crystalloid should I give initially?"
        if st.button("💊 Rx", use_container_width=True): q = "What are recommended empiric antibiotics for suspected sepsis?"
    if ask and q:
        st.session_state.chat.append({"role":"user","content":q})
        try:
            rag = load_rag_system()
            with st.spinner("Retrieving guidelines..."):
                answer, sources = rag.query(q, patient_scores=st.session_state.scores)
            unique_sources = list(dict.fromkeys(sources)) if sources else []
            cite = ("\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in unique_sources[:5]])) if unique_sources else ""
            st.session_state.chat.append({"role":"assistant","content":answer + cite})
        except Exception as e:
            st.session_state.chat.append({"role":"assistant","content":f"Sorry — guidance unavailable right now. ({e})"})
        st.rerun()

# ------------------------- Main Application ------------------------- #
def page_main():
    """Main comprehensive medical dashboard application"""
    
    # Initialize core systems with eager loading
    initialize_rag_system()
    load_ml_models() 
    
    # Status display
    status_col1, status_col2, status_col3 = st.columns([2, 1, 1])
    
    with status_col1:
        st.markdown("### 🏥 Sepsis Clinical Decision Support System")
    
    with status_col2:
        if st.session_state.rag_loaded:
            st.success("🧠 RAG Ready")
        else:
            st.error("❌ RAG Error")
    
    with status_col3:
        if st.session_state.ml_loaded:
            st.success("🤖 ML Ready") 
        else:
            st.warning("⚠️ ML Unavailable")
    
    # Main application tabs
    tab_overview, tab_patient, tab_analytics, tab_assistant, tab_upload = st.tabs([
        "📊 Overview", "🩺 Patient Entry", "📈 Analytics", "💬 AI Assistant", "📁 Data Upload"
    ])
    
    # Tab implementations have been added above
    
    # Sidebar status
    with st.sidebar:
        st.markdown("### 📊 Quick Status")
        
        if st.session_state.scores:
            scores = st.session_state.scores
            st.markdown(f"**Latest NEWS2:** {scores['NEWS2']} ({scores['NEWS2_Risk']})")
            st.markdown(f"**Latest qSOFA:** {scores['qSOFA']} ({scores['qSOFA_Risk']})")
            st.markdown(f"**Latest SIRS:** {scores['SIRS']} ({scores['SIRS_Risk']})")
            
            if scores.get("ML_Risk"):
                st.markdown(f"**ML Risk:** {scores['ML_Risk']}")
        else:
            st.info("No patient data entered yet")
        
        st.markdown("---")
        
        # Quick actions
        if st.button("🆘 Emergency Protocol"):
            st.session_state.pending_question = "What is the emergency sepsis protocol?"
        
        if st.button("📋 Assessment Guide"):
            st.session_state.pending_question = "How do I assess a patient for sepsis?"
        
        st.markdown("---")
        st.markdown(f"**Logged in as:** {st.session_state.get('auth_user', 'Unknown')}")
        
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                if key != "route":
                    del st.session_state[key]
            st.session_state.route = "AUTH"
            st.rerun()

# ------------------------- Router ------------------------- #
def main():
    """Main application router"""
    
    # Initialize route
    if "route" not in st.session_state:
        st.session_state.route = "AUTH"
    
    # Route to appropriate page
    if st.session_state.route == "AUTH":
        page_auth()
    elif st.session_state.route == "HOME":
        page_home()
    elif st.session_state.route == "MAIN":
        page_main()
    else:
        st.session_state.route = "AUTH"
        st.rerun()

if __name__ == "__main__":
    main()
