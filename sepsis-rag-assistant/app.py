<<<<<<< HEAD
import streamlit as st
import pandas as pd
import numpy as np
import os

# Local project imports
from scoring import (
    calculate_news2,
    calculate_qsofa,
    calculate_sirs,
    interpret_risk
)


# Page configuration
st.set_page_config(
    page_title="Sepsis Early Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)
=======
# app.py
# Sepsis RAG Assistant — Auth → Home (nebula hero) → Chat or Vitals (tabs)
# Glassmorphism, responsive, back/home navigation, no blank pages

import os, io, json, math, time, pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ------------------------- Page & Theme ------------------------- #
st.set_page_config(page_title="Sepsis RAG Assistant", page_icon="🩺", layout="wide")

CSS = r"""
:root{
  --bg1:#051428; --bg2:#0b0f18;
  --glass: rgba(255,255,255,.05);
  --glass-2: rgba(255,255,255,.04);
  --stroke: rgba(255,255,255,.10);
  --stroke-2: rgba(255,255,255,.12);
  --fg: rgba(244,246,255,.95);
  --muted: rgba(230,238,255,.60);
  --accent:#7fd3ff;
}
body { background: radial-gradient(1200px 650px at 20% -10%, #1a2240 0%, var(--bg1) 40%) }
.main .block-container{ max-width: 1200px; padding-top:.5rem; padding-bottom:1.2rem; }

.small{ color:var(--muted); font-size:.92rem; }
.card{
  background: var(--glass); border:1px solid var(--stroke); border-radius:16px; padding:14px;
  backdrop-filter: blur(10px) saturate(135%); -webkit-backdrop-filter: blur(10px) saturate(135%);
  box-shadow: 0 10px 28px rgba(0,0,0,.35);
}
.gcard{
  background: var(--glass-2); border:1px solid var(--stroke); border-radius:18px; padding:18px;
  backdrop-filter: blur(12px) saturate(140%); -webkit-backdrop-filter: blur(12px) saturate(140%);
  box-shadow: 0 12px 36px rgba(0,0,0,.38);
}
.hero{
  position: relative; overflow:hidden;
  background: linear-gradient(145deg, rgba(8,12,28,.6), rgba(30,35,60,.35));
  border:1px solid var(--stroke); border-radius:24px; padding:26px 22px;
}
.hero h1{ margin:0 0 8px; font-size:34px; }
.hero p{ margin:0; color:var(--muted); }

/* nebula canvas inside hero */
.hero .nebula { position:absolute; inset:0; z-index:-1; opacity:.7; filter: blur(0px); }

/* CTA tiles */
.cta{
  display:flex; gap:16px; flex-wrap:wrap; margin-top:18px;
}
.cta .tile{
  flex: 1 1 300px;
  padding:16px; border-radius:16px; border:1px solid var(--stroke-2);
  background: linear-gradient(180deg, rgba(25,32,60,.45), rgba(18,24,45,.35));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 12px 26px rgba(0,0,0,.30);
}
.cta .title{ font-weight:900; font-size:18px; margin-bottom:6px; }
.cta .sub{ color:var(--muted); font-size:.95rem; }
.cta .btn-wrap{ margin-top:12px; }

.stButton>button{
  width:100%; color:#fff !important;
  background:linear-gradient(180deg, #2b3b6a 0%, #182341 90%) !important;
  border:1px solid #314066 !important;
  border-radius:14px !important;
  padding:10px 14px !important;
  font-weight:800 !important;
  box-shadow:0 8px 22px rgba(0,0,0,.25);
}
.stButton>button:hover{ transform: translateY(-1px); box-shadow:0 12px 28px rgba(0,0,0,.30); }

/* Tabs */
.stTabs [role="tablist"] { gap:8px; }
.stTabs [role="tab"]{
  padding:6px 10px;
  background:rgba(255,255,255,.04);
  border:1px solid var(--stroke) !important;
  border-radius:10px !important;
}
.stTabs [aria-selected="true"]{
  background:#16203a !important; border-color:#223055 !important; color:#eaf0ff !important; font-weight:800;
}

/* KPI risk chips */
.risk-badge{ display:inline-block; padding:.2rem .55rem; border-radius:.6rem; font-weight:800; font-size:.75rem }
.risk-low { background:#10b98120;color:#10b981;border:1px solid #10b98150; }
.risk-med { background:#f59e0b20;color:#f59e0b;border:1px solid #f59e0b50; }
.risk-high{ background:#ef444420;color:#ef4444;border:1px solid #ef444450; }

/* Auth layout */
.auth-wrap { min-height: calc(100vh - 120px); display: grid; place-items: center; }
.auth-card { width:100%; max-width:520px; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# tiny SVG background (doesn't push content)
components.html("<div style='position:fixed;inset:0;z-index:-1;pointer-events:none'></div>", height=1)

PLOTLY_TEMPLATE = "plotly_dark"

# ------------------------- Optional project modules ------------------------- #
try:
    from scoring import calculate_news2, calculate_qsofa, calculate_sirs, interpret_risk
except Exception:
    calculate_news2 = calculate_qsofa = calculate_sirs = interpret_risk = None

try:
    from rag_system import SepsisRAG
except Exception:
    SepsisRAG = None

# ------------------------- State & Navigation ------------------------- #
def init_state():
    st.session_state.setdefault("route", "AUTH")
    st.session_state.setdefault("auth_user", None)
    st.session_state.setdefault("nav_stack", [])  # for Back
    st.session_state.setdefault("timeline", [])
    st.session_state.setdefault("scores", None)
    st.session_state.setdefault("vitals", {
        "temperature": 37.0, "heart_rate": 80, "respiratory_rate": 16,
        "systolic_bp": 120, "spo2": 98, "consciousness": "Alert", "wbc": 0
    })
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("profile", {
        "username": "", "patient_name": "", "mrn": "", "age": 0, "sex": "Male",
        "height_cm": 0.0, "weight_kg": 0.0, "allergies": "", "comorbidities": "",
        "chief_complaint": "", "current_meds": "", "infection_source": "",
        "code_status": "Full Code", "notes": "", "attachments": []
    })
init_state()

def goto(route:str):
    # push current route to stack and go
    if st.session_state.route != route:
        st.session_state.nav_stack.append(st.session_state.route)
        st.session_state.route = route
        st.rerun()

def go_back():
    if st.session_state.nav_stack:
        st.session_state.route = st.session_state.nav_stack.pop()
    else:
        st.session_state.route = "HOME"
    st.rerun()

# ------------------------- Helpers ------------------------- #
@st.cache_resource
def load_rag_system():
    if SepsisRAG is None:
        raise RuntimeError("RAG backend not available (rag_system.py missing)")
    return SepsisRAG()
>>>>>>> 13f805555b3536fde72e37e09b2bb701e594ca63

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #212529;
    }
    
    .metric-subtitle {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.25rem;
    }
    
    .analysis-section {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 1.5rem;
    }
    
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e9ecef;
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #212529;
    }
    
    .recommended-action {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    
    .score-high {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    
    .score-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    
    .score-low {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    
    .sidebar-section {
        margin-bottom: 2rem;
    }
    
    .sidebar-title {
        font-weight: 600;
        margin-bottom: 1rem;
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

<<<<<<< HEAD
# Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <h2 style="margin: 0; color: #28a745;">🩺 Sepsis Early Prediction</h2>
            <span style="color: #6c757d;">Real-time Patient Monitoring & Risk Scoring</span>
        </div>
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span style="color: #6c757d;">☀️ Refresh</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">NAVIGATION</div>', unsafe_allow_html=True)
    
    nav_options = [
        "📊 Patient Dashboard",
        "🧪 Vitals Entry", 
        "🧮 Risk Scores",
        "🤖 AI Assistant",
        "📤 Upload Patient Data"
    ]
    
    selected_nav = st.selectbox("", nav_options, index=2, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">FILTERS</div>', unsafe_allow_html=True)
    admitted_only = st.checkbox("🏥 Show Admitted Patients", value=True)
    critical_alerts = st.checkbox("⚠️ Show Critical Alerts", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Total Patients Screened</div>
        <div class="metric-value">120</div>
        <div class="metric-subtitle">Based on uploaded vitals</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">High Risk Alerts</div>
        <div class="metric-value">28</div>
        <div class="metric-subtitle">Patients flagged urgent</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Analysis Status</div>
        <div class="metric-value">Complete</div>
    </div>
    """, unsafe_allow_html=True)

# Helper function to format scores
def format_score(score, score_type="percentage"):
    if score_type == "percentage":
        if score >= 90:
            return f'<span class="score-high">{score}%</span>'
        elif score >= 70:
            return f'<span class="score-medium">{score}%</span>'
        else:
            return f'<span class="score-low">{score}%</span>'
    else:  # Overall score
        if score >= 85:
            return f'<span class="score-high">{score}%</span>'
        elif score >= 70:
            return f'<span class="score-medium">{score}%</span>'
        else:
            return f'<span class="score-low">{score}%</span>'

# Cluster 19 Analysis
st.markdown("""
<div class="analysis-section">
    <div class="section-header">
        <div class="section-title">🔍 Cluster 19 Analysis</div>
        <div class="recommended-action">Recommended Action: 1</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Create sample data for Cluster 19
cluster19_data = {
    'Index 1': [19, 19, 527],
    'Provider 1': ['Michael Thompson, DO', 'Michael Thompson, DO', 'Michael N Thompson, DO'],
    'Index 2': [532, 529, 528],
    'Provider 2': ['Michael N Thompson, DO', 'Mike Thompson, DO', 'Mike Thompson, DO'],
    'Overall Score': [94.7, 87.6, 94.5],
    'Name Score': [100.0, 100.0, 100.0],
    'NPI Match': ['🔺', '🔺', '🔺'],
    'Address Score': [100.0, 100.0, 100.0],
    'Phone Match': ['✅', '✅', '✅'],
    'License Score': [100.0, 100.0, 100.0]
}

df19 = pd.DataFrame(cluster19_data)

# Display table with custom formatting
st.markdown("**Cluster 19 Analysis Results:**")
for idx, row in df19.iterrows():
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([1, 3, 1, 3, 2, 2, 1.5, 2, 1.5, 2])
    
    with col1:
        st.write(row['Index 1'])
    with col2:
        st.write(row['Provider 1'])
    with col3:
        st.write(row['Index 2'])
    with col4:
        st.write(row['Provider 2'])
    with col5:
        st.markdown(format_score(row['Overall Score'], "overall"), unsafe_allow_html=True)
    with col6:
        st.markdown(format_score(row['Name Score']), unsafe_allow_html=True)
    with col7:
        st.write(row['NPI Match'])
    with col8:
        st.markdown(format_score(row['Address Score']), unsafe_allow_html=True)
    with col9:
        st.write(row['Phone Match'])
    with col10:
        st.markdown(format_score(row['License Score']), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Cluster 3 Analysis
st.markdown("""
<div class="analysis-section">
    <div class="section-header">
        <div class="section-title">🔍 Cluster 3 Analysis</div>
        <div class="recommended-action">Recommended Action: 2</div>
    </div>
</div>
""", unsafe_allow_html=True)

cluster3_data = {
    'Index 1': [3],
    'Provider 1': ['Joseph Chen, MD PhD'],
    'Index 2': [563],
    'Provider 2': ['Joseph D Chen, MD PhD'],
    'Overall Score': [84.3],
    'Name Score': [100.0],
    'NPI Match': ['🔺'],
    'Address Score': [100.0],
    'Phone Match': ['✅'],
    'License Score': [100.0]
}

df3 = pd.DataFrame(cluster3_data)

st.markdown("**Cluster 3 Analysis Results:**")
for idx, row in df3.iterrows():
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([1, 3, 1, 3, 2, 2, 1.5, 2, 1.5, 2])
    
    with col1:
        st.write(row['Index 1'])
    with col2:
        st.write(row['Provider 1'])
    with col3:
        st.write(row['Index 2'])
    with col4:
        st.write(row['Provider 2'])
    with col5:
        st.markdown(format_score(row['Overall Score'], "overall"), unsafe_allow_html=True)
    with col6:
        st.markdown(format_score(row['Name Score']), unsafe_allow_html=True)
    with col7:
        st.write(row['NPI Match'])
    with col8:
        st.markdown(format_score(row['Address Score']), unsafe_allow_html=True)
    with col9:
        st.write(row['Phone Match'])
    with col10:
        st.markdown(format_score(row['License Score']), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Cluster 7 Analysis
st.markdown("""
<div class="analysis-section">
    <div class="section-header">
        <div class="section-title">🔍 Cluster 7 Analysis</div>
        <div class="recommended-action">Recommended Action: 2</div>
    </div>
</div>
""", unsafe_allow_html=True)

cluster7_data = {
    'Index 1': [7],
    'Provider 1': ['Christopher Gonzalez, DO PhD'],
    'Index 2': [568],
    'Provider 2': ['Christopher N Gonzalez, DO PhD'],
    'Overall Score': [92.1],
    'Name Score': [100.0],
    'NPI Match': ['🔺'],
    'Address Score': [100.0],
    'Phone Match': ['✅'],
    'License Score': [100.0]
}

df7 = pd.DataFrame(cluster7_data)

st.markdown("**Cluster 7 Analysis Results:**")
for idx, row in df7.iterrows():
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([1, 3, 1, 3, 2, 2, 1.5, 2, 1.5, 2])
    
    with col1:
        st.write(row['Index 1'])
    with col2:
        st.write(row['Provider 1'])
    with col3:
        st.write(row['Index 2'])
    with col4:
        st.write(row['Provider 2'])
    with col5:
        st.markdown(format_score(row['Overall Score'], "overall"), unsafe_allow_html=True)
    with col6:
        st.markdown(format_score(row['Name Score']), unsafe_allow_html=True)
    with col7:
        st.write(row['NPI Match'])
    with col8:
        st.markdown(format_score(row['Address Score']), unsafe_allow_html=True)
    with col9:
        st.write(row['Phone Match'])
    with col10:
        st.markdown(format_score(row['License Score']), unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("*HILabs Healthcare Analytics Dashboard - Provider Deduplication Analysis*")
=======
def predict_cluster(vitals, kmeans, scaler):
    if not kmeans or not scaler: return "N/A"
    X = scaler.transform([[
        vitals["temperature"], vitals["heart_rate"], vitals["respiratory_rate"],
        vitals["systolic_bp"], vitals["spo2"], vitals.get("wbc", 0)
    ]])
    return int(kmeans.predict(X)[0])

def risk_badge(label: str):
    l = (label or "").lower(); cls = "risk-low"
    if "mod" in l or "medium" in l: cls = "risk-med"
    if "high" in l or "severe" in l or "critical" in l: cls = "risk-high"
    return f'<span class="risk-badge {cls}">{label}</span>'

# Fallback scoring if scoring.py is missing
def fb_qsofa(v):
    s=0
    if (v.get('respiratory_rate') or 0) >= 22: s+=1
    if (v.get('systolic_bp') or 999) <= 100: s+=1
    if (str(v.get('consciousness','Alert')).lower() != 'alert'): s+=1
    return s
def fb_sirs(v):
    s=0; t=v.get('temperature'); hr=v.get('heart_rate'); rr=v.get('respiratory_rate'); w=v.get('wbc')
    if t is not None and (t>38 or t<36): s+=1
    if hr is not None and hr>90: s+=1
    if rr is not None and rr>20: s+=1
    if w is not None and (w>12 or w<4): s+=1
    return s
def fb_news2(v):
    s=0
    rr=v.get('respiratory_rate',18) or 18; spo2=v.get('spo2',98) or 98; sbp=v.get('systolic_bp',120) or 120
    hr=v.get('heart_rate',80) or 80; temp=v.get('temperature'); cons=str(v.get('consciousness','Alert')).lower()
    if rr<=8: s+=3
    elif 9<=rr<=11: s+=1
    elif 21<=rr<=24: s+=2
    elif rr>=25: s+=3
    if spo2<=91: s+=3
    elif 92<=spo2<=93: s+=2
    elif spo2 in (94,95): s+=1
    if sbp<=90: s+=3
    elif 91<=sbp<=100: s+=2
    elif 101<=sbp<=110: s+=1
    if hr<=40 or hr>=131: s+=3
    elif 111<=hr<=130: s+=2
    elif 91<=hr<=110: s+=1
    if temp is not None:
        if temp<=35.0: s+=3
        elif temp>=39.1: s+=2
        elif 35.1<=temp<=36.0: s+=1
    if cons!='alert': s+=3
    return s
def fb_interpret(news2,qsofa,sirs):
    if news2>=7 or qsofa>=2: return ("High","#ff4b4b","Urgent clinical escalation required. Consider sepsis bundle.")
    if news2>=5:            return ("Medium","#ffc700","Urgent review by clinician. Monitor closely.")
    if news2>=1 or sirs>=2: return ("Low-Medium","#2b9aff","Requires assessment and monitoring.")
    return ("Low","#00c48c","Continue routine monitoring")

def compute_scores(vitals):
    if calculate_news2 and calculate_qsofa and calculate_sirs:
        n_s, n_r = calculate_news2(vitals)
        q_s, q_r = calculate_qsofa(vitals)
        s_s, s_r = calculate_sirs(vitals)
    else:
        n_s, q_s, s_s = fb_news2(vitals), fb_qsofa(vitals), fb_sirs(vitals)
        n_r, q_r, s_r = "—","—","—"
    kmeans, scaler = load_kmeans_and_scaler()
    cl = predict_cluster(vitals, kmeans, scaler)
    st.session_state.scores = {
        "news2": (n_s, n_r), "qsofa": (q_s, q_r), "sirs": (s_s, s_r),
        "cluster": cl, "vitals": vitals, "ts": time.time()
    }

# ------------------------- Shared Top Bar ------------------------- #
def topbar(show_back:bool=True):
    c1, c2, c3 = st.columns([1,6,1])
    with c1:
        if show_back and st.button("← Back", key=f"back_{st.session_state.route}"):
            go_back()
    with c2:
        st.markdown("<div style='text-align:center' class='small'>", unsafe_allow_html=True)
        st.markdown("**Sepsis RAG Assistant**")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        col = st.container()
        with col:
            if st.button("🏠 Home", key=f"home_{st.session_state.route}"):
                goto("HOME")

# ------------------------- Pages ------------------------- #
def page_auth():
    st.markdown("<div class='auth-wrap'><div class='gcard auth-card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;margin:0 0 6px;'>Welcome to Sepsis RAG Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<div class='small' style='text-align:center;margin-bottom:10px'>Sign in or create an account</div>", unsafe_allow_html=True)
    tabs = st.tabs(["Login", "Register"])
    with tabs[0]:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            if u and p:
                st.session_state.auth_user = u
                goto("HOME")
            else:
                st.error("Enter both username and password.")
    with tabs[1]:
        u2 = st.text_input("Create username", key="reg_user")
        p2 = st.text_input("Create password", type="password", key="reg_pass")
        if st.button("Create account", key="reg_btn"):
            if u2 and p2:
                st.session_state.auth_user = u2
                goto("HOME")
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
    components.html(nebula_svg(), height=0)  # inject (absolute) without spacing
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
        if st.button("Open Vitals", key="home_to_vitals"):
            goto("VITALS")
        st.markdown("</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>📊 View Risk Scores</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub'>NEWS2, qSOFA, SIRS, Cluster.</div>", unsafe_allow_html=True)
        st.markdown("<div class='btn-wrap'>", unsafe_allow_html=True)
        if st.button("Go to Dashboard", key="home_to_dash"):
            goto("VITALS")
        st.markdown("</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='tile'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>💬 Ask Clinical Questions</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub'>Get guideline-grounded answers.</div>", unsafe_allow_html=True)
        st.markdown("<div class='btn-wrap'>", unsafe_allow_html=True)
        if st.button("Open Chatbot", key="home_to_chat"):
            goto("CHAT")
        st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # end cta
    st.markdown("</div>", unsafe_allow_html=True)  # end hero

    # Footer actions
    col_l, col_c, col_r = st.columns([1,6,1])
    with col_l:
        if st.button("Logout", key="logout_btn"):
            st.session_state.auth_user = None
            st.session_state.nav_stack.clear()
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

# ------------------------- Router (never blank) ------------------------- #
valid = {"AUTH","HOME","VITALS","CHAT"}
if st.session_state.route not in valid:
    st.session_state.route = "AUTH"

if st.session_state.route == "AUTH":
    page_auth()
elif st.session_state.route == "HOME":
    page_home()
elif st.session_state.route == "VITALS":
    page_vitals_tabs()
elif st.session_state.route == "CHAT":
    page_chat()
else:
    st.session_state.route = "AUTH"
    st.rerun()
// added login page
// asadd okay
>>>>>>> 13f805555b3536fde72e37e09b2bb701e594ca63
