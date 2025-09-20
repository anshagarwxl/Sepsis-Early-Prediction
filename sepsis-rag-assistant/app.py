"""
Sepsis RAG Assistant - Elegant Clinical Decision Support
Enhanced with Apple-inspired design principles
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import json
from typing import Dict, Optional
import os

# Page config with enhanced theme
st.set_page_config(
    page_title="Sepsis Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try RAG import
try:
    from rag_system import SepsisRAG
    RAG_AVAILABLE = True
except:
    RAG_AVAILABLE = True  # Force available, handle errors gracefully

# Schema
SCHEMA = [
    "Timestamp", "Temperature", "HeartRate", "RespiratoryRate", 
    "SystolicBP", "DiastolicBP", "SpO2", "WBC", "GCS", 
    "Consciousness", "Notes", "NEWS2", "NEWS2_Risk", 
    "qSOFA", "qSOFA_Risk", "SIRS", "SIRS_Risk", 
    "ML_Prediction", "ML_Prob", "ML_Risk"
]

# Apple-Inspired Design System CSS
GLASS_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    :root {
        /* Apple-inspired color system */
        --bg-primary: #000000;
        --bg-secondary: #1c1c1e;
        --bg-tertiary: #2c2c2e;
        --bg-quaternary: #3a3a3c;
        
        /* Glass morphism */
        --glass-primary: rgba(255,255,255,0.05);
        --glass-secondary: rgba(255,255,255,0.03);
        --glass-hover: rgba(255,255,255,0.08);
        
        /* Borders */
        --border-primary: rgba(255,255,255,0.1);
        --border-secondary: rgba(255,255,255,0.05);
        --border-accent: rgba(0,122,255,0.3);
        
        /* Text */
        --text-primary: rgba(255,255,255,0.98);
        --text-secondary: rgba(255,255,255,0.7);
        --text-tertiary: rgba(255,255,255,0.5);
        
        /* Apple system colors */
        --blue: #007AFF;
        --green: #30D158;
        --orange: #FF9F0A;
        --red: #FF453A;
        --purple: #BF5AF2;
        --teal: #40C8E0;
        
        /* Gradients */
        --gradient-primary: linear-gradient(135deg, var(--blue) 0%, var(--purple) 100%);
        --gradient-success: linear-gradient(135deg, var(--green) 0%, var(--teal) 100%);
        --gradient-warning: linear-gradient(135deg, var(--orange) 0%, var(--red) 100%);
        
        /* Shadows */
        --shadow-small: 0 1px 3px rgba(0,0,0,0.12);
        --shadow-medium: 0 4px 12px rgba(0,0,0,0.15);
        --shadow-large: 0 8px 32px rgba(0,0,0,0.25);
        --shadow-xl: 0 16px 64px rgba(0,0,0,0.4);
        
        /* Border radius */
        --radius-small: 8px;
        --radius-medium: 12px;
        --radius-large: 16px;
        --radius-xl: 24px;
    }
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .main {
        background: var(--bg-primary);
        padding: 0;
    }
    
    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        color: var(--text-primary);
    }
    
    /* Remove default padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Hero Section */
    .hero-section {
        background: var(--glass-primary);
        backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-xl);
        padding: 3rem 2rem;
        margin: 0 0 3rem 0;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: var(--gradient-primary);
        opacity: 0.8;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.025em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: var(--text-secondary);
        margin: 1rem 0 0 0;
        font-weight: 400;
        letter-spacing: -0.01em;
    }
    
    .hero-status {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--glass-secondary);
        border: 1px solid var(--border-secondary);
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin-top: 1.5rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Navigation Pills */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin: 0 0 3rem 0;
        padding: 0.5rem;
        background: var(--glass-primary);
        border-radius: var(--radius-large);
        border: 1px solid var(--border-primary);
        backdrop-filter: blur(20px);
    }
    
    /* Card System */
    .card {
        background: var(--glass-primary);
        backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-large);
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .card:hover {
        background: var(--glass-hover);
        border-color: var(--border-accent);
        transform: translateY(-1px);
        box-shadow: var(--shadow-medium);
    }
    
    .card-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 1rem 0;
        letter-spacing: -0.01em;
    }
    
    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .kpi-card {
        background: var(--glass-primary);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-medium);
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-primary);
        opacity: 0.6;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: var(--border-accent);
        box-shadow: var(--shadow-medium);
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    
    .kpi-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin: 0.5rem 0 0 0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Risk Pills */
    .risk-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(8px);
        margin-top: 0.5rem;
    }
    
    .risk-high {
        background: linear-gradient(135deg, var(--red) 0%, #FF6B6B 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(255, 69, 58, 0.3);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, var(--orange) 0%, #FFB366 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(255, 159, 10, 0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, var(--green) 0%, #66D9A3 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(48, 209, 88, 0.3);
    }
    
    /* Form Styling */
    .form-section {
        background: var(--glass-primary);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-large);
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .form-group {
        margin-bottom: 1.5rem;
    }
    
    .form-group-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 1rem 0;
        letter-spacing: -0.01em;
    }
    
    /* Input Styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        background: var(--glass-secondary) !important;
        border: 1px solid var(--border-secondary) !important;
        border-radius: var(--radius-small) !important;
        color: var(--text-primary) !important;
        backdrop-filter: blur(8px);
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--border-accent) !important;
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: var(--gradient-primary) !important;
        border: none !important;
        border-radius: var(--radius-medium) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-small) !important;
        letter-spacing: -0.01em !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: var(--shadow-medium) !important;
        filter: brightness(1.1) !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-primary) !important;
    }
    
    /* Chat Styling */
    .chat-message {
        background: var(--glass-primary);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-medium);
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .chat-user {
        background: var(--glass-hover);
        border-color: var(--border-accent);
    }
    
    /* Alert Styling */
    .alert-critical {
        background: linear-gradient(135deg, rgba(255, 69, 58, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        border: 1px solid var(--red);
        border-radius: var(--radius-medium);
        padding: 1.5rem;
        margin: 1rem 0;
        color: var(--text-primary);
    }
    
    .alert-success {
        background: linear-gradient(135deg, rgba(48, 209, 88, 0.1) 0%, rgba(102, 217, 163, 0.1) 100%);
        border: 1px solid var(--green);
        border-radius: var(--radius-medium);
        padding: 1.5rem;
        margin: 1rem 0;
        color: var(--text-primary);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .kpi-grid {
            grid-template-columns: 1fr 1fr;
        }
        
        .form-section {
            padding: 1rem;
        }
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Loading states */
    .loading {
        opacity: 0.6;
        pointer-events: none;
    }
</style>
"""

# Apply the CSS
st.markdown(GLASS_CSS, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame(columns=SCHEMA)
    if "vitals" not in st.session_state:
        st.session_state.vitals = {
            "temperature": 37.0, "heart_rate": 80, "respiratory_rate": 16,
            "systolic_bp": 120, "diastolic_bp": 80, "spo2": 98,
            "consciousness": "Alert", "wbc": 8.0, "gcs": 15
        }
    if "scores" not in st.session_state:
        st.session_state.scores = None
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "current_view" not in st.session_state:
        st.session_state.current_view = "overview"

# Initialize RAG system
def get_rag_system():
    if st.session_state.rag_system is None:
        try:
            # Use hardcoded API key since dotenv is not available
            api_key = "AIzaSyAu_8lYWrc4UreqRfvSX5eXMXJ1f17W2Xw"
            st.session_state.rag_system = SepsisRAG(gemini_api_key=api_key)
            return st.session_state.rag_system
        except Exception as e:
            # Return None if RAG fails to initialize
            st.session_state.rag_system = None
            return None
    return st.session_state.rag_system

# Clinical scoring functions
def calculate_news2(vitals: Dict) -> tuple:
    """Calculate NEWS2 score"""
    score = 0
    
    # Temperature
    temp = vitals.get("temperature", 37.0)
    if temp <= 35.0: score += 3
    elif temp <= 36.0: score += 1
    elif temp >= 39.1: score += 2
    elif temp >= 38.1: score += 1
    
    # Heart rate
    hr = vitals.get("heart_rate", 80)
    if hr <= 40: score += 3
    elif hr <= 50: score += 1
    elif hr >= 131: score += 3
    elif hr >= 111: score += 2
    elif hr >= 91: score += 1
    
    # Respiratory rate
    rr = vitals.get("respiratory_rate", 16)
    if rr <= 8: score += 3
    elif rr <= 11: score += 1
    elif rr >= 25: score += 3
    elif rr >= 21: score += 2
    
    # SpO2
    spo2 = vitals.get("spo2", 98)
    if spo2 <= 91: score += 3
    elif spo2 <= 93: score += 2
    elif spo2 <= 95: score += 1
    
    # Systolic BP
    sbp = vitals.get("systolic_bp", 120)
    if sbp <= 90: score += 3
    elif sbp <= 100: score += 2
    elif sbp <= 110: score += 1
    elif sbp >= 220: score += 3
    
    # Consciousness
    consciousness = vitals.get("consciousness", "Alert")
    if consciousness != "Alert": score += 3
    
    # Risk level
    if score >= 7: risk = "High"
    elif score >= 5: risk = "Medium"
    else: risk = "Low"
    
    return score, risk

def calculate_qsofa(vitals: Dict) -> tuple:
    """Calculate qSOFA score"""
    score = 0
    
    # Respiratory rate >= 22
    if vitals.get("respiratory_rate", 16) >= 22: score += 1
    
    # Systolic BP <= 100
    if vitals.get("systolic_bp", 120) <= 100: score += 1
    
    # Altered mental status
    if vitals.get("consciousness", "Alert") != "Alert": score += 1
    
    risk = "High" if score >= 2 else "Low"
    return score, risk

def calculate_sirs(vitals: Dict) -> tuple:
    """Calculate SIRS score"""
    score = 0
    
    # Temperature
    temp = vitals.get("temperature", 37.0)
    if temp > 38.0 or temp < 36.0: score += 1
    
    # Heart rate > 90
    if vitals.get("heart_rate", 80) > 90: score += 1
    
    # Respiratory rate > 20
    if vitals.get("respiratory_rate", 16) > 20: score += 1
    
    # WBC
    wbc = vitals.get("wbc", 8.0)
    if wbc and (wbc > 12.0 or wbc < 4.0): score += 1
    
    if score >= 3: risk = "High"
    elif score >= 2: risk = "Medium"
    else: risk = "Low"
    
    return score, risk

def compute_all_scores(vitals: Dict) -> Dict:
    """Compute all clinical scores"""
    news2_score, news2_risk = calculate_news2(vitals)
    qsofa_score, qsofa_risk = calculate_qsofa(vitals)
    sirs_score, sirs_risk = calculate_sirs(vitals)
    
    return {
        "NEWS2": news2_score,
        "NEWS2_Risk": news2_risk,
        "qSOFA": qsofa_score,
        "qSOFA_Risk": qsofa_risk,
        "SIRS": sirs_score,
        "SIRS_Risk": sirs_risk,
        "ML_Prediction": None,
        "ML_Prob": None,
        "ML_Risk": None
    }

def pill(text: str) -> str:
    """Create a colored pill for risk levels"""
    color_class = "pill-low"
    if "High" in text: color_class = "pill-high"
    elif "Medium" in text: color_class = "pill-medium"
    return f'<span class="pill {color_class}">{text}</span>'

# Component functions
def render_kpi_tile(title, value, risk_level=None, description=None):
    """Create an enhanced KPI tile using Streamlit containers"""
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{title}**")
            st.metric("", value)
            if description:
                st.caption(description)
        
        with col2:
            if risk_level:
                risk_class = f"risk-{risk_level.lower()}"
                st.markdown(f'<div class="risk-pill {risk_class}">{risk_level}</div>', unsafe_allow_html=True)

def render_overview_gauges(scores_dict):
    """Render all overview gauges in a clean layout"""
    if not scores_dict:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_kpi_tile("NEWS2", scores_dict["NEWS2"], scores_dict["NEWS2_Risk"], "National Early Warning Score")
    
    with col2:
        render_kpi_tile("qSOFA", scores_dict["qSOFA"], scores_dict["qSOFA_Risk"], "Quick Sequential Organ Failure")
    
    with col3:
        render_kpi_tile("SIRS", scores_dict["SIRS"], scores_dict["SIRS_Risk"], "Systemic Inflammatory Response")
    
    with col4:
        risk_score = (scores_dict["NEWS2"] * 2 + scores_dict["qSOFA"] * 3 + scores_dict["SIRS"] * 2) / 7 * 100
        overall_risk = "High" if risk_score > 70 else "Medium" if risk_score > 40 else "Low"
        render_kpi_tile("Overall Risk", f"{risk_score:.0f}%", overall_risk, "Composite Risk Assessment")

def render_clinical_interpretation(scores_dict):
    """Render clinical interpretation section"""
    with st.container(border=True):
        st.markdown("### 🔍 Clinical Interpretation")
        
        high_scores = [k for k, v in scores_dict.items() if "Risk" in k and v == "High"]
        if high_scores:
            st.error("🚨 **HIGH RISK SEPSIS DETECTED** - Immediate intervention required")
            
            st.markdown("**🚨 Immediate Actions Required:**")
            action_items = [
                "Obtain blood cultures before antibiotics",
                "Start empiric antibiotics within 1 hour", 
                "Administer IV fluid resuscitation (30ml/kg)",
                "Monitor lactate and blood pressure",
                "Consider ICU consultation",
                "Reassess in 30 minutes"
            ]
            
            for item in action_items:
                st.markdown(f"• {item}")
        else:
            st.success("✅ **Low to moderate risk** - Continue standard monitoring")

def render_patient_form():
    """Render the patient assessment form"""
    with st.form("vitals_form", clear_on_submit=False):
        # Primary Vitals Section
        st.markdown("#### 🌡️ Primary Vital Signs")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Temperature & Pulse**")
            temperature = st.number_input("Temperature (°C)", 30.0, 45.0, 
                                        st.session_state.vitals["temperature"], 0.1)
            heart_rate = st.number_input("Heart Rate (bpm)", 30, 250, 
                                       int(st.session_state.vitals["heart_rate"]), 1)
        
        with col2:
            st.markdown("**Respiratory & Oxygen**")
            respiratory_rate = st.number_input("Respiratory Rate (/min)", 5, 60, 
                                             int(st.session_state.vitals["respiratory_rate"]), 1)
            spo2 = st.number_input("SpO₂ (%)", 70, 100, 
                                 int(st.session_state.vitals["spo2"]), 1)
        
        with col3:
            st.markdown("**Blood Pressure**")
            systolic_bp = st.number_input("Systolic BP (mmHg)", 60, 300, 
                                        int(st.session_state.vitals["systolic_bp"]), 1)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", 40, 200, 
                                         int(st.session_state.vitals["diastolic_bp"]), 1)
        
        st.markdown("#### 🧪 Laboratory & Neurological")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            wbc = st.number_input("WBC Count (×10³/µL)", 0.0, 50.0, 
                                float(st.session_state.vitals["wbc"]), 0.1)
        
        with col2:
            gcs = st.number_input("Glasgow Coma Scale", 3, 15, 
                                int(st.session_state.vitals["gcs"]), 1)
        
        with col3:
            consciousness = st.selectbox("Consciousness Level (AVPU)", 
                                       ["Alert", "Voice", "Pain", "Unresponsive"],
                                       index=["Alert", "Voice", "Pain", "Unresponsive"].index(
                                           st.session_state.vitals["consciousness"]))
        
        # Clinical Notes
        st.markdown("#### 📝 Clinical Notes")
        notes = st.text_area("Additional observations, symptoms, or clinical context", 
                            height=100, 
                            placeholder="e.g., Patient reports fever and chills for 2 days, suspected UTI...")
        
        # Submit Button
        if st.form_submit_button("🔬 Calculate Risk Scores", use_container_width=True):
            process_patient_form(temperature, heart_rate, respiratory_rate, systolic_bp, 
                                diastolic_bp, spo2, wbc, gcs, consciousness, notes)

def process_patient_form(temperature, heart_rate, respiratory_rate, systolic_bp, 
                        diastolic_bp, spo2, wbc, gcs, consciousness, notes):
    """Process patient form submission"""
    # Update vitals
    vitals = {
        "temperature": temperature,
        "heart_rate": heart_rate,
        "respiratory_rate": respiratory_rate,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "spo2": spo2,
        "wbc": wbc,
        "gcs": gcs,
        "consciousness": consciousness
    }
    
    # Calculate scores using preserved core logic
    scores = compute_all_scores(vitals)
    
    # Create record
    record = {
        "Timestamp": dt.datetime.now(),
        "Temperature": temperature,
        "HeartRate": heart_rate,
        "RespiratoryRate": respiratory_rate,
        "SystolicBP": systolic_bp,
        "DiastolicBP": diastolic_bp,
        "SpO2": spo2,
        "WBC": wbc,
        "GCS": gcs,
        "Consciousness": consciousness,
        "Notes": notes
    }
    record.update(scores)
    
    # Save to session
    st.session_state.vitals = vitals
    st.session_state.scores = scores
    
    # Add to dataframe
    new_df = pd.DataFrame([record], columns=SCHEMA)
    st.session_state.data = pd.concat([st.session_state.data, new_df], ignore_index=True)
    
    st.success("✅ Risk scores calculated successfully! View results in the Overview tab.")
    st.session_state.current_view = "overview"
    st.rerun()

def render_analytics_tab(df):
    """Render the complete analytics dashboard"""
    if df.empty:
        with st.container(border=True):
            st.markdown("### 📊 No Data Available")
            st.info("Enter patient assessments to view analytics and trends")
        return
    
    st.markdown("### 📈 Clinical Analytics Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_kpi_tile("Total Cases", str(len(df)), description="Patients assessed")
    
    with col2:
        high_risk_count = (df["NEWS2_Risk"] == "High").sum()
        risk_level = "High" if high_risk_count > len(df) * 0.3 else "Medium" if high_risk_count > 0 else "Low"
        render_kpi_tile("High Risk", str(high_risk_count), risk_level, "Critical cases")
    
    with col3:
        avg_news2 = df["NEWS2"].mean()
        risk_level = "High" if avg_news2 > 7 else "Medium" if avg_news2 > 5 else "Low"
        render_kpi_tile("Avg NEWS2", f"{avg_news2:.1f}", risk_level, "Population average")
    
    with col4:
        latest_time = df["Timestamp"].max()
        hours_ago = (dt.datetime.now() - latest_time).total_seconds() / 3600
        render_kpi_tile("Last Update", f"{hours_ago:.0f}h ago", description="Most recent case")
    
    # Risk Distribution Chart
    with st.container(border=True):
        st.markdown("### 📊 Risk Distribution")
        risk_counts = df["NEWS2_Risk"].value_counts()
        st.bar_chart(risk_counts)
    
    # Recent Cases Table
    with st.container(border=True):
        st.markdown("### 📋 Recent Cases")
        
        # Display last 10 records with key information
        display_cols = ["Timestamp", "Temperature", "HeartRate", "SystolicBP", "NEWS2", "NEWS2_Risk", "qSOFA_Risk"]
        recent_data = df[display_cols].tail(10).sort_values("Timestamp", ascending=False)
        st.dataframe(recent_data, use_container_width=True)
        
        # Export functionality
        if st.button("📄 Export All Data", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV File", csv, "sepsis_analytics.csv", "text/csv", use_container_width=True)

def render_chat_interface():
    """Render the chat interface"""
    with st.container(border=True):
        if not st.session_state.chat:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: var(--text-secondary);">
                <h4>👋 Hello! I'm your clinical AI assistant</h4>
                <p>Ask me about sepsis diagnosis, treatment protocols, or clinical guidelines.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat messages
        for msg in st.session_state.chat:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message chat-user">
                    <strong>🧑‍⚕️ You:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message">
                    <strong>🤖 Assistant:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)

def render_quick_questions():
    """Render quick question buttons"""
    col1, col2, col3 = st.columns(3)
    
    quick_questions = {
        "💊 Antibiotics": "What are the recommended empiric antibiotics for sepsis?",
        "💧 Fluid Therapy": "How much fluid should I give for septic shock?",
        "🔍 Diagnostics": "What lab tests should I order for suspected sepsis?",
        "📊 qSOFA Criteria": "What are the qSOFA criteria and how do I interpret them?",
        "🚨 Sepsis-3": "What are the updated Sepsis-3 definitions?",
        "⏰ Time Targets": "What are the key time targets for sepsis management?"
    }
    
    questions_list = list(quick_questions.items())
    
    with col1:
        for i in range(0, len(questions_list), 3):
            if i < len(questions_list):
                if st.button(questions_list[i][0], use_container_width=True, key=f"q1_{i}"):
                    handle_question(questions_list[i][1])
    
    with col2:
        for i in range(1, len(questions_list), 3):
            if i < len(questions_list):
                if st.button(questions_list[i][0], use_container_width=True, key=f"q2_{i}"):
                    handle_question(questions_list[i][1])
    
    with col3:
        for i in range(2, len(questions_list), 3):
            if i < len(questions_list):
                if st.button(questions_list[i][0], use_container_width=True, key=f"q3_{i}"):
                    handle_question(questions_list[i][1])

def render_sidebar():
    """Render enhanced sidebar"""
    with st.sidebar:
        st.markdown("### 🩺 Quick Actions")
        
        # Quick preset buttons with better styling
        if st.button("🔴 High Risk Patient", use_container_width=True):
            st.session_state.vitals.update({
                "temperature": 39.2, "heart_rate": 125, "respiratory_rate": 28,
                "systolic_bp": 88, "spo2": 89, "consciousness": "Voice"
            })
            st.rerun()
        
        if st.button("🟡 Medium Risk Patient", use_container_width=True):
            st.session_state.vitals.update({
                "temperature": 38.5, "heart_rate": 105, "respiratory_rate": 22,
                "systolic_bp": 95, "spo2": 94, "consciousness": "Alert"
            })
            st.rerun()
        
        if st.button("🟢 Normal Vitals", use_container_width=True):
            st.session_state.vitals.update({
                "temperature": 37.0, "heart_rate": 75, "respiratory_rate": 16,
                "systolic_bp": 120, "spo2": 98, "consciousness": "Alert"
            })
            st.rerun()
        
        st.markdown("---")
        
        # System Status
        st.markdown("### 🔧 System Status")
        st.success("🤖 RAG System: Online")
        
        if st.session_state.data is not None and not st.session_state.data.empty:
            st.info(f"📊 {len(st.session_state.data)} Records")

def create_hero_section():
    """Create the main hero section"""
    rag_status = "🟢 Available" if RAG_AVAILABLE else "🔴 Unavailable"
    
    st.markdown(f"""
    <div class="hero-section fade-in">
        <h1 class="hero-title">Sepsis Clinical Decision Support</h1>
        <p class="hero-subtitle">Advanced AI-powered risk assessment with real-time clinical guidance</p>
        <div class="hero-status">
            <span>🤖 AI System</span>
            <span>{rag_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_kpi_card(title, value, risk_level=None, description=None):
    """Create an enhanced KPI card"""
    risk_pill = ""
    if risk_level:
        risk_class = f"risk-{risk_level.lower()}"
        risk_pill = f'<div class="risk-pill {risk_class}">{risk_level} Risk</div>'
    
    desc_text = f'<div style="color: var(--text-tertiary); font-size: 0.8rem; margin-top: 0.5rem;">{description}</div>' if description else ""
    
    return f"""
    <div class="kpi-card fade-in">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{title}</div>
        {risk_pill}
        {desc_text}
    </div>
    """

def create_form_section(title, content):
    """Create a form section with proper styling"""
    return f"""
    <div class="form-section">
        <div class="form-group-title">{title}</div>
        {content}
    </div>
    """

def create_alert(message, alert_type="info"):
    """Create styled alerts"""
    alert_class = f"alert-{alert_type}"
    icon = "🚨" if alert_type == "critical" else "✅" if alert_type == "success" else "ℹ️"
    
    return f"""
    <div class="{alert_class}">
        <strong>{icon} {message}</strong>
    </div>
    """

def render_clinical_interpretation(scores):
    """Render clinical interpretation based on scores"""
    with st.container(border=True):
        st.markdown("### 🔍 Clinical Interpretation")
        
        high_scores = [k for k, v in scores.items() if "Risk" in k and v == "High"]
        if high_scores:
            st.markdown(create_alert("HIGH RISK SEPSIS DETECTED - Immediate intervention required", "critical"), 
                       unsafe_allow_html=True)
            
            st.markdown("**🚨 Immediate Actions Required:**")
            action_items = [
                "Obtain blood cultures before antibiotics",
                "Start empiric antibiotics within 1 hour",
                "Administer IV fluid resuscitation (30ml/kg)",
                "Monitor lactate and blood pressure",
                "Consider ICU consultation",
                "Reassess in 30 minutes"
            ]
            
            for item in action_items:
                st.markdown(f"• {item}")
        else:
            st.markdown(create_alert("Low to moderate risk - Continue standard monitoring", "success"), 
                       unsafe_allow_html=True)

# Main app
def main():
    init_session_state()
    
    # Hero Section
    create_hero_section()
    
    # Enhanced Sidebar
    render_sidebar()
    
    # Main Navigation - using a more elegant approach
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Overview", use_container_width=True):
            st.session_state.current_view = "overview"
    
    with col2:
        if st.button("🩺 Patient Assessment", use_container_width=True):
            st.session_state.current_view = "patient"
    
    with col3:
        if st.button("📈 Analytics", use_container_width=True):
            st.session_state.current_view = "analytics"
    
    with col4:
        if st.button("💬 AI Assistant", use_container_width=True):
            st.session_state.current_view = "assistant"
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dynamic Content Based on Current View
    if st.session_state.current_view == "overview":
        render_overview()
    elif st.session_state.current_view == "patient":
        render_patient_assessment()
    elif st.session_state.current_view == "analytics":
        render_analytics()
    elif st.session_state.current_view == "assistant":
        render_ai_assistant()

def render_overview():
    """Render the overview dashboard using components"""
    if st.session_state.scores:
        # KPI Grid using component
        render_overview_gauges(st.session_state.scores)
        
        # Clinical Interpretation using component
        render_clinical_interpretation(st.session_state.scores)
    else:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h3>👋 Welcome to Sepsis Clinical Decision Support</h3>
                <p style="color: var(--text-secondary); margin: 1rem 0;">
                    Enter patient vitals in the <strong>Patient Assessment</strong> tab to begin risk evaluation
                </p>
                <div style="margin-top: 2rem;">
                    <div style="background: var(--glass-primary); border-radius: 12px; padding: 1rem; display: inline-block;">
                        Click <strong>Patient Assessment</strong> to get started
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_patient_assessment():
    """Render the patient assessment form using components"""
    st.markdown("### 🩺 Patient Vital Signs Assessment")
    render_patient_form()

def render_analytics():
    """Render the analytics dashboard using components"""
    render_analytics_tab(st.session_state.data)

def render_ai_assistant():
    """Render the AI assistant using components"""
    # Chat container
    render_chat_interface()
    
    # Chat input - moved up
    st.markdown("### ✍️ Ask a Question")
    question = st.text_area("Type your clinical question here...", 
                           height=100, 
                           placeholder="e.g., What are the indications for vasopressors in septic shock?")
    
    if st.button("🚀 Ask Assistant", use_container_width=True):
        if question.strip():
            handle_question(question)
    
    # Quick question buttons - moved down
    st.markdown("### 🔍 Quick Questions")
    render_quick_questions()

def handle_question(question):
    """Handle user questions and generate responses"""
    st.session_state.chat.append({"role": "user", "content": question})
    
    # Try RAG system
    rag = get_rag_system()
    if rag:
        try:
            with st.spinner("🔍 Searching medical guidelines..."):
                result = rag.query(question)
                if isinstance(result, tuple):
                    answer, sources = result
                else:
                    answer = result
                    sources = []
            
            response = answer
            if sources:
                response += f"\n\n**📚 Sources:** {', '.join(sources[:3])}"
            
            st.session_state.chat.append({"role": "assistant", "content": response})
        except Exception as e:
            # If RAG fails, use fallback response
            response = get_fallback_response(question)
            st.session_state.chat.append({"role": "assistant", "content": response})
    else:
        # RAG not available, use comprehensive fallback
        response = get_fallback_response(question)
        st.session_state.chat.append({"role": "assistant", "content": response})
    
    st.rerun()

def get_fallback_response(question):
    """Provide fallback responses for common questions"""
    question_lower = question.lower()
    
    fallback_responses = {
        "qsofa": "**qSOFA Criteria (Quick Sequential Organ Failure Assessment):**\n\n1. **Respiratory rate** ≥22/min\n2. **Systolic blood pressure** ≤100 mmHg\n3. **Altered mental status** (GCS <15)\n\n**Interpretation:** Score ≥2 suggests high risk of poor outcomes and should prompt consideration of organ dysfunction.",
        
        "antibiotics": "**Empiric Antibiotics for Sepsis:**\n\n• **Goal:** Administer within 1 hour of recognition\n• **Broad-spectrum coverage** recommended initially\n• **Options:** Piperacillin-tazobactam, Ceftriaxone + Metronidazole, or Meropenem\n• **Consider local resistance patterns**\n• **Obtain cultures before antibiotics when possible**",
        
        "fluid": "**Fluid Resuscitation in Sepsis:**\n\n• **Initial:** 30 mL/kg of crystalloids within 3 hours\n• **Reassess frequently** for signs of fluid overload\n• **Monitor:** Blood pressure, urine output, lactate\n• **Avoid fluid overload** - balance resuscitation with organ perfusion",
        
        "news2": "**NEWS2 (National Early Warning Score 2):**\n\nScores: Temperature, Pulse, Respiratory Rate, Oxygen Saturation, Systolic BP, Consciousness, Oxygen Therapy\n\n• **0-4:** Low risk\n• **5-6:** Medium risk\n• **≥7:** High risk requiring urgent medical attention",
        
        "sirs": "**SIRS Criteria (Systemic Inflammatory Response Syndrome):**\n\n1. **Temperature** >38°C or <36°C\n2. **Heart rate** >90 bpm\n3. **Respiratory rate** >20/min\n4. **WBC** >12,000 or <4,000 cells/mm³\n\n**≥2 criteria = SIRS**",
        
        "sepsis-3": "**Sepsis-3 Definitions:**\n\n• **Sepsis:** Life-threatening organ dysfunction due to dysregulated host response to infection\n• **Septic Shock:** Sepsis with circulatory and cellular/metabolic dysfunction\n• **Organ dysfunction:** SOFA score increase ≥2 points",
        
        "time": "**Key Time Targets:**\n\n• **Blood cultures:** Before antibiotics when possible\n• **Antibiotics:** Within 1 hour\n• **Fluid resuscitation:** 30 mL/kg within 3 hours\n• **Lactate measurement:** Within 6 hours\n• **Reassessment:** Every 30 minutes initially"
    }
    
    for key, response in fallback_responses.items():
        if key in question_lower:
            return response
    
    return "I recommend consulting current sepsis guidelines. Key principles include early recognition, prompt antibiotics, adequate fluid resuscitation, and source control. For specific clinical scenarios, please consult with your institution's protocols or infectious disease specialists."

if __name__ == "__main__":
    main()