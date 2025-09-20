"""
Sepsis RAG Assistant - Fast Clinical Decision Support
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Sepsis RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try RAG import
try:
    from rag_system import SepsisRAG
    RAG_AVAILABLE = True
except:
    RAG_AVAILABLE = False

# Schema
SCHEMA = [
    "Timestamp", "Temperature", "HeartRate", "RespiratoryRate", 
    "SystolicBP", "DiastolicBP", "SpO2", "WBC", "GCS", 
    "Consciousness", "Notes", "NEWS2", "NEWS2_Risk", 
    "qSOFA", "qSOFA_Risk", "SIRS", "SIRS_Risk", 
    "ML_Prediction", "ML_Prob", "ML_Risk"
]

# CSS
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #1e1e2e 0%, #2d1b69 100%); }
    .stApp { background: linear-gradient(135deg, #1e1e2e 0%, #2d1b69 100%); }
    .card { 
        background: rgba(255,255,255,0.1); 
        backdrop-filter: blur(10px); 
        border-radius: 15px; 
        padding: 1rem; 
        border: 1px solid rgba(255,255,255,0.2); 
        margin: 0.5rem 0;
    }
    .metric-card {
        background: rgba(20,20,25,0.8);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .pill {
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
    .pill-high { background: #ff6b6b; color: white; }
    .pill-medium { background: #ffcc66; color: black; }
    .pill-low { background: #00d1a7; color: white; }
</style>
""", unsafe_allow_html=True)

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

# Initialize RAG system
def get_rag_system():
    if st.session_state.rag_system is None and RAG_AVAILABLE:
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                st.session_state.rag_system = SepsisRAG(gemini_api_key=api_key)
                return st.session_state.rag_system
        except Exception as e:
            st.error(f"RAG initialization failed: {e}")
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

# Main app
def main():
    init_session_state()
    
    # Header
    st.markdown("# 🏥 Sepsis Clinical Decision Support")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🩺 Quick Actions")
        
        if st.button("🔴 High Risk Preset"):
            st.session_state.vitals.update({
                "temperature": 39.2, "heart_rate": 125, "respiratory_rate": 28,
                "systolic_bp": 88, "spo2": 89, "consciousness": "Voice"
            })
            st.rerun()
        
        if st.button("🟡 Medium Risk Preset"):
            st.session_state.vitals.update({
                "temperature": 38.5, "heart_rate": 105, "respiratory_rate": 22,
                "systolic_bp": 95, "spo2": 94, "consciousness": "Alert"
            })
            st.rerun()
        
        if st.button("🟢 Normal Vitals"):
            st.session_state.vitals.update({
                "temperature": 37.0, "heart_rate": 75, "respiratory_rate": 16,
                "systolic_bp": 120, "spo2": 98, "consciousness": "Alert"
            })
            st.rerun()
        
        st.markdown("---")
        
        # RAG Status
        if RAG_AVAILABLE:
            rag = get_rag_system()
            if rag:
                st.success("🧠 RAG System Ready")
            else:
                st.warning("⚠️ RAG System Loading...")
        else:
            st.error("❌ RAG System Unavailable")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🩺 Patient Entry", "📈 Analytics", "💬 AI Assistant"])
    
    # Overview Tab
    with tab1:
        if st.session_state.scores:
            scores = st.session_state.scores
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("NEWS2", scores["NEWS2"])
                st.markdown(pill(scores["NEWS2_Risk"]), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("qSOFA", scores["qSOFA"])
                st.markdown(pill(scores["qSOFA_Risk"]), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("SIRS", scores["SIRS"])
                st.markdown(pill(scores["SIRS_Risk"]), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                risk_score = (scores["NEWS2"] * 2 + scores["qSOFA"] * 3 + scores["SIRS"] * 2) / 7 * 100
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Overall Risk", f"{risk_score:.0f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk interpretation
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Clinical Interpretation")
            
            high_scores = [k for k, v in scores.items() if "Risk" in k and v == "High"]
            if high_scores:
                st.error(f"⚠️ **HIGH RISK DETECTED** - {', '.join([k.replace('_Risk', '') for k in high_scores])}")
                st.markdown("**Immediate Actions Required:**")
                st.markdown("• Obtain blood cultures")
                st.markdown("• Start empiric antibiotics within 1 hour")
                st.markdown("• Administer IV fluid resuscitation (30ml/kg)")
                st.markdown("• Monitor closely and consider ICU admission")
            else:
                st.success("✅ **Low to moderate risk** - Continue monitoring")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📋 Enter patient vitals to view risk assessment")
    
    # Patient Entry Tab
    with tab2:
        st.markdown("### 🩺 Patient Vital Signs")
        
        with st.form("vitals_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Primary Vitals**")
                temperature = st.number_input("Temperature (°C)", 30.0, 45.0, 
                                            st.session_state.vitals["temperature"], 0.1)
                heart_rate = st.number_input("Heart Rate (bpm)", 30, 250, 
                                           int(st.session_state.vitals["heart_rate"]), 1)
                respiratory_rate = st.number_input("Respiratory Rate (/min)", 5, 60, 
                                                 int(st.session_state.vitals["respiratory_rate"]), 1)
            
            with col2:
                st.markdown("**Blood Pressure & Oxygen**")
                systolic_bp = st.number_input("Systolic BP (mmHg)", 60, 300, 
                                            int(st.session_state.vitals["systolic_bp"]), 1)
                diastolic_bp = st.number_input("Diastolic BP (mmHg)", 40, 200, 
                                             int(st.session_state.vitals["diastolic_bp"]), 1)
                spo2 = st.number_input("SpO₂ (%)", 70, 100, 
                                     int(st.session_state.vitals["spo2"]), 1)
            
            with col3:
                st.markdown("**Additional Parameters**")
                wbc = st.number_input("WBC (×10³/µL)", 0.0, 50.0, 
                                    float(st.session_state.vitals["wbc"]), 0.1)
                gcs = st.number_input("Glasgow Coma Scale", 3, 15, 
                                    int(st.session_state.vitals["gcs"]), 1)
                consciousness = st.selectbox("Consciousness (AVPU)", 
                                           ["Alert", "Voice", "Pain", "Unresponsive"],
                                           index=["Alert", "Voice", "Pain", "Unresponsive"].index(
                                               st.session_state.vitals["consciousness"]))
            
            notes = st.text_area("Clinical Notes", height=80)
            
            if st.form_submit_button("💾 Calculate Scores", use_container_width=True):
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
                
                # Calculate scores
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
                
                st.success("✅ Scores calculated successfully!")
                st.rerun()
    
    # Analytics Tab
    with tab3:
        df = st.session_state.data
        
        if df.empty:
            st.info("📊 No data available. Enter patient vitals first.")
        else:
            st.markdown("### 📈 Clinical Analytics")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                high_risk_count = (df["NEWS2_Risk"] == "High").sum()
                st.metric("High Risk Cases", high_risk_count)
            with col3:
                avg_news2 = df["NEWS2"].mean()
                st.metric("Average NEWS2", f"{avg_news2:.1f}")
            
            # Risk distribution
            st.markdown("**Risk Distribution**")
            risk_counts = df["NEWS2_Risk"].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.4
            )])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data export
            if st.button("📄 Export Data"):
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, "sepsis_data.csv", "text/csv")
    
    # AI Assistant Tab
    with tab4:
        st.markdown("### 💬 Clinical AI Assistant")
        
        # Chat display
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat:
                st.info("💡 Ask questions about sepsis diagnosis, treatment, or guidelines!")
            
            for msg in st.session_state.chat:
                if msg["role"] == "user":
                    st.markdown(f"**🧑‍⚕️ You:** {msg['content']}")
                else:
                    st.markdown(f"**🤖 Assistant:** {msg['content']}")
                st.markdown("---")
        
        # Quick questions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💊 Antibiotics"):
                question = "What are the recommended empiric antibiotics for sepsis?"
        with col2:
            if st.button("💧 Fluid Therapy"):
                question = "How much fluid should I give for septic shock?"
        with col3:
            if st.button("🔍 Diagnostics"):
                question = "What lab tests should I order for suspected sepsis?"
        
        # Chat input
        question = st.text_area("Ask a question:", height=100, 
                               placeholder="e.g., What are the qSOFA criteria?")
        
        if st.button("🚀 Ask Assistant", use_container_width=True):
            if question.strip():
                st.session_state.chat.append({"role": "user", "content": question})
                
                # Try RAG system
                rag = get_rag_system()
                if rag:
                    try:
                        with st.spinner("🔍 Searching medical guidelines..."):
                            answer, sources = rag.query(question)
                        
                        response = answer
                        if sources:
                            response += f"\n\n**📚 Sources:** {', '.join(sources[:3])}"
                        
                        st.session_state.chat.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.session_state.chat.append({
                            "role": "assistant", 
                            "content": f"❌ Sorry, I encountered an error: {str(e)}"
                        })
                else:
                    # Fallback responses
                    fallback_responses = {
                        "qsofa": "qSOFA criteria: 1) Respiratory rate ≥22/min, 2) Systolic BP ≤100 mmHg, 3) Altered mental status. Score ≥2 suggests organ dysfunction.",
                        "antibiotics": "Empiric antibiotics for sepsis: Broad-spectrum coverage recommended within 1 hour. Consider piperacillin-tazobactam, ceftriaxone + metronidazole, or carbapenem based on local resistance patterns.",
                        "fluid": "Fluid resuscitation: 30 mL/kg of crystalloids within 3 hours for sepsis-induced hypoperfusion. Reassess frequently and avoid fluid overload.",
                        "news2": "NEWS2 scoring: Temperature, pulse, respiratory rate, oxygen saturation, systolic BP, consciousness, oxygen therapy. Score ≥7 indicates high risk.",
                        "sirs": "SIRS criteria: 1) Temperature >38°C or <36°C, 2) Heart rate >90 bpm, 3) Respiratory rate >20/min, 4) WBC >12,000 or <4,000 cells/mm³."
                    }
                    
                    response = "Based on clinical guidelines:\n\n"
                    for key, value in fallback_responses.items():
                        if key.lower() in question.lower():
                            response += value
                            break
                    else:
                        response += "I recommend consulting current sepsis guidelines. Key points: Early recognition, prompt antibiotics, adequate fluid resuscitation, and source control."
                    
                    st.session_state.chat.append({"role": "assistant", "content": response})
                
                st.rerun()

if __name__ == "__main__":
    main()