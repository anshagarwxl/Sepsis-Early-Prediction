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
