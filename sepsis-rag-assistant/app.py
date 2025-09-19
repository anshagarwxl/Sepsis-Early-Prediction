
# app.py
# Sepsis Risk Assistant — Dark Glassmorphism PRO
# --------------------------------------------------------------
# One-file Streamlit app with a premium dark-glass UI, animated hero,
# KPI tiles, radial gauges, crisp charts, ML risk (optional model),
# RAG assistant (optional rag_system.py), upload, duplicates, and export.
#
# Run:  streamlit run app.py
# Reqs: streamlit, pandas, numpy, matplotlib, (optional) joblib, python-dotenv

from __future__ import annotations

import os
import json
import math
import numbers
import datetime as dt
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---- Optional libs (safe fallbacks) ----
try:
    import joblib
except Exception:
    joblib = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------- App Meta --------------------
st.set_page_config(
    page_title="Sepsis Risk Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths for optional ML pipeline
MODEL_PATH = Path("models/knn_sepsis_model.joblib")
META_PATH  = Path("models/meta.json")

# -------------------- Global CSS / Theme --------------------
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
}

.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 18px 16px 18px;
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

h1, h2, h3 { color: var(--ink); }
small, .muted { color: var(--muted); }

.stButton>button {
  border-radius: 12px;
  border: 1px solid var(--border);
  background: linear-gradient(135deg, var(--blue), var(--violet));
  color: #06131f;
  font-weight: 900;
}

a, a:visited { color: var(--blue) !important; text-decoration: none; }
</style>
"""
st.markdown(GLASS_CSS, unsafe_allow_html=True)

# -------------------- Animated Hero --------------------
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
        <div style="font-weight:900; letter-spacing:.4px; font-size:26px;">Sepsis Risk Assistant</div>
        <div class="muted" style="margin-top:2px;">Dark glassmorphism • Clinical scores + ML • RAG • Upload • Duplicates • Exports</div>
      </div>
      <div style="margin-left:auto;">
        <span class="pill ml">Live Session</span>
      </div>
    </div>
  </div>
</div>
"""
st.markdown(HERO, unsafe_allow_html=True)
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# -------------------- Session State --------------------
SCHEMA = [
    "Timestamp","Temperature","HeartRate","RespiratoryRate",
    "SystolicBP","DiastolicBP","SpO2","WBC","GCS",
    "Consciousness","Notes",
    "NEWS2","NEWS2_Risk","qSOFA","qSOFA_Risk","SIRS","SIRS_Risk",
    "ML_Prediction","ML_Prob","ML_Risk"
]
if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame(columns=SCHEMA)

# -------------------- ML Loader --------------------
@st.cache_resource(show_spinner=False)
def load_ml():
    if joblib is None or not MODEL_PATH.exists() or not META_PATH.exists():
        return None, {}
    try:
        model = joblib.load(MODEL_PATH)
        meta  = json.loads(META_PATH.read_text())
        return model, meta
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load ML model: {e}")
        return None, {}

MODEL, META = load_ml()

def feature_order(meta: Dict[str, Any]) -> List[str]:
    for k in ("feature_columns","features","cols"):
        if isinstance(meta.get(k), (list,tuple)) and meta[k]:
            return list(meta[k])
    # sensible default order
    return ["Temperature","HeartRate","RespiratoryRate","SystolicBP","DiastolicBP","SpO2","WBC","GCS","Age"]

# -------------------- Clinical Scores --------------------
def news2(temp: float, hr: float, rr: float, sbp: float, spo2: float, cons: str) -> Tuple[int,str]:
    s = 0
    # RR
    if rr <= 8: s += 3
    elif 9 <= rr <= 11: s += 1
    elif 12 <= rr <= 20: s += 0
    elif 21 <= rr <= 24: s += 2
    else: s += 3
    # SpO2 (scale 1)
    if spo2 <= 91: s += 3
    elif 92 <= spo2 <= 93: s += 2
    elif 94 <= spo2 <= 95: s += 1
    # Temp
    if temp <= 35.0: s += 3
    elif 35.1 <= temp <= 36.0: s += 1
    elif 36.1 <= temp <= 38.0: s += 0
    elif 38.1 <= temp <= 39.0: s += 1
    else: s += 2
    # SBP
    if sbp <= 90: s += 3
    elif 91 <= sbp <= 100: s += 2
    elif 101 <= sbp <= 110: s += 1
    # HR
    if hr <= 40 or hr >= 131: s += 3
    elif 111 <= hr <= 130: s += 2
    elif 91 <= hr <= 110: s += 1
    # Consciousness
    if cons.lower() != "alert": s += 3

    # Risk buckets + red flag
    if s >= 7:
        r = "High"
    elif s >= 5:
        r = "Medium"
    else:
        red = any([
            rr <= 8 or rr >= 25,
            spo2 <= 91,
            temp <= 35.0 or temp >= 39.1,
            sbp <= 90,
            hr <= 40 or hr >= 131,
            cons.lower() != "alert",
        ])
        r = "Medium" if red else "Low"
    return s, r

def qsofa(rr: float, sbp: float, gcs: float|None, cons: str) -> Tuple[int,str]:
    s = 0
    if rr >= 22: s += 1
    if sbp <= 100: s += 1
    altered = (gcs is not None and gcs < 15) or (cons.lower() != "alert")
    if altered: s += 1
    r = "High" if s >= 2 else ("Medium" if s == 1 else "Low")
    return s, r

def sirs(temp: float, hr: float, rr: float, wbc: float|None) -> Tuple[int,str]:
    c = 0
    if temp > 38.0 or temp < 36.0: c += 1
    if hr > 90: c += 1
    if rr > 20: c += 1
    if wbc is not None and (wbc > 12 or wbc < 4): c += 1
    r = "High" if c >= 3 else ("Medium" if c == 2 else "Low")
    return c, r

def compute_all(v: Dict[str, Any]) -> Dict[str, Any]:
    temp = float(v["Temperature"])
    hr   = float(v["HeartRate"])
    rr   = float(v["RespiratoryRate"])
    sbp  = float(v["SystolicBP"])
    spo2 = float(v["SpO2"])
    wbc  = v.get("WBC", None)
    wbc  = float(wbc) if wbc not in (None, "", np.nan) else None
    gcs  = v.get("GCS", None)
    gcs  = float(gcs) if gcs not in (None, "", np.nan) else None
    cons = str(v.get("Consciousness", "Alert"))

    n2, n2r  = news2(temp, hr, rr, sbp, spo2, cons)
    qf, qfr  = qsofa(rr, sbp, gcs, cons)
    si, sir  = sirs(temp, hr, rr, wbc)

    ml_pred, ml_prob, ml_risk = None, None, None
    if MODEL is not None:
        try:
            feats = feature_order(META)
            row = {c: v.get(c, None) for c in feats}
            X   = pd.DataFrame([row], columns=feats)
            y   = MODEL.predict(X)[0]
            ml_pred = int(y) if isinstance(y, numbers.Number) else y
            if hasattr(MODEL, "predict_proba"):
                proba = MODEL.predict_proba(X)
                ml_prob = float(proba[0][1]) if proba.shape[1] == 2 else float(np.max(proba[0]))
            ml_risk = "High" if (ml_pred == 1) else "Low"
        except Exception as e:
            st.warning(f"ML inference failed: {e}")

    return dict(
        NEWS2=n2, NEWS2_Risk=n2r,
        qSOFA=qf, qSOFA_Risk=qfr,
        SIRS=si, SIRS_Risk=sir,
        ML_Prediction=ml_pred, ML_Prob=ml_prob, ML_Risk=ml_risk
    )

# -------------------- Reusable UI bits --------------------
def pill(text: str) -> str:
    key = text.lower()
    cls = "low" if key=="low" else "med" if key=="medium" else "high" if key=="high" else "ml"
    return f'<span class="pill {cls}">{text}</span>'

def kpi_tile(title: str, value: str, sub: str=""):
    st.markdown(
        f'<div class="tile"><div class="kpi">{value}</div>'
        f'<div class="kpi-sub">{title}</div>'
        f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True
    )

def gauge_svg(label: str, value: float, max_val: float, color: str, size: int=170):
    pct = max(0.0, min(value/max_val, 1.0))
    angle = pct * 270.0
    r = 64
    cx = size//2
    cy = size//2 + 12
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

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("### Controls")
    st.markdown(
        "<div class='soft'>Use the tabs below to navigate. Add vitals first, then explore analytics and ML/RAG.</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.caption("Model status:")
    if MODEL is None:
        st.error("No ML model loaded")
    else:
        st.success("ML model loaded")
    st.caption("RAG status:")
    try:
        import rag_system  # noqa
        st.success("RAG available")
    except Exception:
        st.warning("RAG not available")

# -------------------- Main Tabs --------------------
tab_overview, tab_patient, tab_analytics, tab_duplicates, tab_assistant, tab_upload = st.tabs(
    ["🏠 Overview", "🩺 Patient", "📈 Analytics", "🧬 Duplicates", "💬 Assistant", "📂 Upload"]
)

# === OVERVIEW ===
with tab_overview:
    st.markdown("#### At-a-glance")

    df = st.session_state["data"]
    total = len(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_tile("Total Records", f"{total}")
    with col2:
        if total:
            high_news = (df["NEWS2_Risk"] == "High").sum()
            kpi_tile("High NEWS2 %", f"{(100*high_news/total):.1f}%")
        else:
            kpi_tile("High NEWS2 %", "0.0%")
    with col3:
        if total and "ML_Risk" in df.columns and df["ML_Risk"].notna().any():
            high_ml = (df["ML_Risk"] == "High").sum()
            kpi_tile("ML High-Risk %", f"{(100*high_ml/total):.1f}%")
        else:
            kpi_tile("ML High-Risk %", "—", "Load a model to enable.")
    with col4:
        if total:
            core = ["Temperature","HeartRate","RespiratoryRate","SystolicBP","SpO2","WBC","GCS","Consciousness"]
            completeness = np.mean([df[c].notna().mean() if c in df else 0 for c in core]) * 100
            kpi_tile("Data Completeness", f"{completeness:.0f}%")
        else:
            kpi_tile("Data Completeness", "0%")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Latest snapshot card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Latest Snapshot**")
    if total == 0:
        st.info("No vitals yet. Go to **Patient** tab to add a record.")
    else:
        latest = df.iloc[-1]
        cA, cB, cC, cD = st.columns(4)
        with cA:
            st.markdown(
                gauge_svg("NEWS2", float(latest["NEWS2"]), 20, "#ff6b6b" if latest["NEWS2_Risk"]=="High" else "#ffcc66" if latest["NEWS2_Risk"]=="Medium" else "#00d1a7"),
                unsafe_allow_html=True
            )
        with cB:
            st.markdown(
                gauge_svg("qSOFA", float(latest["qSOFA"]), 3, "#ff6b6b" if latest["qSOFA_Risk"]=="High" else "#ffcc66" if latest["qSOFA_Risk"]=="Medium" else "#00d1a7"),
                unsafe_allow_html=True
            )
        with cC:
            st.markdown(
                gauge_svg("SIRS", float(latest["SIRS"]), 4, "#ff6b6b" if latest["SIRS_Risk"]=="High" else "#ffcc66" if latest["SIRS_Risk"]=="Medium" else "#00d1a7"),
                unsafe_allow_html=True
            )
        with cD:
            if pd.notna(latest["ML_Risk"]):
                col = "#ff6b6b" if latest["ML_Risk"]=="High" else "#7fd3ff"
                score = 100*float(latest["ML_Prob"]) if pd.notna(latest["ML_Prob"]) else 50
                st.markdown(
                    gauge_svg("ML Risk%", score, 100, col),
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<div class="soft">Load an ML model to view ML gauge.</div>', unsafe_allow_html=True)

        st.markdown(
            f"- NEWS2: **{int(latest['NEWS2'])}** {pill(latest['NEWS2_Risk'])} &nbsp; "
            f"qSOFA: **{int(latest['qSOFA'])}** {pill(latest['qSOFA_Risk'])} &nbsp; "
            f"SIRS: **{int(latest['SIRS'])}** {pill(latest['SIRS_Risk'])} &nbsp; "
            + (f"ML: {pill(latest['ML_Risk'])} " + (f"(p={latest['ML_Prob']:.2f})" if pd.notna(latest['ML_Prob']) else "") if pd.notna(latest['ML_Risk']) else "ML: —"),
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# === PATIENT ENTRY ===
with tab_patient:
    st.markdown("#### Add / Score Vitals")
    with st.form("vitals_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            temperature = st.number_input("Temperature (°C)", 30.0, 45.0, 37.0, step=0.1)
            heart_rate  = st.number_input("Heart Rate (bpm)", 0, 250, 92, step=1)
            respiratory = st.number_input("Respiratory Rate (breaths/min)", 0, 60, 22, step=1)
            systolic    = st.number_input("Systolic BP (mmHg)", 0, 300, 98, step=1)
        with c2:
            diastolic   = st.number_input("Diastolic BP (mmHg)", 0, 200, 64, step=1)
            spo2        = st.number_input("SpO₂ (%)", 0, 100, 93, step=1)
            wbc         = st.number_input("WBC (×10³/µL)", 0.0, 60.0, 12.1, step=0.1)
            gcs         = st.number_input("GCS (3–15)", 3, 15, 14, step=1)
        with c3:
            consciousness = st.selectbox("Consciousness (AVPU)", ["Alert","Verbal","Pain","Unresponsive"], index=1)
            notes         = st.text_area("Notes (optional)", "", height=90)
            submitted     = st.form_submit_button("Save & Score")

    if submitted:
        errs=[]
        if heart_rate <= 0: errs.append("Heart rate invalid.")
        if respiratory <= 0: errs.append("Respiratory rate invalid.")
        if systolic <= 0: errs.append("Systolic BP invalid.")
        if spo2 < 0 or spo2 > 100: errs.append("SpO₂ must be 0–100.")
        if errs:
            for e in errs: st.error(e)
        else:
            rec = dict(
                Timestamp=dt.datetime.now(),
                Temperature=float(temperature), HeartRate=float(heart_rate),
                RespiratoryRate=float(respiratory), SystolicBP=float(systolic),
                DiastolicBP=float(diastolic), SpO2=float(spo2),
                WBC=float(wbc), GCS=float(gcs),
                Consciousness=str(consciousness), Notes=str(notes).strip()
            )
            rec.update(compute_all(rec))
            st.session_state["data"] = pd.concat(
                [st.session_state["data"], pd.DataFrame([rec], columns=SCHEMA)],
                ignore_index=True
            )
            st.success("Vitals saved. Scores computed.")
            with st.expander("Show latest record"):
                st.dataframe(st.session_state["data"].tail(1), use_container_width=True)

# === ANALYTICS ===
with tab_analytics:
    st.markdown("#### Risk & Quality Analytics")
    df = st.session_state["data"]

    if df.empty:
        st.info("Add some vitals in the Patient tab.")
    else:
        # Risk distribution (NEWS2)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**NEWS2 Risk Distribution**")
        counts = df["NEWS2_Risk"].value_counts()
        for cat in ["Low","Medium","High"]:
            if cat not in counts: counts.loc[cat] = 0
        counts = counts[["Low","Medium","High"]]

        fig, ax = plt.subplots(figsize=(4.8,4.0))
        ax.pie(
            counts.values,
            labels=[f"{k} ({int(v)})" for k,v in counts.items()],
            startangle=90, counterclock=False, autopct="%1.1f%%",
            wedgeprops={'width':0.38, 'edgecolor':'#1a2030'},
            colors=["#00d1a7","#ffcc66","#ff6b6b"]
        )
        ax.axis('equal')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # Completeness
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Field Completeness (%)**")
        fields = ["Temperature","HeartRate","RespiratoryRate","SystolicBP","DiastolicBP","SpO2","WBC","GCS","Consciousness","Notes"]
        comp = []
        for c in fields:
            if c not in df:
                comp.append(0.0)
            else:
                if df[c].dtype == object:
                    filled = df[c].astype(str).str.strip().ne("").mean() * 100.0
                else:
                    filled = df[c].notna().mean() * 100.0
                comp.append(filled)
        st.bar_chart(pd.DataFrame({"Field":fields,"Filled %":comp}).set_index("Field"))
        st.markdown('</div>', unsafe_allow_html=True)

        # NEWS2 Trend
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**NEWS2 Trend**")
        tr = df[["Timestamp","NEWS2"]].copy()
        if not pd.api.types.is_datetime64_any_dtype(tr["Timestamp"]):
            tr["Timestamp"] = pd.to_datetime(tr["Timestamp"], errors="coerce").fillna(pd.Timestamp.now())
        tr = tr.sort_values("Timestamp").set_index("Timestamp")
        st.line_chart(tr["NEWS2"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Export
        st.markdown('<div class="soft">', unsafe_allow_html=True)
        st.markdown("**Export session data**")
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="sepsis_session.csv",
            mime="text/csv"
        )
        st.markdown('</div>', unsafe_allow_html=True)

# === DUPLICATES ===
with tab_duplicates:
    st.markdown("#### Duplicate / Similar Records")
    df = st.session_state["data"]
    if df.empty:
        st.info("No data to analyze yet.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Exact-match duplicates over key vitals.")
        subset = ["Temperature","HeartRate","RespiratoryRate","SystolicBP","DiastolicBP","SpO2"]
        mask = df.duplicated(subset=subset, keep=False)
        dups = df[mask].copy()
        if dups.empty:
            st.success("No exact duplicates detected.")
        else:
            st.warning(f"Potential duplicates: {len(dups)} record(s).")
            groups = dups.groupby(subset).apply(lambda x: list(x.index)).to_dict()
            for k, idxs in groups.items():
                if len(idxs) >= 2:
                    st.markdown(
                        f"- Records {idxs} share "
                        f"Temp={k[0]}, HR={k[1]}, RR={k[2]}, SBP={k[3]}, DBP={k[4]}, SpO₂={k[5]}"
                    )
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
            st.dataframe(dups.reset_index(drop=True), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# === ASSISTANT (RAG) ===
with tab_assistant:
    st.markdown("#### AI Assistant (RAG)")
    try:
        import rag_system  # your module exposing: get_answer / answer_query / ask / run
        rag_ok = True
    except Exception as e:
        rag_ok = False
        st.error(f"RAG not available: {e}")

    if rag_ok:
        if "chat" not in st.session_state:
            st.session_state["chat"] = []
        q = st.text_input("Ask about sepsis or guidelines", "Initial fluid bolus for suspected sepsis?")
        c1, c2 = st.columns([1,6])
        with c1:
            if st.button("Ask"):
                try:
                    if hasattr(rag_system, "get_answer"):
                        resp = rag_system.get_answer(q)
                    elif hasattr(rag_system, "answer_query"):
                        resp = rag_system.answer_query(q)
                    elif hasattr(rag_system, "ask"):
                        resp = rag_system.ask(q)
                    else:
                        resp = rag_system.run(q)
                    ans, src = "", []
                    if isinstance(resp, dict):
                        ans = resp.get("answer","")
                        src = resp.get("sources", [])
                    elif isinstance(resp, tuple):
                        ans = resp[0]; src = resp[1] if len(resp)>1 else []
                    else:
                        ans = str(resp)
                    st.session_state["chat"].append({"q":q, "a":ans, "s":src})
                except Exception as e:
                    st.error(f"RAG error: {e}")
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        for item in st.session_state["chat"]:
            st.markdown(f"**Q:** {item['q']}")
            st.markdown(f"**A:** {item['a']}")
            if item["s"]:
                st.markdown("**Sources**")
                for i, s in enumerate(item["s"], start=1):
                    with st.expander(f"Source {i}"):
                        if isinstance(s, dict):
                            st.markdown(f"**{s.get('title') or s.get('source') or 'Document'}**")
                            st.write(s.get("content","")[:800] or "(no preview)")
                        else:
                            st.write(str(s))
            st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# === UPLOAD ===
with tab_upload:
    st.markdown("#### Upload Vitals CSV")
    st.caption("Columns: Temperature, HeartRate, RespiratoryRate, SystolicBP, DiastolicBP, SpO2, WBC, GCS, Consciousness, Notes")
    up = st.file_uploader("Choose CSV", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            # normalize common aliases
            alias = {
                "temp":"Temperature","temperature":"Temperature",
                "hr":"HeartRate","heart rate":"HeartRate","heartrate":"HeartRate",
                "resp":"RespiratoryRate","respiratoryrate":"RespiratoryRate","resp rate":"RespiratoryRate",
                "sbp":"SystolicBP","systolicbp":"SystolicBP",
                "dbp":"DiastolicBP","diastolicbp":"DiastolicBP",
                "spo2":"SpO2","o2sat":"SpO2",
                "wbc":"WBC","gcs":"GCS",
                "consciousness level":"Consciousness"
            }
            df_in = df_in.rename(columns={c: alias.get(c.strip().lower(), c) for c in df_in.columns})
            req = ["Temperature","HeartRate","RespiratoryRate","SystolicBP","DiastolicBP","SpO2","WBC","GCS","Consciousness"]
            miss = [c for c in req if c not in df_in.columns]
            if miss:
                st.error(f"Missing required columns: {miss}")
            else:
                for c in ["Temperature","HeartRate","RespiratoryRate","SystolicBP","DiastolicBP","SpO2","WBC","GCS"]:
                    df_in[c] = pd.to_numeric(df_in[c], errors="coerce")
                df_in["Consciousness"] = df_in["Consciousness"].fillna("Alert").astype(str)
                if "Notes" not in df_in: df_in["Notes"] = ""
                df_in["Timestamp"] = dt.datetime.now()

                out = []
                for _, r in df_in.iterrows():
                    row = dict(
                        Timestamp=r["Timestamp"],
                        Temperature=float(r["Temperature"]) if not pd.isna(r["Temperature"]) else 0.0,
                        HeartRate=float(r["HeartRate"]) if not pd.isna(r["HeartRate"]) else 0.0,
                        RespiratoryRate=float(r["RespiratoryRate"]) if not pd.isna(r["RespiratoryRate"]) else 0.0,
                        SystolicBP=float(r["SystolicBP"]) if not pd.isna(r["SystolicBP"]) else 0.0,
                        DiastolicBP=float(r["DiastolicBP"]) if not pd.isna(r["DiastolicBP"]) else 0.0,
                        SpO2=float(r["SpO2"]) if not pd.isna(r["SpO2"]) else 0.0,
                        WBC=float(r["WBC"]) if not pd.isna(r["WBC"]) else None,
                        GCS=float(r["GCS"]) if not pd.isna(r["GCS"]) else None,
                        Consciousness=str(r["Consciousness"]) if pd.notna(r["Consciousness"]) else "Alert",
                        Notes=str(r["Notes"]) if pd.notna(r["Notes"]) else ""
                    )
                    row.update(compute_all(row))
                    out.append(row)

                if out:
                    df_out = pd.DataFrame(out, columns=SCHEMA)
                    st.session_state["data"] = pd.concat([st.session_state["data"], df_out], ignore_index=True)
                    st.success(f"Imported {len(df_out)} records from {up.name}.")
                    st.dataframe(df_out.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Upload failed: {e}")

# -------------------- Footer --------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; opacity:.8;">'
    'Sepsis Risk Assistant • Dark Glassmorphism PRO • '
    f'Python {os.sys.version.split()[0]}'
    '</div>',
    unsafe_allow_html=True
)