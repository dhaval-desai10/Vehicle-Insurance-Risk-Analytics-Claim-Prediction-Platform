"""
PAGE 4 — Customer Advisor Dashboard
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_car_data, load_motor_data

st.set_page_config(page_title="Customer Advisor · InsureAI", page_icon="🤝", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0f1117;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1d2e,#12151f);border-right:1px solid #2d3147;}
h1{background:linear-gradient(135deg,#10b981,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800!important;}
h2,h3{color:#e2e8f0!important;font-weight:700!important;}
[data-testid="metric-container"]{background:#1e2130;border:1px solid #2d3147;border-radius:12px;padding:14px 18px!important;}
[data-testid="stMetricValue"]{color:#f9fafb!important;font-weight:800;font-size:26px!important;}
[data-testid="stMetricLabel"]{color:#9ca3af!important;}
#MainMenu,footer{visibility:hidden;}
</style>""", unsafe_allow_html=True)

st.title("🤝 Customer Advisor Dashboard")
st.caption("Customer risk profiling, premium benchmarking, and personalised recommendations")

with st.spinner("Loading data…"):
    car = load_car_data()
    motor = load_motor_data(50_000)

tab = st.tabs(["🔍 Customer Lookup", "📊 Segment Analysis", "💡 Premium Benchmarks"])

# ── CUSTOMER LOOKUP ───────────────────────────────────────────────────────────
with tab[0]:
    st.markdown("### 🧑 Build Customer Risk Profile")
    st.caption("Fill in the customer details to compute a Risk Score and premium advice")

    c1, c2, c3 = st.columns(3)
    with c1:
        age_grp    = st.selectbox("Age Group", ["16-25","26-39","40-64","65+"], index=1)
        gender     = st.selectbox("Gender", ["male","female"])
        education  = st.selectbox("Education", ["none","high school","university"])
    with c2:
        income     = st.selectbox("Income Level", ["poverty","working class","middle class","upper class"], index=2)
        vehicle_type = st.selectbox("Vehicle Type", ["sedan","sports car","SUV","truck","bus"], index=0)
        vehicle_yr = st.selectbox("Vehicle Year", ["before 2015","after 2015"])
    with c3:
        past_acc   = st.slider("Past Accidents", 0, 10, 0)
        speeding   = st.slider("Speeding Violations", 0, 15, 0)
        duis       = st.slider("DUIs", 0, 5, 0)

    if st.button("🚀 Calculate Risk Score & Premium Advice", type="primary"):
        # Rule-based risk score
        score = 50  # base
        age_map = {"16-25": +20, "26-39": -5, "40-64": -10, "65+": +5}
        score += age_map.get(age_grp, 0)
        score += past_acc * 8
        score += speeding * 4
        score += duis * 12
        if income == "poverty":    score += 10
        if income == "upper class": score -= 8
        if vehicle_type == "sports car": score += 15
        if vehicle_yr == "before 2015":  score += 5
        if education == "university":    score -= 6
        score = int(np.clip(score, 0, 100))

        risk_label = "🟢 Low Risk" if score < 35 else \
                     "🟡 Medium Risk" if score < 60 else \
                     "🔴 High Risk" if score < 80 else "🔴🔴 Very High Risk"
        risk_color = "#10b981" if score < 35 else "#f59e0b" if score < 60 else "#ef4444"

        base_premium = 3000
        multiplier = 1 + (score / 100) * 1.5
        rec_premium = int(base_premium * multiplier)

        # Display Result Card
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{risk_color}22,{risk_color}08);
                    border:2px solid {risk_color};border-radius:16px;padding:28px;margin:16px 0;">
            <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;">
                <div style="text-align:center;">
                    <div style="font-size:64px;font-weight:900;color:{risk_color};">{score}</div>
                    <div style="color:#9ca3af;font-size:14px;font-weight:600;">RISK SCORE</div>
                </div>
                <div style="flex:1;">
                    <div style="font-size:22px;font-weight:800;color:#f1f5f9;margin-bottom:8px;">
                        {risk_label}
                    </div>
                    <div style="color:#9ca3af;font-size:14px;line-height:1.6;">
                        Based on age group, driving history, vehicle type, income, and education level.
                    </div>
                    <div style="margin-top:12px;font-size:18px;font-weight:700;color:{risk_color};">
                        💰 Recommended Premium: <span style="font-size:24px;">{rec_premium:,}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Risk Score", "font": {"color": "#e2e8f0", "size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9ca3af"},
                "bar":  {"color": risk_color},
                "steps": [
                    {"range": [0,  35], "color": "rgba(16,185,129,0.2)"},
                    {"range": [35, 60], "color": "rgba(245,158,11,0.2)"},
                    {"range": [60, 80], "color": "rgba(239,68,68,0.2)"},
                    {"range": [80,100], "color": "rgba(127,29,29,0.2)"},
                ],
                "threshold": {"line": {"color": "white", "width": 4}, "value": score}
            },
            number={"font": {"color": "#f9fafb", "size": 48}}
        ))
        fig_gauge.update_layout(paper_bgcolor="#1e2130", font_color="#e2e8f0",
                                height=280, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Recommendations
        st.markdown("##### 💡 Advisor Recommendations")
        if past_acc > 3:
            st.warning("🚨 Customer has high accident history — consider defensive driving program discount.")
        if duis > 0:
            st.error(f"⛔ {duis} DUI(s) recorded — mandatory surcharge applies. Advise SR-22 filing.")
        if speeding > 5:
            st.warning(f"⚠️ {speeding} speeding violations — premium surcharge of {speeding*2}% recommended.")
        if score < 35:
            st.success("✅ Excellent profile! Offer loyalty discount and multi-policy bundle.")
        if vehicle_type == "sports car":
            st.info("🏎️ Sports car category — add collision endorsement recommendation.")

# ── SEGMENT ANALYSIS ──────────────────────────────────────────────────────────
with tab[1]:
    st.markdown("#### 📊 Customer Segments by Demographics")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Income vs Claim Rate")
        inc = car.groupby("INCOME")["OUTCOME"].mean().reset_index()
        inc["Claim Rate %"] = inc["OUTCOME"] * 100
        colors_map = {"poverty":"#ef4444","working class":"#f59e0b",
                      "middle class":"#10b981","upper class":"#6366f1"}
        fig = px.bar(inc, x="INCOME", y="Claim Rate %", color="INCOME",
                     color_discrete_map=colors_map, template="plotly_dark",
                     text=inc["Claim Rate %"].map("{:.1f}%".format))
        fig.update_traces(textposition="outside")
        fig.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                          height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("##### Credit Score vs Outcome")
        cs = car.groupby("OUTCOME")["CREDIT_SCORE"].agg(["mean","std"]).reset_index()
        fig2 = px.bar(cs, x="OUTCOME", y="mean", error_y="std",
                      color="OUTCOME",
                      color_discrete_map={0:"#10b981",1:"#ef4444"},
                      labels={"OUTCOME":"Claim","mean":"Avg Credit Score"},
                      template="plotly_dark",
                      text=cs["mean"].map("{:.3f}".format))
        fig2.update_traces(textposition="outside")
        fig2.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### 🗺️ Claim Rate by Age × Income (Bubble Chart)")
    seg = car.groupby(["AGE","INCOME"]).agg(
        Claim_Rate=("OUTCOME","mean"),
        Count=("OUTCOME","count")
    ).reset_index()
    seg["Claim Rate %"] = seg["Claim_Rate"] * 100
    fig3 = px.scatter(seg, x="AGE", y="INCOME", size="Count",
                      color="Claim Rate %", color_continuous_scale="RdYlGn_r",
                      size_max=60, template="plotly_dark",
                      hover_data={"Count": True, "Claim Rate %": ":.1f"})
    fig3.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=350)
    st.plotly_chart(fig3, use_container_width=True)

# ── PREMIUM BENCHMARKS ────────────────────────────────────────────────────────
with tab[2]:
    st.markdown("#### 💰 Premium Benchmarks by Vehicle Type")
    pm = motor.groupby("TYPE_VEHICLE")["PREMIUM"].agg(["mean","median","min","max","count"]).reset_index()
    pm.columns = ["Vehicle Type","Avg Premium","Median Premium","Min Premium","Max Premium","Policies"]
    pm = pm.sort_values("Avg Premium", ascending=False).head(15)

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name="Avg Premium",    x=pm["Vehicle Type"], y=pm["Avg Premium"],    marker_color="#6366f1"))
    fig4.add_trace(go.Bar(name="Median Premium", x=pm["Vehicle Type"], y=pm["Median Premium"], marker_color="#8b5cf6"))
    fig4.update_layout(barmode="group", template="plotly_dark",
                       paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                       height=380, xaxis_tickangle=-30,
                       legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig4, use_container_width=True)

    st.dataframe(pm.style.format({
        "Avg Premium": "{:,.0f}", "Median Premium": "{:,.0f}",
        "Min Premium": "{:,.0f}", "Max Premium": "{:,.0f}",
        "Policies": "{:,.0f}"
    }), use_container_width=True, hide_index=True)
