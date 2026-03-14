"""
PAGE 2 — Insurance Manager Dashboard
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_motor_data

st.set_page_config(page_title="Insurance Manager · InsureAI", page_icon="👔", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0f1117;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1d2e,#12151f);border-right:1px solid #2d3147;}
h1{background:linear-gradient(135deg,#8b5cf6,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800!important;}
h2,h3{color:#e2e8f0!important;font-weight:700!important;}
[data-testid="metric-container"]{background:#1e2130;border:1px solid #2d3147;border-radius:12px;padding:14px 18px!important;}
[data-testid="stMetricValue"]{color:#f9fafb!important;font-weight:800;font-size:26px!important;}
[data-testid="stMetricLabel"]{color:#9ca3af!important;}
#MainMenu,footer{visibility:hidden;}
</style>""", unsafe_allow_html=True)

st.title("👔 Insurance Manager Dashboard")
st.caption("Executive overview — portfolio performance, financial KPIs, and claim trends")

with st.spinner("Loading data…"):
    df = load_motor_data(100_000)

# ── Sidebar Filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Filters")
    veh_types = ["All"] + sorted(df["TYPE_VEHICLE"].dropna().unique().tolist())
    sel_veh = st.selectbox("Vehicle Type", veh_types)
    usages = ["All"] + sorted(df["USAGE"].dropna().unique().tolist()) if "USAGE" in df.columns else ["All"]
    sel_use = st.selectbox("Usage", usages)
    year_range = st.slider("Production Year", int(df["PROD_YEAR"].min()), int(df["PROD_YEAR"].max()),
                           (2000, 2015)) if "PROD_YEAR" in df.columns else (2000, 2015)

    filt = df.copy()
    if sel_veh != "All":
        filt = filt[filt["TYPE_VEHICLE"] == sel_veh]
    if sel_use != "All" and "USAGE" in filt.columns:
        filt = filt[filt["USAGE"] == sel_use]
    if "PROD_YEAR" in filt.columns:
        filt = filt[filt["PROD_YEAR"].between(*year_range)]

total_premium  = filt["PREMIUM"].sum()
total_claims   = filt["CLAIM_PAID"].sum()
n_policies     = len(filt)
claim_rate     = filt["CLAIM_FLAG"].mean() * 100
loss_ratio     = (total_claims / total_premium * 100) if total_premium > 0 else 0
avg_premium    = filt["PREMIUM"].mean()

# ── KPI Row ───────────────────────────────────────────────────────────────────
st.markdown("### 📈 Key Performance Indicators")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("📋 Total Policies",  f"{n_policies:,}")
k2.metric("💰 Gross Premium",   f"{total_premium:,.0f}")
k3.metric("💸 Total Claims",    f"{total_claims:,.0f}")
k4.metric("⚠️ Claim Rate",      f"{claim_rate:.1f}%",  delta=f"{'↑ High' if claim_rate>30 else '✓ OK'}")
k5.metric("📉 Loss Ratio",      f"{loss_ratio:.1f}%",  delta=f"{'⚠ Over 100%' if loss_ratio>100 else '✓ Profitable'}")
k6.metric("💵 Avg Premium",     f"{avg_premium:,.0f}")

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
row1a, row1b = st.columns(2)

with row1a:
    st.markdown("#### 📊 Premium vs Claims by Vehicle Type (Top 10)")
    grp = filt.groupby("TYPE_VEHICLE").agg(
        Total_Premium=("PREMIUM", "sum"),
        Total_Claims=("CLAIM_PAID", "sum"),
        Count=("PREMIUM", "count")
    ).reset_index().nlargest(10, "Total_Premium")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Premium", x=grp["TYPE_VEHICLE"],
                         y=grp["Total_Premium"], marker_color="#6366f1"))
    fig.add_trace(go.Bar(name="Claims",  x=grp["TYPE_VEHICLE"],
                         y=grp["Total_Claims"],  marker_color="#ef4444"))
    fig.update_layout(barmode="group", template="plotly_dark",
                      paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                      height=340, legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

with row1b:
    st.markdown("#### 🥧 Premium Distribution by Vehicle Usage")
    if "USAGE" in filt.columns:
        use_grp = filt.groupby("USAGE")["PREMIUM"].sum().reset_index()
        use_grp.columns = ["Usage", "Total_Premium"]
        fig2 = px.pie(use_grp, names="Usage", values="Total_Premium",
                      color_discrete_sequence=px.colors.qualitative.Bold,
                      template="plotly_dark", hole=0.45)
        fig2.update_layout(paper_bgcolor="#1e2130", height=340)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Usage column not available")

row2a, row2b = st.columns(2)

with row2a:
    st.markdown("#### 📅 Policies Written by Year")
    if "PROD_YEAR" in filt.columns:
        yr = filt.groupby("PROD_YEAR").size().reset_index(name="Policies")
        yr = yr[(yr["PROD_YEAR"] >= 1990) & (yr["PROD_YEAR"] <= 2020)]
        fig3 = px.area(yr, x="PROD_YEAR", y="Policies",
                       color_discrete_sequence=["#8b5cf6"],
                       template="plotly_dark")
        fig3.update_traces(line=dict(width=2.5), fillcolor="rgba(139,92,246,0.25)")
        fig3.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=300,
                           xaxis_title="Production Year", yaxis_title="Number of Policies")
        st.plotly_chart(fig3, use_container_width=True)

with row2b:
    st.markdown("#### 📈 Loss Ratio by Vehicle Type")
    lr = filt.groupby("TYPE_VEHICLE").agg(
        Premium=("PREMIUM", "sum"),
        Claims=("CLAIM_PAID", "sum")
    ).reset_index()
    lr["Loss_Ratio"] = (lr["Claims"] / lr["Premium"].replace(0, np.nan) * 100).fillna(0).clip(0, 300)
    lr = lr.sort_values("Loss_Ratio", ascending=False).head(12)
    fig4 = px.bar(lr, x="TYPE_VEHICLE", y="Loss_Ratio",
                  color="Loss_Ratio", color_continuous_scale="RdYlGn_r",
                  template="plotly_dark",
                  text=lr["Loss_Ratio"].map("{:.0f}%".format))
    fig4.add_hline(y=100, line_dash="dash", line_color="#fbbf24",
                   annotation_text="100% Break-even")
    fig4.update_traces(textposition="outside")
    fig4.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=300)
    st.plotly_chart(fig4, use_container_width=True)

# ── Top Vehicles Table ────────────────────────────────────────────────────────
st.markdown("#### 🏆 Top 15 Vehicle Types by Premium Collected")
top = filt.groupby("TYPE_VEHICLE").agg(
    Policies=("PREMIUM", "count"),
    Total_Premium=("PREMIUM", "sum"),
    Total_Claims=("CLAIM_PAID", "sum"),
    Claim_Rate=("CLAIM_FLAG", "mean"),
    Avg_Premium=("PREMIUM", "mean")
).reset_index().nlargest(15, "Total_Premium")
top["Loss_Ratio (%)"] = (top["Total_Claims"] / top["Total_Premium"].replace(0,np.nan) * 100).fillna(0).round(1)
top["Claim_Rate"] = (top["Claim_Rate"] * 100).round(1).astype(str) + "%"
top["Total_Premium"] = top["Total_Premium"].map("{:,.0f}".format)
top["Total_Claims"]  = top["Total_Claims"].map("{:,.0f}".format)
top["Avg_Premium"]   = top["Avg_Premium"].map("{:,.0f}".format)
st.dataframe(top.rename(columns={
    "TYPE_VEHICLE": "Vehicle Type",
    "Total_Premium": "Total Premium",
    "Total_Claims": "Total Claims",
    "Claim_Rate": "Claim Rate",
    "Avg_Premium": "Avg Premium"
}), use_container_width=True, hide_index=True)
