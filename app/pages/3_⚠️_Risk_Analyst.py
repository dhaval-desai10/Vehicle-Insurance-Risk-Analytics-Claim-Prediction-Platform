"""
PAGE 3 — Risk Analyst Dashboard
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_loader import load_motor_data, load_car_data

st.set_page_config(page_title="Risk Analyst · InsureAI", page_icon="⚠️", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0f1117;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1d2e,#12151f);border-right:1px solid #2d3147;}
h1{background:linear-gradient(135deg,#f59e0b,#ef4444);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800!important;}
h2,h3{color:#e2e8f0!important;font-weight:700!important;}
[data-testid="metric-container"]{background:#1e2130;border:1px solid #2d3147;border-radius:12px;padding:14px 18px!important;}
[data-testid="stMetricValue"]{color:#f9fafb!important;font-weight:800;font-size:26px!important;}
[data-testid="stMetricLabel"]{color:#9ca3af!important;}
#MainMenu,footer{visibility:hidden;}
</style>""", unsafe_allow_html=True)

st.title("⚠️ Risk Analyst Dashboard")
st.caption("Identify high-risk segments, claim probability patterns, and exposure analysis")

with st.spinner("Loading data…"):
    df  = load_motor_data(100_000)
    car = load_car_data()

tab = st.tabs(["🔴 Risk Heatmaps", "📊 Claim Analysis", "👤 Profile Risk (Car Data)"])

# ── RISK HEATMAPS ─────────────────────────────────────────────────────────────
with tab[0]:
    c1, c2, c3, c4 = st.columns(4)
    high_risk = df[df["CLAIM_FLAG"] == 1]
    c1.metric("High-Risk Policies", f"{len(high_risk):,}", f"{len(high_risk)/len(df)*100:.1f}% of portfolio")
    c2.metric("Highest Claim Type", df.groupby("TYPE_VEHICLE")["CLAIM_PAID"].mean().idxmax())
    c3.metric("Peak Risk Year", str(int(df.groupby("PROD_YEAR")["CLAIM_FLAG"].mean().idxmax())) if "PROD_YEAR" in df else "N/A")
    c4.metric("Avg Claim (Risk)", f"{high_risk['CLAIM_PAID'].mean():,.0f}")

    st.markdown("---")

    # Heatmap: Vehicle Type vs Vehicle Age vs Claim Rate
    st.markdown("#### 🔥 Claim Rate Heatmap — Vehicle Type × Vehicle Age Bucket")
    df["Age_Bucket"] = pd.cut(df["VEHICLE_AGE"], bins=[0,5,10,15,20,30,60],
                               labels=["0-5y","5-10y","10-15y","15-20y","20-30y","30y+"])
    heat = df.groupby(["TYPE_VEHICLE","Age_Bucket"])["CLAIM_FLAG"].mean().reset_index()
    heat.columns = ["Vehicle Type","Age Bucket","Claim Rate"]
    heat["Claim Rate %"] = (heat["Claim Rate"] * 100).round(1)

    top_types = df["TYPE_VEHICLE"].value_counts().head(10).index.tolist()
    heat = heat[heat["Vehicle Type"].isin(top_types)]
    piv = heat.pivot(index="Vehicle Type", columns="Age Bucket", values="Claim Rate %").fillna(0)

    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
        colorscale="RdYlGn_r", zmid=20,
        text=piv.round(1).values, texttemplate="%{text}%",
        showscale=True, colorbar=dict(title="Claim Rate %")
    ))
    fig.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                      font=dict(color="#e2e8f0"), height=420,
                      xaxis_title="Vehicle Age Group", yaxis_title="Vehicle Type")
    st.plotly_chart(fig, use_container_width=True)

    # Risk tier distribution
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🏷️ Risk Tier by Insured Value")
        df["Risk_Tier"] = pd.cut(df["INSURED_VALUE"],
                                  bins=[-1, 100000, 500000, 1000000, 5000000, float("inf")],
                                  labels=["Very Low", "Low", "Medium", "High", "Very High"])
        rt = df.groupby("Risk_Tier")["CLAIM_FLAG"].mean().reset_index()
        rt["Claim Rate %"] = rt["CLAIM_FLAG"] * 100
        colors = {"Very Low": "#10b981", "Low": "#84cc16",
                  "Medium": "#f59e0b", "High": "#ef4444", "Very High": "#7f1d1d"}
        fig2 = px.bar(rt, x="Risk_Tier", y="Claim Rate %",
                      color="Risk_Tier",
                      color_discrete_map=colors, template="plotly_dark",
                      text=rt["Claim Rate %"].map("{:.1f}%".format))
        fig2.update_traces(textposition="outside")
        fig2.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                           height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("#### 📈 Claim Probability by Vehicle Age")
        age_cr = df.groupby("VEHICLE_AGE")["CLAIM_FLAG"].mean().reset_index()
        age_cr = age_cr[age_cr["VEHICLE_AGE"] <= 40]
        age_cr.columns = ["Vehicle Age", "Claim Rate"]
        age_cr["Claim Rate %"] = age_cr["Claim Rate"] * 100
        fig3 = px.line(age_cr, x="Vehicle Age", y="Claim Rate %",
                       color_discrete_sequence=["#f59e0b"], template="plotly_dark")
        fig3.update_traces(line_width=3)
        fig3.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=320)
        st.plotly_chart(fig3, use_container_width=True)

# ── CLAIM ANALYSIS ────────────────────────────────────────────────────────────
with tab[1]:
    st.markdown("#### 💸 Top 10 Highest Average Claims by Vehicle Type")
    top_cl = df[df["CLAIM_PAID"] > 0].groupby("TYPE_VEHICLE").agg(
        Avg_Claim=("CLAIM_PAID", "mean"),
        Max_Claim=("CLAIM_PAID", "max"),
        Count=("CLAIM_PAID", "count")
    ).reset_index().nlargest(10, "Avg_Claim")

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name="Avg Claim", x=top_cl["TYPE_VEHICLE"],
                          y=top_cl["Avg_Claim"], marker_color="#ef4444"))
    fig4.add_trace(go.Scatter(name="Max Claim", x=top_cl["TYPE_VEHICLE"],
                              y=top_cl["Max_Claim"], mode="markers+lines",
                              marker=dict(color="#fbbf24", size=10)))
    fig4.update_layout(barmode="overlay", template="plotly_dark",
                       paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                       height=380, legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig4, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 📦 Claim Severity Distribution by Usage")
        if "USAGE" in df.columns:
            clm_u = df[df["CLAIM_PAID"] > 0].copy()
            p99 = clm_u["CLAIM_PAID"].quantile(0.97)
            clm_u = clm_u[clm_u["CLAIM_PAID"] < p99]
            top_uses = clm_u["USAGE"].value_counts().head(6).index
            clm_u = clm_u[clm_u["USAGE"].isin(top_uses)]
            fig5 = px.violin(clm_u, x="USAGE", y="CLAIM_PAID", color="USAGE",
                             template="plotly_dark",
                             color_discrete_sequence=px.colors.qualitative.Set2)
            fig5.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                               height=320, showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig5, use_container_width=True)

    with col_b:
        st.markdown("#### 🔵 Claim Count by Make (Top 10)")
        mk = df[df["CLAIM_PAID"] > 0].groupby("MAKE").size().reset_index(name="Claims")
        mk = mk.nlargest(10, "Claims")
        fig6 = px.funnel(mk, y="MAKE", x="Claims",
                         color_discrete_sequence=["#6366f1"],
                         template="plotly_dark")
        fig6.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=320)
        st.plotly_chart(fig6, use_container_width=True)

    # Treemap of exposure
    st.markdown("#### 🌳 Risk Exposure Treemap (Premium by Type×Make)")
    if "MAKE" in df.columns:
        tr = df.groupby(["TYPE_VEHICLE","MAKE"])["PREMIUM"].sum().reset_index()
        tr.columns = ["Vehicle Type","Make","Total Premium"]
        tr = tr.nlargest(60, "Total Premium")
        fig7 = px.treemap(tr, path=["Vehicle Type","Make"], values="Total Premium",
                          color="Total Premium", color_continuous_scale="Blues",
                          template="plotly_dark")
        fig7.update_layout(paper_bgcolor="#1e2130", height=420)
        st.plotly_chart(fig7, use_container_width=True)

# ── PROFILE RISK (CAR DATA) ───────────────────────────────────────────────────
with tab[2]:
    st.markdown("#### 🔬 Risk Factor Analysis — Car Insurance Dataset")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Speeding Violations vs Claim Rate")
        sv = car.groupby("SPEEDING_VIOLATIONS")["OUTCOME"].mean().reset_index()
        sv["Claim Rate %"] = sv["OUTCOME"] * 100
        sv = sv[sv["SPEEDING_VIOLATIONS"] <= 15]
        fig8 = px.bar(sv, x="SPEEDING_VIOLATIONS", y="Claim Rate %",
                      color="Claim Rate %", color_continuous_scale="RdYlGn_r",
                      template="plotly_dark")
        fig8.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=300)
        st.plotly_chart(fig8, use_container_width=True)

    with col_b:
        st.markdown("##### DUIs vs Claim Rate")
        du = car.groupby("DUIS")["OUTCOME"].mean().reset_index()
        du["Claim Rate %"] = du["OUTCOME"] * 100
        fig9 = px.bar(du, x="DUIS", y="Claim Rate %",
                      color="Claim Rate %", color_continuous_scale="Reds",
                      template="plotly_dark")
        fig9.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=300)
        st.plotly_chart(fig9, use_container_width=True)

    # Risk matrix
    st.markdown("##### 🎯 Risk Matrix: Past Accidents × Speeding Violations → Claim Rate")
    car["Acc_Bucket"] = car["PAST_ACCIDENTS"].clip(0, 8)
    car["Spd_Bucket"] = car["SPEEDING_VIOLATIONS"].clip(0, 10)
    rm = car.groupby(["Acc_Bucket","Spd_Bucket"])["OUTCOME"].mean().reset_index()
    rm.columns = ["Past Accidents","Speeding Violations","Claim Rate"]
    piv_rm = rm.pivot(index="Past Accidents", columns="Speeding Violations", values="Claim Rate").fillna(0)
    fig10 = go.Figure(go.Heatmap(
        z=(piv_rm.values*100).round(1), x=piv_rm.columns.tolist(), y=piv_rm.index.tolist(),
        colorscale="RdYlGn_r", text=(piv_rm.values*100).round(1),
        texttemplate="%{text}%", showscale=True,
        colorbar=dict(title="Claim Rate %")
    ))
    fig10.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                        font=dict(color="#e2e8f0"), height=400,
                        xaxis_title="Speeding Violations", yaxis_title="Past Accidents")
    st.plotly_chart(fig10, use_container_width=True)
