"""
PAGE 1 — EDA Dashboard
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_loader import load_car_data, load_motor_data

st.set_page_config(page_title="EDA Dashboard · InsureAI", page_icon="📊", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0f1117;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1d2e,#12151f);border-right:1px solid #2d3147;}
h1{background:linear-gradient(135deg,#6366f1,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800!important;}
h2,h3{color:#e2e8f0!important;font-weight:700!important;}
[data-testid="metric-container"]{background:#1e2130;border:1px solid #2d3147;border-radius:12px;padding:14px 18px!important;}
[data-testid="stMetricValue"]{color:#f9fafb!important;font-weight:800;font-size:26px!important;}
[data-testid="stMetricLabel"]{color:#9ca3af!important;}
#MainMenu,footer{visibility:hidden;}
</style>""", unsafe_allow_html=True)

st.title("📊 Exploratory Data Analysis Dashboard")
st.caption("Deep dive into vehicle insurance data quality, distributions, and patterns")

tab = st.tabs(["🚗 Motor Dataset", "👤 Car Insurance Dataset"])

# ── MOTOR DATA TAB ────────────────────────────────────────────────────────────
with tab[0]:
    with st.spinner("Loading motor data…"):
        df = load_motor_data(50_000)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Vehicle Types", f"{df['TYPE_VEHICLE'].nunique()}")
    c3.metric("Avg Premium", f"{df['PREMIUM'].mean():,.0f}")
    c4.metric("Claim Rate", f"{df['CLAIM_FLAG'].mean()*100:.1f}%")
    c5.metric("Avg Claim Paid", f"{df[df['CLAIM_PAID']>0]['CLAIM_PAID'].mean():,.0f}")

    st.markdown("---")

    # Missing values heatmap
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🔍 Missing Values by Column")
        miss = (df.isnull().sum() / len(df) * 100).reset_index()
        miss.columns = ["Column", "Missing %"]
        miss = miss[miss["Missing %"] > 0].sort_values("Missing %", ascending=True)
        if miss.empty:
            st.success("✅ No missing values detected!")
        else:
            fig = px.bar(miss, x="Missing %", y="Column", orientation="h",
                         color="Missing %", color_continuous_scale="Reds",
                         template="plotly_dark")
            fig.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                              height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### 🚗 Policies by Vehicle Type")
        vc = df["TYPE_VEHICLE"].value_counts().head(10).reset_index()
        vc.columns = ["Vehicle Type", "Count"]
        fig2 = px.bar(vc, x="Count", y="Vehicle Type", orientation="h",
                      color="Count", color_continuous_scale="Blues",
                      template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                           height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Premium distribution
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### 💰 Premium Distribution")
        p99 = df["PREMIUM"].quantile(0.99)
        fig3 = px.histogram(df[df["PREMIUM"] < p99], x="PREMIUM", nbins=60,
                            color_discrete_sequence=["#6366f1"], template="plotly_dark")
        fig3.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=280)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown("#### 📅 Claim Paid Distribution (Claims Only)")
        claims = df[df["CLAIM_PAID"] > 0]["CLAIM_PAID"]
        p99c = claims.quantile(0.99)
        fig4 = px.histogram(claims[claims < p99c], x="CLAIM_PAID", nbins=60,
                            color_discrete_sequence=["#ef4444"], template="plotly_dark")
        fig4.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=280)
        st.plotly_chart(fig4, use_container_width=True)

    # Claim rate by vehicle type
    st.markdown("#### ⚠️ Claim Rate by Vehicle Type")
    cr = df.groupby("TYPE_VEHICLE").agg(
        Claim_Rate=("CLAIM_FLAG", "mean"),
        Count=("CLAIM_FLAG", "count"),
        Avg_Premium=("PREMIUM", "mean")
    ).reset_index().sort_values("Claim_Rate", ascending=False).head(12)
    cr["Claim_Rate_pct"] = cr["Claim_Rate"] * 100

    fig5 = px.bar(cr, x="TYPE_VEHICLE", y="Claim_Rate_pct",
                  color="Claim_Rate_pct", color_continuous_scale="RdYlGn_r",
                  text=cr["Claim_Rate_pct"].map("{:.1f}%".format),
                  template="plotly_dark", hover_data=["Count", "Avg_Premium"])
    fig5.update_traces(textposition="outside")
    fig5.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                       height=360, xaxis_title="Vehicle Type", yaxis_title="Claim Rate (%)")
    st.plotly_chart(fig5, use_container_width=True)

    # Premium vs Claim scatter
    st.markdown("#### 🔴 Premium vs Claim Paid (sampled)")
    sample = df[df["CLAIM_PAID"] > 0].sample(min(2000, len(df[df["CLAIM_PAID"]>0])))
    p99p = sample["PREMIUM"].quantile(0.98)
    p99cl = sample["CLAIM_PAID"].quantile(0.98)
    sample_f = sample[(sample["PREMIUM"] < p99p) & (sample["CLAIM_PAID"] < p99cl)]
    fig6 = px.scatter(sample_f, x="PREMIUM", y="CLAIM_PAID",
                      color="TYPE_VEHICLE", size="INSURED_VALUE",
                      size_max=14, opacity=0.7,
                      template="plotly_dark",
                      color_discrete_sequence=px.colors.qualitative.Vivid)
    fig6.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=380)
    st.plotly_chart(fig6, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### 🔗 Correlation Matrix (Numeric Features)")
    num_cols = ["INSURED_VALUE", "PREMIUM", "CLAIM_PAID", "VEHICLE_AGE",
                "SEATS_NUM", "POLICY_MONTHS", "CLAIM_FLAG"]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr()
    fig7 = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale="RdBu", zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        showscale=True
    ))
    fig7.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                       font=dict(color="#e2e8f0"), height=420)
    st.plotly_chart(fig7, use_container_width=True)

    # Vehicle age distribution
    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("#### 🚘 Vehicle Age Distribution")
        fig8 = px.histogram(df, x="VEHICLE_AGE", nbins=30,
                            color_discrete_sequence=["#8b5cf6"], template="plotly_dark")
        fig8.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=280)
        st.plotly_chart(fig8, use_container_width=True)

    with col_f:
        st.markdown("#### 🏭 Top 10 Vehicle Makes")
        makes = df["MAKE"].value_counts().head(10).reset_index()
        makes.columns = ["Make", "Count"]
        fig9 = px.pie(makes, names="Make", values="Count",
                      color_discrete_sequence=px.colors.qualitative.Set3,
                      template="plotly_dark", hole=0.4)
        fig9.update_layout(paper_bgcolor="#1e2130", height=280)
        st.plotly_chart(fig9, use_container_width=True)

    # Raw data sample
    with st.expander("📋 View Raw Data Sample (500 rows)"):
        st.dataframe(df.head(500), use_container_width=True, height=320)

# ── CAR INSURANCE TAB ─────────────────────────────────────────────────────────
with tab[1]:
    with st.spinner("Loading car insurance data…"):
        car = load_car_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(car):,}")
    c2.metric("Claim Rate", f"{car['OUTCOME'].mean()*100:.1f}%")
    c3.metric("Vehicle Types", f"{car['VEHICLE_TYPE'].nunique()}")
    c4.metric("Avg Annual Mileage", f"{car['ANNUAL_MILEAGE'].mean():,.0f}")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 👤 Outcome by Age Group")
        ag = car.groupby("AGE")["OUTCOME"].mean().reset_index()
        ag["Claim Rate %"] = ag["OUTCOME"] * 100
        fig = px.bar(ag, x="AGE", y="Claim Rate %", color="Claim Rate %",
                     color_continuous_scale="OrRd", template="plotly_dark",
                     text=ag["Claim Rate %"].map("{:.1f}%".format))
        fig.update_traces(textposition="outside")
        fig.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### 🚗 Claim Rate by Vehicle Type")
        vt = car.groupby("VEHICLE_TYPE")["OUTCOME"].mean().reset_index()
        vt["Claim Rate %"] = vt["OUTCOME"] * 100
        fig2 = px.bar(vt, x="VEHICLE_TYPE", y="Claim Rate %",
                      color="Claim Rate %", color_continuous_scale="Plasma",
                      template="plotly_dark")
        fig2.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### 🎓 Education vs Claim Rate")
        ed = car.groupby("EDUCATION")["OUTCOME"].mean().reset_index()
        ed["Claim Rate %"] = ed["OUTCOME"] * 100
        fig3 = px.bar(ed, x="EDUCATION", y="Claim Rate %",
                      color="Claim Rate %", color_continuous_scale="Blues",
                      template="plotly_dark")
        fig3.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown("#### 💳 Credit Score Distribution by Outcome")
        fig4 = px.box(car, x="OUTCOME", y="CREDIT_SCORE", color="OUTCOME",
                      color_discrete_map={0:"#10b981", 1:"#ef4444"},
                      template="plotly_dark",
                      labels={"OUTCOME": "Claim (0=No, 1=Yes)"})
        fig4.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=300)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### 🔢 Past Accidents vs Claim Outcome")
    pa = car.groupby(["PAST_ACCIDENTS", "OUTCOME"]).size().reset_index(name="Count")
    fig5 = px.bar(pa, x="PAST_ACCIDENTS", y="Count", color="OUTCOME",
                  barmode="group", color_discrete_map={0:"#10b981", 1:"#ef4444"},
                  template="plotly_dark",
                  labels={"OUTCOME": "Claim"})
    fig5.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=320)
    st.plotly_chart(fig5, use_container_width=True)

    with st.expander("📋 View Car Insurance Raw Data"):
        st.dataframe(car.head(300), use_container_width=True, height=320)
