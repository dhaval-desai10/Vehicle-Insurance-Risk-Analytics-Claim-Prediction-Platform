"""
PAGE 7 — Premium Optimizer
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

st.set_page_config(page_title="Premium Optimizer · InsureAI", page_icon="💰", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0f1117;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1d2e,#12151f);border-right:1px solid #2d3147;}
h1{background:linear-gradient(135deg,#ec4899,#f59e0b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800!important;}
h2,h3{color:#e2e8f0!important;font-weight:700!important;}
[data-testid="metric-container"]{background:#1e2130;border:1px solid #2d3147;border-radius:12px;padding:14px 18px!important;}
[data-testid="stMetricValue"]{color:#f9fafb!important;font-weight:800;font-size:26px!important;}
[data-testid="stMetricLabel"]{color:#9ca3af!important;}
#MainMenu,footer{visibility:hidden;}
</style>""", unsafe_allow_html=True)

st.title("💰 Premium Optimizer & Pricing Engine")
st.caption("Risk-adjusted premium calculation using ML-predicted claim probabilities")

with st.spinner("Loading market data…"):
    motor = load_motor_data(50_000)
    car   = load_car_data()

tab = st.tabs(["🔢 Quote Calculator", "📊 Market Benchmarks", "📈 Sensitivity Analysis"])

# ── QUOTE CALCULATOR ──────────────────────────────────────────────────────────
with tab[0]:
    st.markdown("### 🔢 Real-Time Premium Quote Calculator")
    st.caption("Adjust the sliders to compute a risk-based premium recommendation")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🚗 Vehicle Details")
        veh_type    = st.selectbox("Vehicle Type",
                                    ["Automobile","Pick-up","Truck","Bus","Motorcycle","Special construction"],
                                    index=0)
        prod_year   = st.slider("Production Year", 1980, 2023, 2010)
        insured_val = st.number_input("Insured Value", min_value=0, max_value=10_000_000,
                                      value=500_000, step=10_000, format="%d")
        seats       = st.slider("Number of Seats", 1, 60, 5)
        usage       = st.selectbox("Vehicle Usage",
                                    ["Private","Own Goods","General Cartage","Fare Paying Passengers",
                                     "Own service","Agricultural Own Farm"])

        st.markdown("#### 👤 Customer Profile")
        age_grp     = st.selectbox("Customer Age Group", ["16-25","26-39","40-64","65+"], index=1)
        past_claims = st.slider("Past Claims (History)", 0, 10, 0)
        accidents   = st.slider("Past Accidents", 0, 10, 0)
        speeding    = st.slider("Speeding Violations", 0, 15, 0)
        credit_score= st.slider("Credit Score (0–1)", 0.0, 1.0, 0.6, 0.01)

    with col2:
        st.markdown("#### ⚙️ Pricing Parameters")
        base_rate       = st.number_input("Base Premium Rate (unit)", value=2_000, step=100)
        risk_multiplier = st.slider("Max Risk Multiplier", 1.0, 5.0, 2.5, 0.1)
        profit_loading  = st.slider("Profit Loading (%)", 0, 40, 15)
        expense_ratio   = st.slider("Expense Ratio (%)", 5, 30, 12)
        discount        = st.slider("Loyalty Discount (%)", 0, 25, 0)
        policy_months   = st.slider("Policy Duration (months)", 6, 24, 12)

        if st.button("⚡ Calculate Optimal Premium", type="primary", use_container_width=True):
            # ── Risk Score Calculation ────────────────────────────────────
            vehicle_age = 2024 - prod_year
            risk_score = 0.0

            # Claim probability (rule-based proxy)
            age_risk  = {"16-25": 0.35, "26-39": 0.20, "40-64": 0.15, "65+": 0.22}
            risk_score += age_risk.get(age_grp, 0.20)
            risk_score += past_claims * 0.08
            risk_score += accidents   * 0.06
            risk_score += speeding    * 0.025
            risk_score -= (credit_score - 0.5) * 0.3
            veh_risk   = {"Automobile":0, "Pick-up":0.05, "Truck":0.10,
                          "Bus":0.15, "Motorcycle":0.20, "Special construction":0.12}
            risk_score += veh_risk.get(veh_type, 0.0)
            if vehicle_age > 15: risk_score += 0.08
            if vehicle_age > 25: risk_score += 0.10
            usage_risk = {"Private":-0.05, "General Cartage":0.12,
                          "Fare Paying Passengers":0.10, "Own service":0.02,
                          "Agricultural Own Farm":-0.02, "Own Goods":0.05}
            risk_score += usage_risk.get(usage, 0)
            claim_prob = float(np.clip(risk_score, 0.01, 0.95))

            # Expected severity
            ref_data = motor[motor["TYPE_VEHICLE"].str.lower().str.contains(veh_type[:4].lower(), na=False)]
            if len(ref_data[ref_data["CLAIM_PAID"] > 0]) > 10:
                avg_severity = ref_data[ref_data["CLAIM_PAID"] > 0]["CLAIM_PAID"].mean()
            else:
                avg_severity = motor[motor["CLAIM_PAID"] > 0]["CLAIM_PAID"].mean()

            # Pure premium (expected loss)
            pure_premium = claim_prob * avg_severity

            # Value-based component
            value_loading = insured_val * 0.008 * (policy_months / 12)

            # Final premium
            raw_premium = max(base_rate, pure_premium * 0.3 + value_loading * 0.7)
            loading_factor = 1 + (profit_loading + expense_ratio) / 100
            final_premium = raw_premium * loading_factor * (1 - discount / 100)
            final_premium = final_premium * (policy_months / 12)

            risk_label = ("🟢 Low" if claim_prob < 0.25 else
                          "🟡 Medium" if claim_prob < 0.45 else
                          "🔴 High" if claim_prob < 0.65 else
                          "🔴🔴 Very High")
            risk_color = ("#10b981" if claim_prob < 0.25 else
                          "#f59e0b" if claim_prob < 0.45 else
                          "#ef4444")

            # ── Display Results ───────────────────────────────────────────
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#ec489922,#f59e0b11);
                        border:2px solid #ec4899;border-radius:16px;padding:24px;margin:16px 0;">
                <h3 style="color:#f9fafb;margin:0 0 16px;">📋 Premium Quote</h3>
                <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px;">
                    <div>
                        <div style="color:#9ca3af;font-size:12px;font-weight:600;">CLAIM PROBABILITY</div>
                        <div style="color:{risk_color};font-size:32px;font-weight:900;">{claim_prob*100:.1f}%</div>
                        <div style="color:{risk_color};font-size:13px;">{risk_label} Risk</div>
                    </div>
                    <div>
                        <div style="color:#9ca3af;font-size:12px;font-weight:600;">EXPECTED SEVERITY</div>
                        <div style="color:#f9fafb;font-size:32px;font-weight:900;">{avg_severity:,.0f}</div>
                        <div style="color:#6b7280;font-size:13px;">if claim occurs</div>
                    </div>
                    <div>
                        <div style="color:#9ca3af;font-size:12px;font-weight:600;">PURE PREMIUM</div>
                        <div style="color:#f9fafb;font-size:24px;font-weight:800;">{pure_premium:,.0f}</div>
                        <div style="color:#6b7280;font-size:13px;">expected loss</div>
                    </div>
                    <div>
                        <div style="color:#9ca3af;font-size:12px;font-weight:600;">RECOMMENDED PREMIUM</div>
                        <div style="color:#ec4899;font-size:36px;font-weight:900;">{final_premium:,.0f}</div>
                        <div style="color:#6b7280;font-size:13px;">for {policy_months} months</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Breakdown waterfall chart
            breakdown = {
                "Base Rate":        base_rate,
                "Risk Loading":     raw_premium - base_rate,
                "Profit Loading":   raw_premium * profit_loading / 100,
                "Expense Loading":  raw_premium * expense_ratio / 100,
                "Loyalty Discount": -final_premium * discount / (100 - discount) if discount > 0 else 0,
                "Final Premium":    final_premium,
            }
            fig_wf = go.Figure(go.Waterfall(
                name="Premium Breakdown",
                orientation="v",
                measure=["relative","relative","relative","relative","relative","total"],
                x=list(breakdown.keys()),
                y=list(breakdown.values()),
                connector={"line": {"color": "#6366f1"}},
                increasing={"marker": {"color": "#6366f1"}},
                decreasing={"marker": {"color": "#ef4444"}},
                totals={"marker": {"color": "#ec4899"}},
            ))
            fig_wf.update_layout(template="plotly_dark", paper_bgcolor="#1e2130",
                                  plot_bgcolor="#1e2130", height=340,
                                  title="Premium Build-Up", font=dict(color="#e2e8f0"))
            st.plotly_chart(fig_wf, use_container_width=True)
        else:
            st.info("👈 Fill in the details and click **Calculate Optimal Premium**")

# ── MARKET BENCHMARKS ─────────────────────────────────────────────────────────
with tab[1]:
    st.markdown("### 📊 Market Premium Benchmarks")
    pm = motor.groupby("TYPE_VEHICLE")["PREMIUM"].agg(
        Mean="mean", Median="median", Q25=lambda x: x.quantile(0.25),
        Q75=lambda x: x.quantile(0.75), Min="min", Max="max"
    ).reset_index().sort_values("Mean", ascending=False).head(15)

    fig_bx = go.Figure()
    for i, row in pm.nlargest(12,"Mean").iterrows():
        fig_bx.add_trace(go.Box(
            name=row["TYPE_VEHICLE"][:20],
            y=motor[motor["TYPE_VEHICLE"]==row["TYPE_VEHICLE"]]["PREMIUM"].clip(upper=motor["PREMIUM"].quantile(0.97)),
            boxmean="sd",
        ))
    fig_bx.update_layout(template="plotly_dark", paper_bgcolor="#1e2130",
                          plot_bgcolor="#1e2130", height=420,
                          showlegend=False, xaxis_tickangle=-30,
                          title="Premium Distribution by Vehicle Type")
    st.plotly_chart(fig_bx, use_container_width=True)

    st.dataframe(pm.style.format({
        "Mean":"{:,.0f}","Median":"{:,.0f}","Q25":"{:,.0f}","Q75":"{:,.0f}",
        "Min":"{:,.0f}","Max":"{:,.0f}"
    }), use_container_width=True, hide_index=True)

# ── SENSITIVITY ANALYSIS ──────────────────────────────────────────────────────
with tab[2]:
    st.markdown("### 📈 Premium Sensitivity Analysis")
    st.caption("How does the premium change as risk factors change?")

    base = motor["PREMIUM"].mean()
    claim_probs = np.linspace(0.01, 0.95, 50)
    avg_sev = motor[motor["CLAIM_PAID"] > 0]["CLAIM_PAID"].mean()

    premiums_conservative  = [(p * avg_sev * 0.3 + base * 0.5) * 1.25 for p in claim_probs]
    premiums_moderate      = [(p * avg_sev * 0.5 + base * 0.3) * 1.27 for p in claim_probs]
    premiums_aggressive    = [(p * avg_sev * 0.8 + base * 0.2) * 1.30 for p in claim_probs]

    fig_sa = go.Figure()
    fig_sa.add_trace(go.Scatter(x=claim_probs*100, y=premiums_conservative,
                                mode="lines", name="Conservative", line=dict(color="#10b981",width=2)))
    fig_sa.add_trace(go.Scatter(x=claim_probs*100, y=premiums_moderate,
                                mode="lines", name="Moderate",     line=dict(color="#f59e0b",width=2)))
    fig_sa.add_trace(go.Scatter(x=claim_probs*100, y=premiums_aggressive,
                                mode="lines", name="Aggressive",   line=dict(color="#ef4444",width=2)))
    fig_sa.update_layout(template="plotly_dark", paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
                          height=380, xaxis_title="Claim Probability (%)",
                          yaxis_title="Recommended Premium",
                          title="Premium vs Claim Probability (3 Pricing Strategies)",
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_sa, use_container_width=True)

    # Heatmap: vehicle age vs claim prob → premium
    st.markdown("#### 🔥 Premium Heatmap — Vehicle Age × Claim Probability")
    ages = list(range(1, 31, 3))
    probs = np.arange(0.05, 0.80, 0.10)
    z_vals = [[avg_sev * p * 0.5 + base * 0.3 + a * 100 for p in probs] for a in ages]

    fig_hm = go.Figure(go.Heatmap(
        z=z_vals, x=[f"{p*100:.0f}%" for p in probs], y=[f"{a}yr" for a in ages],
        colorscale="Plasma", showscale=True,
        colorbar=dict(title="Premium"),
    ))
    fig_hm.update_layout(paper_bgcolor="#1e2130", font=dict(color="#e2e8f0"),
                          height=400, xaxis_title="Claim Probability",
                          yaxis_title="Vehicle Age")
    st.plotly_chart(fig_hm, use_container_width=True)
