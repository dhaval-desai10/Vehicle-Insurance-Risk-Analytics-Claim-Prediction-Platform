"""
PAGE 6 — GenAI Chat (RAG-style NL query over insurance data)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_car_data, load_motor_data
import plotly.express as px

st.set_page_config(page_title="GenAI Chat · InsureAI", page_icon="💬", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0f1117;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1d2e,#12151f);border-right:1px solid #2d3147;}
h1{background:linear-gradient(135deg,#06b6d4,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800!important;}
h2,h3{color:#e2e8f0!important;font-weight:700!important;}
#MainMenu,footer{visibility:hidden;}
.chat-bubble-user{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;
    padding:12px 18px;border-radius:18px 18px 4px 18px;margin:8px 0;max-width:80%;
    margin-left:auto;text-align:right;}
.chat-bubble-ai{background:#1e2130;border:1px solid #2d3147;color:#e2e8f0;
    padding:12px 18px;border-radius:18px 18px 18px 4px;margin:8px 0;max-width:90%;line-height:1.6;}
.chat-icon{font-size:24px;margin-right:8px;}
</style>""", unsafe_allow_html=True)

st.title("💬 GenAI Insurance Data Assistant")
st.caption("Ask natural language questions about your insurance portfolio — powered by data analytics & RAG")

# ── Load & Index Data ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Building knowledge base…")
def build_knowledge_base():
    car   = load_car_data()
    motor = load_motor_data(50_000)

    facts = {}
    # Motor data facts
    facts["total_policies"]      = len(motor)
    facts["claim_rate"]          = motor["CLAIM_FLAG"].mean() * 100
    facts["avg_premium"]         = motor["PREMIUM"].mean()
    facts["top_claim_type"]      = motor.groupby("TYPE_VEHICLE")["CLAIM_FLAG"].mean().idxmax()
    facts["top_claim_rate_pct"]  = motor.groupby("TYPE_VEHICLE")["CLAIM_FLAG"].mean().max() * 100
    facts["lowest_claim_type"]   = motor.groupby("TYPE_VEHICLE")["CLAIM_FLAG"].mean().idxmin()
    facts["avg_claim_paid"]      = motor[motor["CLAIM_PAID"]>0]["CLAIM_PAID"].mean()
    facts["max_claim"]           = motor["CLAIM_PAID"].max()
    facts["top_make"]            = motor["MAKE"].value_counts().idxmax()
    facts["premium_by_type"]     = motor.groupby("TYPE_VEHICLE")["PREMIUM"].mean().to_dict()
    facts["claim_rate_by_type"]  = (motor.groupby("TYPE_VEHICLE")["CLAIM_FLAG"].mean() * 100).to_dict()
    facts["avg_vehicle_age"]     = motor["VEHICLE_AGE"].mean()
    facts["loss_ratio"]          = motor["CLAIM_PAID"].sum() / motor["PREMIUM"].sum() * 100
    facts["top_usage"]           = motor.groupby("USAGE")["CLAIM_FLAG"].mean().idxmax() if "USAGE" in motor else "N/A"

    # Car insurance facts
    facts["car_claim_rate"]      = car["OUTCOME"].mean() * 100
    facts["high_risk_age"]       = car.groupby("AGE")["OUTCOME"].mean().idxmax()
    facts["safest_income"]       = car.groupby("INCOME")["OUTCOME"].mean().idxmin()
    facts["riskiest_income"]     = car.groupby("INCOME")["OUTCOME"].mean().idxmax()
    facts["avg_credit_no_claim"] = car[car["OUTCOME"]==0]["CREDIT_SCORE"].mean()
    facts["avg_credit_claim"]    = car[car["OUTCOME"]==1]["CREDIT_SCORE"].mean()
    facts["sports_claim_rate"]   = car[car["VEHICLE_TYPE"]=="sports car"]["OUTCOME"].mean() * 100 if "sports car" in car["VEHICLE_TYPE"].values else 0
    facts["sedan_claim_rate"]    = car[car["VEHICLE_TYPE"]=="sedan"]["OUTCOME"].mean() * 100 if "sedan" in car["VEHICLE_TYPE"].values else 0

    return facts, motor, car

facts, motor, car = build_knowledge_base()


def answer_query(question: str) -> tuple[str, object]:
    q = question.lower()
    chart = None

    # ── Routing ─────────────────────────────────────────────────────────────
    if any(w in q for w in ["highest claim rate","most claims","riskiest vehicle","highest risk vehicle"]):
        answer = (
            f"🏆 **{facts['top_claim_type']}** has the highest claim rate at "
            f"**{facts['top_claim_rate_pct']:.1f}%**.\n\n"
            f"In contrast, **{facts['lowest_claim_type']}** has the lowest claim rate. "
            f"This vehicle type shows significantly higher risk exposure in our portfolio."
        )
        top5 = sorted(facts["claim_rate_by_type"].items(), key=lambda x: x[1], reverse=True)[:8]
        df_c = pd.DataFrame(top5, columns=["Vehicle Type","Claim Rate %"])
        chart = px.bar(df_c, x="Claim Rate %", y="Vehicle Type", orientation="h",
                       color="Claim Rate %", color_continuous_scale="RdYlGn_r",
                       template="plotly_dark", title="Claim Rate by Vehicle Type")
        chart.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=300)

    elif any(w in q for w in ["average premium","avg premium","mean premium"]):
        if any(v in q for v in ["truck","bus","automobile","pickup","motorcycle","sedan","suv","sports"]):
            kw = next((v for v in ["truck","bus","automobile","pickup","motorcycle","sedan","suv","sports car"]
                       if v in q), None)
            match = {k: v for k, v in facts["premium_by_type"].items()
                     if kw and kw.lower() in k.lower()}
            if match:
                lines = "\n".join([f"  • **{k}**: {v:,.0f}" for k, v in match.items()])
                answer = f"💰 Average premiums for '{kw}' vehicles:\n{lines}"
            else:
                answer = f"💰 The overall average premium is **{facts['avg_premium']:,.0f}**."
        else:
            top5p = sorted(facts["premium_by_type"].items(), key=lambda x: x[1], reverse=True)[:6]
            lines = "\n".join([f"  • **{k}**: {v:,.0f}" for k, v in top5p])
            answer = (f"💰 Overall average premium: **{facts['avg_premium']:,.0f}**\n\n"
                      f"Top 6 by average premium:\n{lines}")
            df_p = pd.DataFrame(top5p, columns=["Vehicle Type","Avg Premium"])
            chart = px.bar(df_p, x="Avg Premium", y="Vehicle Type", orientation="h",
                           color="Avg Premium", color_continuous_scale="Blues",
                           template="plotly_dark", title="Avg Premium by Vehicle Type")
            chart.update_layout(paper_bgcolor="#1e2130", plot_bgcolor="#1e2130", height=280)

    elif any(w in q for w in ["high risk customer","risky customer","risk profile","high risk"]):
        answer = (
            f"⚠️ **High-risk customer profile** based on our dataset:\n\n"
            f"  • **Age group**: {facts['high_risk_age']} has the highest claim rate\n"
            f"  • **Income**: {facts['riskiest_income']} customers show higher risk\n"
            f"  • **Sports cars**: {facts['sports_claim_rate']:.1f}% claim rate vs "
            f"{facts['sedan_claim_rate']:.1f}% for sedans\n"
            f"  • **Credit score**: Claimants avg {facts['avg_credit_claim']:.3f} vs "
            f"{facts['avg_credit_no_claim']:.3f} for non-claimants\n\n"
            f"➡️ Use the **Customer Advisor** page for a personalised risk score."
        )

    elif any(w in q for w in ["claim rate","how many claims","percentage claim"]):
        answer = (
            f"📊 **Overall claim rates:**\n\n"
            f"  • Motor dataset: **{facts['claim_rate']:.1f}%** of policies had claims\n"
            f"  • Car insurance dataset: **{facts['car_claim_rate']:.1f}%** claim rate\n"
            f"  • Average claim paid: **{facts['avg_claim_paid']:,.0f}**\n"
            f"  • Maximum claim paid: **{facts['max_claim']:,.0f}**"
        )

    elif any(w in q for w in ["loss ratio","profitable","profitability"]):
        lr = facts["loss_ratio"]
        status = "✅ Profitable" if lr < 100 else "⚠️ Unprofitable"
        answer = (
            f"📉 **Portfolio Loss Ratio: {lr:.1f}%**\n\n"
            f"Status: **{status}**\n\n"
            f"A loss ratio below 100% means the company is collecting more premium than paying in claims. "
            f"The current ratio indicates {'good underwriting performance.' if lr < 80 else 'moderate pressure on profitability.' if lr < 100 else 'claims exceed premium income — urgent pricing review needed.'}"
        )

    elif any(w in q for w in ["total policies","how many policies","portfolio size"]):
        answer = (
            f"📋 **Total policies in portfolio: {facts['total_policies']:,}**\n\n"
            f"  • Vehicle types covered: {len(facts['premium_by_type'])} types\n"
            f"  • Most common make: **{facts['top_make']}**\n"
            f"  • Average vehicle age: **{facts['avg_vehicle_age']:.1f} years**"
        )

    elif any(w in q for w in ["best premium","recommend premium","optimal premium","pricing"]):
        answer = (
            f"💡 **Premium Optimization Insights:**\n\n"
            f"  • Risk-based pricing uses: Claim Probability × Expected Severity × Base Rate\n"
            f"  • High-risk vehicles ({facts['top_claim_type']}) command higher premiums\n"
            f"  • Current portfolio average: **{facts['avg_premium']:,.0f}**\n"
            f"  • Loss ratio is at **{facts['loss_ratio']:.1f}%** — {'premiums are adequate' if facts['loss_ratio']<100 else 'premiums may need adjustment'}\n\n"
            f"➡️ Visit the **Premium Optimizer** page for a real-time quote calculator."
        )

    elif any(w in q for w in ["safe","safest","low risk","good driver"]):
        answer = (
            f"✅ **Safest customer segment:**\n\n"
            f"  • Income: **{facts['safest_income']} class** shows the lowest claim rate\n"
            f"  • Vehicle: **{facts['lowest_claim_type']}** has the lowest claim rate\n"
            f"  • Good credit score (>0.65) strongly correlates with fewer claims\n"
            f"  • Zero speeding violations and past accidents = lowest risk\n\n"
            f"💡 Offer loyalty discounts and multi-policy bundles to these customers."
        )

    elif any(w in q for w in ["age","young driver","old driver","elderly"]):
        answer = (
            f"👤 **Age & Claim Rate analysis:**\n\n"
            f"  • Highest risk age group: **{facts['high_risk_age']}**\n"
            f"  • Young drivers (16-25) typically have the highest claim frequency due to inexperience\n"
            f"  • Senior drivers (65+) may face increased risk due to reaction time factors\n"
            f"  • Middle-aged drivers (40-64) generally have the best driving record\n\n"
            f"📊 View the **EDA Dashboard → Car Insurance** tab for full age breakdown."
        )

    else:
        answer = (
            f"🤖 I can answer questions like:\n\n"
            f"  • *Which vehicle type has the highest claim rate?*\n"
            f"  • *What is the average premium for trucks?*\n"
            f"  • *Which customers are high risk?*\n"
            f"  • *What is the portfolio loss ratio?*\n"
            f"  • *How many total policies do we have?*\n"
            f"  • *What is the claim rate?*\n"
            f"  • *Who are safest customers?*\n\n"
            f"Try one of the **suggested questions** below, or type your own!"
        )

    return answer, chart


# ── Chat Interface ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Suggested questions
st.markdown("**💡 Suggested Questions:**")
sq_cols = st.columns(3)
suggested = [
    "Which vehicle type has the highest claim rate?",
    "What is the average premium for trucks?",
    "Which customers are high risk?",
    "What is the portfolio loss ratio?",
    "How many total policies do we have?",
    "Who are the safest customers?",
]
for i, sq in enumerate(suggested):
    if sq_cols[i % 3].button(sq, key=f"sq_{i}"):
        st.session_state.chat_history.append(("user", sq))
        ans, chart = answer_query(sq)
        st.session_state.chat_history.append(("ai", ans, chart))

st.markdown("---")

# Chat display
chat_container = st.container()
with chat_container:
    for entry in st.session_state.chat_history:
        if entry[0] == "user":
            st.markdown(f'<div class="chat-bubble-user">🧑 {entry[1]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-ai">🤖 <strong>InsureAI</strong><br><br>{entry[1]}</div>',
                        unsafe_allow_html=True)
            if len(entry) > 2 and entry[2] is not None:
                st.plotly_chart(entry[2], use_container_width=True)

# Input
with st.form("chat_form", clear_on_submit=True):
    col_inp, col_btn = st.columns([5, 1])
    user_input = col_inp.text_input("Ask a question about your insurance data…",
                                    placeholder="e.g. What is the average premium for buses?",
                                    label_visibility="collapsed")
    submitted = col_btn.form_submit_button("Send →", type="primary")

if submitted and user_input.strip():
    st.session_state.chat_history.append(("user", user_input))
    ans, chart = answer_query(user_input)
    st.session_state.chat_history.append(("ai", ans, chart))
    st.rerun()

if st.button("🗑️ Clear Chat", type="secondary"):
    st.session_state.chat_history = []
    st.rerun()
