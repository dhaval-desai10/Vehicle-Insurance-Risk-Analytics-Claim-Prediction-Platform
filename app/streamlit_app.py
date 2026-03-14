r"""
Main Streamlit entry point — Home / Landing page.
Run: streamlit run app/streamlit_app.py  (from d:\kenexai\)
"""
import streamlit as st

st.set_page_config(
    page_title="InsureAI — Vehicle Risk Analytics Platform",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — professional dark theme ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Apply font to headers and non-icon elements safely */
h1, h2, h3, h4, h5, h6, p, ul li, span:not(.material-symbols-rounded), div.stMarkdown, div[data-testid="stMetricValue"] { 
    font-family: 'Inter', sans-serif !important; 
}

/* Main background */
.stApp { background: #0b0f19; }
section[data-testid="stSidebar"] > div { background: #0b0f19; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(195deg, #111827 0%, #0b0f19 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.15);
    min-width: 260px !important;
}
[data-testid="stSidebarNav"] { padding-top: 0; }
[data-testid="stSidebarNav"] ul { gap: 2px; }
[data-testid="stSidebarNav"] li a {
    border-radius: 10px; padding: 9px 16px; font-size: 14px;
    font-weight: 500; color: #94a3b8 !important;
    transition: all .15s ease;
}
[data-testid="stSidebarNav"] li a:hover {
    background: rgba(99,102,241,0.1); color: #c7d2fe !important;
}
[data-testid="stSidebarNav"] li a[aria-selected="true"] {
    background: linear-gradient(90deg,rgba(99,102,241,0.18),rgba(139,92,246,0.1));
    border-left: 3px solid #6366f1; color: #e0e7ff !important;
    font-weight: 600;
}
[data-testid="stSidebarNav"] li a span { font-size: 14px; }

/* Headers */
h1 {
    background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 900 !important; letter-spacing: -0.5px;
}
h2, h3, h4 { color: #e2e8f0 !important; font-weight: 700 !important; }
p, span, label, .stMarkdown { color: #cbd5e1; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(30,33,48,0.9), rgba(20,23,38,0.9));
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 14px; padding: 18px 22px !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
[data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 800 !important; font-size: 28px !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 12px !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.5px; }
[data-testid="stMetricDelta"] { font-size: 13px !important; }

/* Charts */
.js-plotly-plot { border-radius: 14px; overflow: hidden; }

/* Divider */
hr { border-color: rgba(99,102,241,0.12) !important; margin: 20px 0 !important; }

/* Tabs */
button[data-baseweb="tab"] {
    color: #64748b !important; font-weight: 600; font-size: 14px;
    border-bottom: 2px solid transparent;
    padding: 12px 20px !important;
}
button[data-baseweb="tab"]:hover { color: #a5b4fc !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom-color: #6366f1 !important;
}

/* Buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none; font-weight: 700; border-radius: 10px;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35);
    transition: all .2s ease;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.5);
    transform: translateY(-1px);
}

/* Select boxes & inputs */
[data-baseweb="select"] > div { background: #111827 !important; border-color: rgba(99,102,241,0.2) !important; }
.stSlider > div { color: #cbd5e1; }

/* Dataframes */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Expander */
details { background: #111827; border: 1px solid rgba(99,102,241,0.12); border-radius: 12px; }

/* Hide branding but keep functional elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar branding ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:24px 0 8px;">
        <div style="font-size:42px;margin-bottom:6px;">🚗</div>
        <div style="font-size:22px;font-weight:900;
                    background:linear-gradient(135deg,#818cf8,#38bdf8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    letter-spacing:-0.5px;">
            InsureAI
        </div>
        <div style="font-size:11px;color:#64748b;margin-top:2px;font-weight:500;
                    letter-spacing:1.5px;text-transform:uppercase;">
            Risk Analytics Platform
        </div>
    </div>
    <div style="margin:12px 16px;height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,0.3),transparent);"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin:16px 8px;padding:14px 16px;background:rgba(16,185,129,0.08);
                border:1px solid rgba(16,185,129,0.2);border-radius:12px;">
        <div style="color:#34d399;font-size:11px;font-weight:700;letter-spacing:0.5px;
                    text-transform:uppercase;margin-bottom:8px;">Datasets Active</div>
        <div style="color:#94a3b8;font-size:12px;line-height:1.8;">
            ✅ Car_Insurance_Claim.csv<br>
            ✅ motor_data14-2018.csv
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin:8px 8px;padding:14px 16px;background:rgba(99,102,241,0.08);
                border:1px solid rgba(99,102,241,0.15);border-radius:12px;">
        <div style="color:#818cf8;font-size:11px;font-weight:700;letter-spacing:0.5px;
                    text-transform:uppercase;margin-bottom:8px;">ML Models</div>
        <div style="color:#94a3b8;font-size:12px;line-height:1.8;">
            🧠 3 Classifiers trained<br>
            📈 1 Regressor trained<br>
            🔵 K-Means (k=4) ready
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:40px 0 20px;">
    <h1 style="font-size:48px !important;line-height:1.1;margin:0;">
        Vehicle Insurance<br>Risk Analytics Platform
    </h1>
    <p style="color:#94a3b8;font-size:17px;margin-top:14px;max-width:620px;line-height:1.7;">
        AI-powered platform to analyze insurance portfolios, predict claim likelihood,
        segment customers by risk level, and optimize premium pricing.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Quick Stats ───────────────────────────────────────────────────────────────
with st.spinner("Loading datasets..."):
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from utils.data_loader import load_car_data, load_motor_data
        car = load_car_data()
        motor = load_motor_data()
        loaded = True
    except Exception as e:
        st.error(f"Could not load data: {e}")
        loaded = False

if loaded:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Policies", f"{len(motor):,}", "motor dataset")
    c2.metric("Vehicle Types", f"{motor['TYPE_VEHICLE'].nunique()}" if 'TYPE_VEHICLE' in motor else "N/A", "unique categories")
    c3.metric("Claim Rate", f"{motor['CLAIM_FLAG'].mean()*100:.1f}%" if 'CLAIM_FLAG' in motor else "N/A", "policies with claims")
    c4.metric("Avg Premium", f"{motor['PREMIUM'].mean():,.0f}" if 'PREMIUM' in motor else "N/A", "per policy")

st.markdown("---")

# ── Feature Cards ─────────────────────────────────────────────────────────────
st.markdown("### Platform Modules")

modules = [
    ("📊", "EDA Dashboard",      "Data quality, distributions, correlation analysis, and missing value reports.",     "#6366f1"),
    ("👔", "Insurance Manager",   "Executive KPIs — total policies, claim rate, loss ratio, and revenue trends.",      "#8b5cf6"),
    ("⚠️",  "Risk Analyst",       "Risk heatmaps, claim probability matrix, severity analysis, and trend forecast.", "#f59e0b"),
    ("🤝", "Customer Advisor",    "Interactive risk score calculator, segment analysis, and premium benchmarks.",     "#10b981"),
    ("🤖", "ML Prediction",       "Live 3-model predictions with ensemble, clustering, and claim amount regression.","#ef4444"),
    ("💬", "GenAI Chat",          "Natural language Q&A over your insurance data with chart responses.",               "#06b6d4"),
    ("💰", "Premium Optimizer",   "Risk-adjusted premium quoting engine with sensitivity analysis.",                  "#ec4899"),
]

row1 = st.columns(4)
row2 = st.columns(4)
all_cols = row1 + row2[:3]

for i, (icon, title, desc, color) in enumerate(modules):
    with all_cols[i]:
        st.markdown(f"""
        <div style="background:linear-gradient(145deg,rgba(17,24,39,0.95),rgba(11,15,25,0.9));
                    border:1px solid {color}20;
                    border-radius:16px;padding:22px;margin-bottom:12px;
                    min-height:170px;
                    box-shadow:0 4px 24px rgba(0,0,0,0.2);
                    transition:all .2s ease;">
            <div style="width:44px;height:44px;background:{color}15;border-radius:12px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:24px;margin-bottom:14px;">{icon}</div>
            <div style="color:#f1f5f9;font-weight:700;font-size:15px;margin-bottom:8px;">{title}</div>
            <div style="color:#64748b;font-size:13px;line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#475569;font-size:12px;padding:8px 0;letter-spacing:0.3px;">
    InsureAI Platform  ·  Vehicle Insurance Risk Analytics & Claim Prediction  ·  Built with Streamlit + Plotly + Scikit-learn
</div>
""", unsafe_allow_html=True)
