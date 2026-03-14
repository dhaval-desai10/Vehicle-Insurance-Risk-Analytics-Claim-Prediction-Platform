"""
PAGE 5 — ML Prediction (uses saved .pkl models from src/models/saved/)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import roc_curve

st.set_page_config(page_title="ML Prediction · InsureAI", page_icon="🤖", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0f1117;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1d2e,#12151f);border-right:1px solid #2d3147;}
h1{background:linear-gradient(135deg,#ef4444,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800!important;}
h2,h3{color:#e2e8f0!important;font-weight:700!important;}
[data-testid="metric-container"]{background:#1e2130;border:1px solid #2d3147;border-radius:12px;padding:14px 18px!important;}
[data-testid="stMetricValue"]{color:#f9fafb!important;font-weight:800;font-size:26px!important;}
[data-testid="stMetricLabel"]{color:#9ca3af!important;}
#MainMenu,footer{visibility:hidden;}
</style>""", unsafe_allow_html=True)

st.title("🤖 Machine Learning Prediction Engine")
st.caption("Powered by trained Random Forest, Gradient Boosting & Logistic Regression models")

SAVE_DIR = Path(__file__).parent.parent.parent / "src" / "models" / "saved"

# ── Load Saved Models ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained ML models...")
def load_saved_models():
    try:
        models = {
            "classifier":     joblib.load(SAVE_DIR / "classifier.pkl"),
            "scaler_clf":     joblib.load(SAVE_DIR / "scaler_clf.pkl"),
            "clf_features":   joblib.load(SAVE_DIR / "clf_features.pkl"),
            "clf_rf":         joblib.load(SAVE_DIR / "clf_random_forest.pkl"),
            "clf_gb":         joblib.load(SAVE_DIR / "clf_gradient_boosting.pkl"),
            "clf_lr":         joblib.load(SAVE_DIR / "clf_logistic_regression.pkl"),
            "clf_results":    joblib.load(SAVE_DIR / "clf_results.pkl"),
            "regressor":      joblib.load(SAVE_DIR / "regressor.pkl"),
            "reg_features":   joblib.load(SAVE_DIR / "reg_features.pkl"),
            "kmeans":         joblib.load(SAVE_DIR / "kmeans.pkl"),
            "scaler_km":      joblib.load(SAVE_DIR / "scaler_km.pkl"),
            "km_features":    joblib.load(SAVE_DIR / "km_features.pkl"),
            "cluster_labels": joblib.load(SAVE_DIR / "cluster_labels.pkl"),
            "cluster_profiles": joblib.load(SAVE_DIR / "cluster_profiles.pkl"),
        }
        return models, None
    except FileNotFoundError as e:
        return None, str(e)

models, err = load_saved_models()

if err:
    st.error(f"⚠️ Model files not found: `{err}`")
    st.info("Run in terminal: `python src/models/train_models.py`")
    st.stop()

st.success(f"✅ Trained models loaded from `src/models/saved/` — {len(list(SAVE_DIR.glob('*.pkl')))} files")

tab = st.tabs(["🔮 Live Prediction", "📈 Model Performance", "🔵 Customer Clusters", "💸 Claim Amount"])

# ── LIVE PREDICTION ───────────────────────────────────────────────────────────
with tab[0]:
    st.markdown("### 🔮 Real-Time Claim Prediction")
    st.caption("Enter customer details — all 3 trained models predict simultaneously")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Customer Info**")
        p_age_n    = st.slider("Age", 17, 80, 32)
        p_exp      = st.slider("Driving Experience (years)", 0, 40, 10)
        p_income   = st.select_slider("Income Level", ["poverty","working class","middle class","upper class"], value="middle class")
        p_credit   = st.slider("Credit Score", 0.0, 1.0, 0.6, 0.01)
    with col2:
        st.markdown("**Driving History**")
        p_mileage  = st.slider("Annual Mileage", 2000, 30000, 12000, 500)
        p_speeding = st.slider("Speeding Violations", 0, 15, 0)
        p_duis     = st.slider("DUIs", 0, 5, 0)
        p_accidents= st.slider("Past Accidents", 0, 10, 0)
    with col3:
        st.markdown("**Computed Values**")
        inc_map = {"poverty":0,"working class":1,"middle class":2,"upper class":3}
        risk_sc = (p_accidents*15 + p_speeding*5 + p_duis*20 + (1-p_credit)*20)
        risk_sc = float(np.clip(risk_sc, 0, 100))
        st.metric("Risk Score",         f"{risk_sc:.1f} / 100")
        st.metric("Income Score",       f"{inc_map[p_income]}")
        st.metric("Exp. Numeric",       f"{min(p_exp, 35)}")
        risk_tier = ("🟢 Low" if risk_sc<25 else "🟡 Medium" if risk_sc<50 else "🔴 High" if risk_sc<75 else "🔴🔴 Very High")
        st.info(f"Pre-calc Risk: **{risk_tier}**")

    if st.button("🚀 Run All 3 Models", type="primary", use_container_width=True):
        row_data = {
            "age_numeric":          float(p_age_n),
            "exp_numeric":          float(min(p_exp, 35)),
            "income_score":         float(inc_map[p_income]),
            "credit_score":         float(p_credit),
            "annual_mileage":       float(p_mileage),
            "speeding_violations":  float(p_speeding),
            "duis":                 float(p_duis),
            "past_accidents":       float(p_accidents),
            "risk_score":           float(risk_sc),
        }
        features  = models["clf_features"]
        scaler    = models["scaler_clf"]
        X = pd.DataFrame([row_data])[features].fillna(0)
        Xs = scaler.transform(X)

        clf_models = {
            "Random Forest":       models["clf_rf"],
            "Gradient Boosting":   models["clf_gb"],
            "Logistic Regression": models["clf_lr"],
        }
        colors = ["#6366f1","#f59e0b","#10b981"]
        probas = {}

        st.markdown("---")
        st.markdown("#### 🎯 Prediction Results")
        r_cols = st.columns(3)
        for i, (mname, mdl) in enumerate(clf_models.items()):
            from sklearn.linear_model import LogisticRegression as LR
            prob = mdl.predict_proba(Xs)[0][1] if isinstance(mdl, LR) else mdl.predict_proba(X)[0][1]
            probas[mname] = prob
            verdict = "🔴 CLAIM LIKELY" if prob>=0.5 else "🟢 NO CLAIM"
            clr = "#ef4444" if prob>=0.5 else "#10b981"
            with r_cols[i]:
                st.markdown(f"""
                <div style="background:#1e2130;border:1px solid {colors[i]}44;
                            border-top:3px solid {colors[i]};border-radius:14px;
                            padding:20px;text-align:center;">
                    <div style="color:{colors[i]};font-size:12px;font-weight:700;
                                text-transform:uppercase;letter-spacing:.5px;">{mname}</div>
                    <div style="color:{clr};font-size:48px;font-weight:900;margin:8px 0;">
                        {prob*100:.1f}%
                    </div>
                    <div style="color:{clr};font-weight:700;">{verdict}</div>
                </div>""", unsafe_allow_html=True)

        avg_prob = float(np.mean(list(probas.values())))
        clr_e = "#ef4444" if avg_prob>=0.5 else "#10b981"
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#6366f122,#8b5cf622);
                    border:2px solid #6366f1;border-radius:16px;padding:24px;
                    text-align:center;margin:16px 0;">
            <div style="color:#6366f1;font-size:13px;font-weight:700;letter-spacing:.5px;">
                ENSEMBLE PROBABILITY (3-MODEL AVERAGE)
            </div>
            <div style="font-size:60px;font-weight:900;color:{clr_e};">{avg_prob*100:.1f}%</div>
            <div style="color:#9ca3af;">{'Claim is likely — recommend higher premium' if avg_prob>=0.5 else 'Low risk — standard premium applicable'}</div>
        </div>""", unsafe_allow_html=True)

        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_prob*100,
            delta={"reference": 50, "valueformat":".1f","suffix":"%"},
            title={"text":"Ensemble Claim Probability","font":{"color":"#e2e8f0","size":15}},
            gauge={
                "axis": {"range":[0,100],"ticksuffix":"%","tickcolor":"#9ca3af"},
                "bar":  {"color": clr_e},
                "steps":[
                    {"range":[0,25],  "color":"rgba(16,185,129,0.2)"},
                    {"range":[25,50], "color":"rgba(245,158,11,0.2)"},
                    {"range":[50,75], "color":"rgba(239,68,68,0.2)"},
                    {"range":[75,100],"color":"rgba(127,29,29,0.25)"},
                ],
                "threshold":{"line":{"color":"white","width":3},"value":50}
            },
            number={"suffix":"%","font":{"color":"#f9fafb","size":44}}
        ))
        fig_g.update_layout(paper_bgcolor="#1e2130",font_color="#e2e8f0",height=280,margin=dict(t=40,b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        # Feature importance
        rf = models["clf_rf"]
        fi_df = pd.DataFrame({"Feature":features,"Importance":rf.feature_importances_}).sort_values("Importance")
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Purples",
                        template="plotly_dark", title="Feature Importance (Random Forest)")
        fig_fi.update_layout(paper_bgcolor="#1e2130",plot_bgcolor="#1e2130",height=320,showlegend=False)
        st.plotly_chart(fig_fi, use_container_width=True)

# ── MODEL PERFORMANCE ─────────────────────────────────────────────────────────
with tab[1]:
    st.markdown("### 📈 Trained Model Comparison")
    res = models["clf_results"]

    c1,c2,c3 = st.columns(3)
    for i,(n,v) in enumerate(res.items()):
        [c1,c2,c3][i].metric(n, f"AUC: {v['auc']:.4f}", f"Acc: {v['acc']*100:.2f}%")

    st.markdown("""
    <div style="background:#1e2130;border:1px solid #2d3147;border-radius:12px;padding:20px;margin:16px 0;">
        <h4 style="color:#e2e8f0;margin:0 0 12px;">📊 Model Details</h4>
        <table style="width:100%;color:#e2e8f0;border-collapse:collapse;">
            <tr style="border-bottom:1px solid #2d3147;">
                <th style="text-align:left;padding:8px;color:#9ca3af;">Model</th>
                <th style="padding:8px;color:#9ca3af;">AUC Score</th>
                <th style="padding:8px;color:#9ca3af;">Accuracy</th>
                <th style="padding:8px;color:#9ca3af;">Type</th>
            </tr>
    """ + "".join([
        f"""<tr style="border-bottom:1px solid #1a1d2e;">
                <td style="padding:8px;font-weight:600;">{n}</td>
                <td style="padding:8px;text-align:center;color:#6366f1;font-weight:700;">{v['auc']:.4f}</td>
                <td style="padding:8px;text-align:center;">{v['acc']*100:.2f}%</td>
                <td style="padding:8px;text-align:center;color:#9ca3af;">Ensemble</td>
            </tr>"""
        for n,v in res.items()
    ]) + """
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Model performance bar chart
    comp_df = pd.DataFrame([
        {"Model":n, "AUC":v["auc"]*100, "Accuracy":v["acc"]*100} for n,v in res.items()
    ])
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="AUC (%)", x=comp_df["Model"], y=comp_df["AUC"], marker_color="#6366f1"))
    fig_cmp.add_trace(go.Bar(name="Accuracy (%)", x=comp_df["Model"], y=comp_df["Accuracy"], marker_color="#10b981"))
    fig_cmp.update_layout(barmode="group",template="plotly_dark",
                           paper_bgcolor="#1e2130",plot_bgcolor="#1e2130",height=320,
                           legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("#### 🔢 Classification Report Info")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style="background:#1e2130;border:1px solid #2d3147;border-radius:12px;padding:16px;">
            <div style="color:#6366f1;font-weight:700;margin-bottom:8px;">Training Data</div>
            <div style="color:#9ca3af;font-size:13px;line-height:1.8;">
                Dataset: Car_Insurance_Claim.csv<br>
                Records: 10,000 rows<br>
                Train/Test Split: 80% / 20%<br>
                Target: OUTCOME (0=No Claim, 1=Claim)<br>
                Stratified: Yes (class imbalance handled)
            </div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style="background:#1e2130;border:1px solid #2d3147;border-radius:12px;padding:16px;">
            <div style="color:#10b981;font-weight:700;margin-bottom:8px;">Feature Set</div>
            <div style="color:#9ca3af;font-size:13px;line-height:1.8;">
                age_numeric · exp_numeric · income_score<br>
                credit_score · annual_mileage<br>
                speeding_violations · duis · past_accidents<br>
                risk_score<br>
                StandardScaler applied to Logistic Regression
            </div>
        </div>""", unsafe_allow_html=True)

# ── CUSTOMER CLUSTERS ─────────────────────────────────────────────────────────
with tab[2]:
    st.markdown("### 🔵 Customer Risk Clusters (K-Means, k=4)")
    cp = models["cluster_profiles"]
    labels = models["cluster_labels"]

    # Display cluster profiles table
    st.dataframe(cp.reset_index().rename(columns={"index":"Risk Segment"}),
                 use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        # Pie chart of cluster sizes
        sizes = cp["Count"].values
        names = cp.index.tolist()
        fig_pie = px.pie(names=names, values=sizes,
                         color_discrete_sequence=["#10b981","#f59e0b","#f97316","#ef4444"],
                         template="plotly_dark", hole=0.4,
                         title="Cluster Size Distribution")
        fig_pie.update_layout(paper_bgcolor="#1e2130",height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        # Claim rate by cluster
        fig_bar = px.bar(x=cp.index.tolist(), y=(cp["Claim_Rate"].values),
                         color=cp.index.tolist(),
                         color_discrete_sequence=["#10b981","#f59e0b","#f97316","#ef4444"],
                         template="plotly_dark",
                         title="Claim Rate by Risk Cluster",
                         labels={"x":"Segment","y":"Claim Rate %"})
        fig_bar.update_layout(paper_bgcolor="#1e2130",plot_bgcolor="#1e2130",
                               height=320,showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Radar
    st.markdown("#### 🕸️ Cluster Profile Radar Chart")
    categories = ["Avg Credit Score (×100)","Avg Speeding (×10)","Avg Accidents (×10)","Avg DUIs (×20)","Claim Rate %"]
    fig_r = go.Figure()
    colors_r = ["#10b981","#f59e0b","#f97316","#ef4444"]
    fills_r  = ["rgba(16,185,129,0.13)","rgba(245,158,11,0.13)","rgba(249,115,22,0.13)","rgba(239,68,68,0.13)"]
    for i, (idx, row) in enumerate(cp.iterrows()):
        vals = [row["Avg_Credit"]*100, row["Avg_Speeding"]*10, row["Avg_Accidents"]*10,
                row["Avg_DUIs"]*20,   row["Claim_Rate"]]
        fig_r.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=categories+[categories[0]],
            mode="lines+markers", name=str(idx),
            line=dict(color=colors_r[i%4], width=2),
            fill="toself", fillcolor=fills_r[i%4]
        ))
    fig_r.update_layout(polar=dict(bgcolor="#1e2130"),paper_bgcolor="#1e2130",
                         font_color="#e2e8f0",height=420,
                         legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_r, use_container_width=True)

    # Cluster lookup
    st.markdown("#### 🔍 Which Cluster Am I In?")
    ck1, ck2, ck3 = st.columns(3)
    ck_age   = ck1.slider("Age", 17, 80, 35, key="ck_age")
    ck_cred  = ck1.slider("Credit Score", 0.0, 1.0, 0.6, key="ck_cr")
    ck_spd   = ck2.slider("Speeding Violations", 0, 15, 0, key="ck_spd")
    ck_acc   = ck2.slider("Past Accidents", 0, 10, 0, key="ck_acc")
    ck_duis  = ck3.slider("DUIs", 0, 5, 0, key="ck_duis")
    ck_inc   = ck3.select_slider("Income", ["poverty","working class","middle class","upper class"], value="middle class", key="ck_inc")

    if st.button("Find My Cluster", type="primary"):
        imap = {"poverty":0,"working class":1,"middle class":2,"upper class":3}
        row_k = pd.DataFrame([{
            "age_numeric": float(ck_age), "credit_score": float(ck_cred),
            "speeding_violations": float(ck_spd), "past_accidents": float(ck_acc),
            "duis": float(ck_duis), "income_score": float(imap[ck_inc])
        }])[models["km_features"]].fillna(0)
        kx = models["scaler_km"].transform(row_k)
        cid = int(models["kmeans"].predict(kx)[0])
        seg = models["cluster_labels"].get(cid, f"Cluster {cid}")
        seg_color = {"Low Risk 🟢":"#10b981","Medium Risk 🟡":"#f59e0b",
                     "High Risk 🟠":"#f97316","Very High Risk 🔴":"#ef4444"}
        c = seg_color.get(seg, "#6366f1")
        st.markdown(f"""
        <div style="background:{c}22;border:2px solid {c};border-radius:14px;
                    padding:20px;text-align:center;margin-top:12px;">
            <div style="color:{c};font-size:36px;font-weight:900;">{seg}</div>
            <div style="color:#9ca3af;margin-top:8px;">Cluster ID: {cid}</div>
        </div>""", unsafe_allow_html=True)

# ── CLAIM AMOUNT ──────────────────────────────────────────────────────────────
with tab[3]:
    st.markdown("### 💸 Claim Amount Prediction (Regression Model)")
    st.caption("Predicts expected claim payout using Random Forest Regressor trained on motor data")

    ra1, ra2 = st.columns(2)
    with ra1:
        r_insval  = st.number_input("Insured Value", 0, 10_000_000, 500_000, 10_000)
        r_premium = st.number_input("Premium Amount", 0, 500_000, 7_000, 500)
    with ra2:
        r_age     = st.slider("Vehicle Age (years)", 0, 50, 10)
        r_months  = st.slider("Policy Duration (months)", 1, 36, 12)

    if st.button("💰 Predict Claim Amount", type="primary"):
        reg = models["regressor"]
        rfeat = models["reg_features"]
        row_r = pd.DataFrame([{
            "insured_value": float(r_insval),
            "premium":       float(r_premium),
            "vehicle_age":   float(r_age),
            "policy_months": float(r_months),
        }])[rfeat].fillna(0)
        log_pred = reg.predict(row_r)[0]
        amount   = float(np.expm1(log_pred))
        severity = ("Low" if amount<50000 else "Medium" if amount<200000 else "High" if amount<500000 else "Very High")
        sev_col  = {"Low":"#10b981","Medium":"#f59e0b","High":"#f97316","Very High":"#ef4444"}[severity]

        st.markdown(f"""
        <div style="background:{sev_col}22;border:2px solid {sev_col};
                    border-radius:16px;padding:28px;text-align:center;margin:16px 0;">
            <div style="color:#9ca3af;font-size:13px;font-weight:600;text-transform:uppercase;">
                Predicted Claim Amount
            </div>
            <div style="color:{sev_col};font-size:56px;font-weight:900;margin:8px 0;">
                {amount:,.0f}
            </div>
            <div style="color:{sev_col};font-weight:700;">Severity: {severity}</div>
            <div style="color:#6b7280;font-size:13px;margin-top:8px;">
                Predicted using log-transformed Random Forest Regressor
            </div>
        </div>""", unsafe_allow_html=True)

        # Comparison bar
        avg_claim = 243283
        fig_cmp = go.Figure(go.Bar(
            x=["Your Predicted Claim","Portfolio Average"],
            y=[amount, avg_claim],
            marker_color=[sev_col,"#6366f1"],
            text=[f"{amount:,.0f}", f"{avg_claim:,.0f}"],
            textposition="outside",
        ))
        fig_cmp.update_layout(template="plotly_dark",paper_bgcolor="#1e2130",
                               plot_bgcolor="#1e2130",height=300,showlegend=False)
        st.plotly_chart(fig_cmp, use_container_width=True)
