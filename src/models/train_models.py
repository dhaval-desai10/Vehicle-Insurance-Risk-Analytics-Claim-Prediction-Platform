"""
ML Model Training — Vehicle Insurance Risk Analytics Platform
Trains Classification, Regression, and Clustering models → saves to src/models/saved/
Run: python src/models/train_models.py
"""
import pandas as pd
import numpy as np
import sqlite3
import joblib
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, mean_absolute_error, r2_score)
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent.parent.parent
DB_PATH    = BASE_DIR / "data" / "insurance.db"
SAVE_DIR   = Path(__file__).parent / "saved"
SAVE_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  InsureAI ML Training Pipeline")
print("=" * 60)

# ── Load from SQLite ──────────────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
car   = pd.read_sql("SELECT * FROM fact_car_claims",   conn)
motor = pd.read_sql("SELECT * FROM fact_motor_claims", conn)
conn.close()
print(f"\n  Loaded car data:   {len(car):,} rows")
print(f"  Loaded motor data: {len(motor):,} rows")

# ══════════════════════════════════════════════════════════════════
# MODEL 1 — CLASSIFICATION (Predict Claim: Yes/No)
# ══════════════════════════════════════════════════════════════════
print("\n[1/3] TRAINING CLASSIFICATION MODEL...")

FEAT_CLF = ["age_numeric","exp_numeric","income_score","credit_score",
            "annual_mileage","speeding_violations","duis","past_accidents","risk_score"]
FEAT_CLF = [f for f in FEAT_CLF if f in car.columns]

X = car[FEAT_CLF].fillna(0)
y = car["outcome"].astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler_clf = StandardScaler()
X_tr_s = scaler_clf.fit_transform(X_tr)
X_te_s  = scaler_clf.transform(X_te)

# 3 classifiers
clf_models = {
    "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
}

best_auc   = 0
best_clf   = None
best_name  = ""
clf_results = {}

for name, mdl in clf_models.items():
    if name == "Logistic Regression":
        mdl.fit(X_tr_s, y_tr)
        prob = mdl.predict_proba(X_te_s)[:, 1]
    else:
        mdl.fit(X_tr, y_tr)
        prob = mdl.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc  = roc_auc_score(y_te, prob)
    acc  = (pred == y_te).mean()
    clf_results[name] = {"auc": auc, "acc": acc}
    print(f"  {name:25s} → AUC={auc:.4f}  Acc={acc*100:.2f}%")
    if auc > best_auc:
        best_auc  = auc
        best_clf  = mdl
        best_name = name

print(f"\n  🏆 Best classifier: {best_name} (AUC={best_auc:.4f})")

# Save best classifier
joblib.dump(best_clf,   SAVE_DIR / "classifier.pkl")
joblib.dump(scaler_clf, SAVE_DIR / "scaler_clf.pkl")
joblib.dump(FEAT_CLF,   SAVE_DIR / "clf_features.pkl")
print(f"  ✓ Saved: classifier.pkl, scaler_clf.pkl, clf_features.pkl")

# Save all classifiers
for name, mdl in clf_models.items():
    safe_name = name.lower().replace(" ","_")
    joblib.dump(mdl, SAVE_DIR / f"clf_{safe_name}.pkl")
joblib.dump(clf_results, SAVE_DIR / "clf_results.pkl")

# ══════════════════════════════════════════════════════════════════
# MODEL 2 — REGRESSION (Predict Claim Amount)
# ══════════════════════════════════════════════════════════════════
print("\n[2/3] TRAINING REGRESSION MODEL (Claim Amount)...")

# Use motor data — only rows with claims
motor_claims = motor[motor["claim_paid"] > 0].copy()

FEAT_REG = ["insured_value","premium","vehicle_age","policy_months"]
FEAT_REG = [f for f in FEAT_REG if f in motor_claims.columns]

# Clip extreme outliers
p99 = motor_claims["claim_paid"].quantile(0.99)
motor_claims = motor_claims[motor_claims["claim_paid"] <= p99]

Xr = motor_claims[FEAT_REG].fillna(0)
yr = np.log1p(motor_claims["claim_paid"])   # log-transform for normality

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
Xr_tr_s = scaler_reg.fit_transform(Xr_tr)
Xr_te_s  = scaler_reg.transform(Xr_te)

reg_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
reg_model.fit(Xr_tr, yr_tr)
yr_pred = reg_model.predict(Xr_te)
mae = mean_absolute_error(np.expm1(yr_te), np.expm1(yr_pred))
r2  = r2_score(yr_te, yr_pred)

print(f"  Random Forest Regressor:")
print(f"    MAE  = {mae:,.2f}")
print(f"    R²   = {r2:.4f}")

joblib.dump(reg_model,  SAVE_DIR / "regressor.pkl")
joblib.dump(scaler_reg, SAVE_DIR / "scaler_reg.pkl")
joblib.dump(FEAT_REG,   SAVE_DIR / "reg_features.pkl")
print(f"  ✓ Saved: regressor.pkl, scaler_reg.pkl, reg_features.pkl")

# ══════════════════════════════════════════════════════════════════
# MODEL 3 — CLUSTERING (Customer Risk Segments)
# ══════════════════════════════════════════════════════════════════
print("\n[3/3] TRAINING CLUSTERING MODEL (K-Means k=4)...")

FEAT_KM = ["age_numeric","credit_score","speeding_violations","past_accidents","duis","income_score"]
FEAT_KM = [f for f in FEAT_KM if f in car.columns]

Xk = car[FEAT_KM].fillna(0)
scaler_km = StandardScaler()
Xk_s = scaler_km.fit_transform(Xk)

km = KMeans(n_clusters=4, random_state=42, n_init=15, max_iter=300)
km.fit(Xk_s)
car["cluster"] = km.labels_

# Label clusters by claim rate
cluster_claim_rate = car.groupby("cluster")["outcome"].mean()
rank = cluster_claim_rate.rank()
labels = {int(k): v for k, v in rank.map({
    1.0: "Low Risk 🟢", 2.0: "Medium Risk 🟡",
    3.0: "High Risk 🟠", 4.0: "Very High Risk 🔴"
}).items()}

cluster_profiles = car.groupby("cluster").agg(
    Avg_Credit=("credit_score","mean"),
    Avg_Speeding=("speeding_violations","mean"),
    Avg_Accidents=("past_accidents","mean"),
    Avg_DUIs=("duis","mean"),
    Claim_Rate=("outcome","mean"),
    Count=("outcome","count")
).round(3)
cluster_profiles.index = [labels[i] for i in cluster_profiles.index]

print(f"\n  Cluster Profiles:")
for idx, row in cluster_profiles.iterrows():
    print(f"    {idx:25s} → Claim Rate: {row['Claim_Rate']*100:.1f}%  N={row['Count']:,}")

joblib.dump(km,               SAVE_DIR / "kmeans.pkl")
joblib.dump(scaler_km,        SAVE_DIR / "scaler_km.pkl")
joblib.dump(FEAT_KM,          SAVE_DIR / "km_features.pkl")
joblib.dump(labels,           SAVE_DIR / "cluster_labels.pkl")
joblib.dump(cluster_profiles, SAVE_DIR / "cluster_profiles.pkl")
print(f"\n  ✓ Saved: kmeans.pkl, scaler_km.pkl, cluster_labels.pkl")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ✅ ALL MODELS TRAINED & SAVED to {SAVE_DIR}")
print(f"{'='*60}")
files = list(SAVE_DIR.glob("*.pkl"))
for f in sorted(files):
    print(f"    📦 {f.name}")
print()
