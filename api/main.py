"""
FastAPI Backend — Vehicle Insurance Risk Analytics Platform
Serves ML predictions via REST API.
Run: python -m uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import sqlite3
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent
SAVE_DIR  = BASE_DIR / "src" / "models" / "saved"
DB_PATH   = BASE_DIR / "data" / "insurance.db"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="InsureAI API",
    description="Vehicle Insurance Risk Analytics & Claim Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Models ───────────────────────────────────────────────────────────────
def load_models():
    models = {}
    try:
        models["classifier"]       = joblib.load(SAVE_DIR / "classifier.pkl")
        models["scaler_clf"]       = joblib.load(SAVE_DIR / "scaler_clf.pkl")
        models["clf_features"]     = joblib.load(SAVE_DIR / "clf_features.pkl")
        models["regressor"]        = joblib.load(SAVE_DIR / "regressor.pkl")
        models["reg_features"]     = joblib.load(SAVE_DIR / "reg_features.pkl")
        models["kmeans"]           = joblib.load(SAVE_DIR / "kmeans.pkl")
        models["scaler_km"]        = joblib.load(SAVE_DIR / "scaler_km.pkl")
        models["km_features"]      = joblib.load(SAVE_DIR / "km_features.pkl")
        models["cluster_labels"]   = joblib.load(SAVE_DIR / "cluster_labels.pkl")
        models["clf_results"]      = joblib.load(SAVE_DIR / "clf_results.pkl")
        print("✅ All ML models loaded successfully")
    except FileNotFoundError as e:
        print(f"⚠️  Model file not found: {e}")
        print("   Run: python src/models/train_models.py  first!")
    return models

_models = {}

@app.on_event("startup")
async def startup():
    global _models
    _models = load_models()

# ── Schemas ───────────────────────────────────────────────────────────────────
class ClaimPredictRequest(BaseModel):
    age_numeric: float = 32.0
    exp_numeric: float = 10.0
    income_score: float = 2.0
    credit_score: float = 0.6
    annual_mileage: float = 12000.0
    speeding_violations: float = 0.0
    duis: float = 0.0
    past_accidents: float = 0.0
    risk_score: Optional[float] = None

class ClaimAmountRequest(BaseModel):
    insured_value: float = 500000.0
    premium: float = 5000.0
    vehicle_age: float = 10.0
    policy_months: float = 12.0

class ClusterRequest(BaseModel):
    age_numeric: float = 32.0
    credit_score: float = 0.6
    speeding_violations: float = 0.0
    past_accidents: float = 0.0
    duis: float = 0.0
    income_score: float = 2.0

class NLQueryRequest(BaseModel):
    question: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": len(_models) > 0,
        "model_files": list(SAVE_DIR.glob("*.pkl")) if SAVE_DIR.exists() else []
    }

@app.get("/")
def root():
    return {
        "name":    "InsureAI API",
        "version": "1.0.0",
        "docs":    "/docs",
        "endpoints": ["/health", "/predict/claim", "/predict/amount",
                      "/predict/cluster", "/recommend/premium",
                      "/stats/overview", "/query/nl"]
    }

@app.post("/predict/claim")
def predict_claim(req: ClaimPredictRequest):
    if not _models:
        raise HTTPException(503, "Models not loaded. Run train_models.py first.")
    clf      = _models["classifier"]
    scaler   = _models["scaler_clf"]
    features = _models["clf_features"]

    # Auto-compute risk_score if not given
    risk = req.risk_score if req.risk_score is not None else (
        req.past_accidents * 15 + req.speeding_violations * 5 +
        req.duis * 20 + (1 - req.credit_score) * 20
    )

    row = pd.DataFrame([{
        "age_numeric":          req.age_numeric,
        "exp_numeric":          req.exp_numeric,
        "income_score":         req.income_score,
        "credit_score":         req.credit_score,
        "annual_mileage":       req.annual_mileage,
        "speeding_violations":  req.speeding_violations,
        "duis":                 req.duis,
        "past_accidents":       req.past_accidents,
        "risk_score":           risk
    }])[features].fillna(0)

    from sklearn.linear_model import LogisticRegression
    if isinstance(clf, LogisticRegression):
        prob = clf.predict_proba(scaler.transform(row))[0][1]
    else:
        prob = clf.predict_proba(row)[0][1]

    risk_label = ("Low"       if prob < 0.25 else
                  "Medium"    if prob < 0.50 else
                  "High"      if prob < 0.75 else
                  "Very High")

    return {
        "claim_probability":  round(float(prob), 4),
        "claim_probability_pct": f"{prob*100:.2f}%",
        "prediction":         "Claim Likely" if prob >= 0.5 else "No Claim",
        "risk_level":         risk_label,
        "confidence":         f"{max(prob, 1-prob)*100:.1f}%",
    }

@app.post("/predict/amount")
def predict_amount(req: ClaimAmountRequest):
    if not _models:
        raise HTTPException(503, "Models not loaded.")
    reg      = _models["regressor"]
    features = _models["reg_features"]

    row = pd.DataFrame([{
        "insured_value": req.insured_value,
        "premium":       req.premium,
        "vehicle_age":   req.vehicle_age,
        "policy_months": req.policy_months
    }])[features].fillna(0)

    log_pred = reg.predict(row)[0]
    amount   = float(np.expm1(log_pred))

    return {
        "predicted_claim_amount": round(amount, 2),
        "formatted":              f"{amount:,.2f}",
        "severity_bucket":        ("Low"    if amount < 10000 else
                                   "Medium" if amount < 50000 else
                                   "High"   if amount < 200000 else "Very High"),
    }

@app.post("/predict/cluster")
def predict_cluster(req: ClusterRequest):
    if not _models:
        raise HTTPException(503, "Models not loaded.")
    km       = _models["kmeans"]
    scaler   = _models["scaler_km"]
    features = _models["km_features"]
    labels   = _models["cluster_labels"]

    row = pd.DataFrame([{
        "age_numeric":         req.age_numeric,
        "credit_score":        req.credit_score,
        "speeding_violations": req.speeding_violations,
        "past_accidents":      req.past_accidents,
        "duis":                req.duis,
        "income_score":        req.income_score,
    }])[features].fillna(0)

    cluster_id = int(km.predict(scaler.transform(row))[0])
    return {
        "cluster_id":    cluster_id,
        "risk_segment":  labels.get(cluster_id, f"Cluster {cluster_id}"),
    }

@app.get("/recommend/premium")
def recommend_premium(
    vehicle_type: str = "Automobile",
    vehicle_age: float = 10.0,
    insured_value: float = 500000.0,
    past_accidents: float = 0.0,
    speeding_violations: float = 0.0,
    duis: float = 0.0,
    credit_score: float = 0.6,
    policy_months: float = 12.0
):
    # Rule-based risk-adjusted pricing
    base_rate = 2000.0
    risk_score = (
        past_accidents * 15 + speeding_violations * 5 + duis * 20 +
        (1 - credit_score) * 20 + max(0, vehicle_age - 10) * 2
    )
    risk_factor = 1 + (risk_score / 100) * 1.5
    value_load  = insured_value * 0.008 * (policy_months / 12)
    raw_premium = (base_rate * risk_factor + value_load) * (policy_months / 12)
    final       = raw_premium * 1.27   # 15% profit + 12% expense

    return {
        "base_rate":           base_rate,
        "risk_score":          round(risk_score, 2),
        "risk_factor":         round(risk_factor, 3),
        "pure_premium":        round(raw_premium, 2),
        "recommended_premium": round(final, 2),
        "policy_months":       policy_months,
    }

@app.get("/stats/overview")
def stats_overview():
    try:
        conn  = sqlite3.connect(DB_PATH)
        motor = pd.read_sql("SELECT * FROM fact_motor_claims LIMIT 100000", conn)
        car   = pd.read_sql("SELECT * FROM fact_car_claims", conn)
        conn.close()
        return {
            "total_motor_policies": len(motor),
            "total_car_policies":   len(car),
            "motor_claim_rate":     f"{motor['claim_flag'].mean()*100:.2f}%",
            "car_claim_rate":       f"{car['outcome'].mean()*100:.2f}%",
            "avg_premium":          round(float(motor["premium"].mean()), 2),
            "avg_claim_paid":       round(float(motor[motor["claim_paid"]>0]["claim_paid"].mean()), 2),
            "loss_ratio":           f"{motor['claim_paid'].sum()/motor['premium'].sum()*100:.2f}%",
        }
    except Exception as e:
        return {"error": str(e), "note": "Run ETL pipeline first: python src/etl_pipeline.py"}

@app.post("/query/nl")
def nl_query(req: NLQueryRequest):
    """Simple keyword-based NL answer (no LLM required)"""
    q = req.question.lower()
    try:
        conn  = sqlite3.connect(DB_PATH)
        motor = pd.read_sql("SELECT * FROM fact_motor_claims LIMIT 50000", conn)
        car   = pd.read_sql("SELECT * FROM fact_car_claims", conn)
        conn.close()
    except Exception:
        return {"answer": "Database not ready. Run ETL pipeline first.", "data": None}

    if "highest claim" in q or "most claims" in q:
        top = motor.groupby("type_vehicle")["claim_flag"].mean().idxmax()
        rate = motor.groupby("type_vehicle")["claim_flag"].mean().max()
        return {"answer": f"{top} has the highest claim rate at {rate*100:.1f}%", "vehicle_type": top}
    elif "average premium" in q or "avg premium" in q:
        return {"answer": f"The average premium is {motor['premium'].mean():,.0f}", "value": float(motor['premium'].mean())}
    elif "loss ratio" in q:
        lr = motor["claim_paid"].sum() / motor["premium"].sum() * 100
        return {"answer": f"Portfolio loss ratio is {lr:.1f}%", "loss_ratio": round(lr, 2)}
    elif "claim rate" in q:
        return {"answer": f"Overall claim rate is {motor['claim_flag'].mean()*100:.1f}%",
                "motor_claim_rate": round(float(motor['claim_flag'].mean()*100), 2),
                "car_claim_rate":   round(float(car['outcome'].mean()*100), 2)}
    else:
        return {"answer": "Ask about: claim rates, average premiums, loss ratio, high-risk vehicles."}
