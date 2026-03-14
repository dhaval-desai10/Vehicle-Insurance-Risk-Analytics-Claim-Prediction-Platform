"""
Data loader utility — reads both CSVs and caches them.
"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent.parent / "Dataset"
CAR_CSV     = DATASET_DIR / "Car_Insurance_Claim.csv"
MOTOR_CSV   = DATASET_DIR / "motor_data14-2018.csv"


@st.cache_data(show_spinner=False)
def load_car_data() -> pd.DataFrame:
    df = pd.read_csv(CAR_CSV)
    df.columns = [c.strip().upper() for c in df.columns]
    # Fill numeric nulls with median
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    # Fill categorical nulls with mode
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    df["OUTCOME"] = df["OUTCOME"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_motor_data(sample: int = 50_000) -> pd.DataFrame:
    df = pd.read_csv(MOTOR_CSV, nrows=sample, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    # Parse dates
    for dc in ["INSR_BEGIN", "INSR_END"]:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], format="%d-%b-%y", errors="coerce")
    # Numeric coerce
    for col in ["INSURED_VALUE", "PREMIUM", "CLAIM_PAID"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # Derived columns
    if "PROD_YEAR" in df.columns:
        df["VEHICLE_AGE"] = 2024 - pd.to_numeric(df["PROD_YEAR"], errors="coerce").fillna(2010)
    if "CLAIM_PAID" in df.columns:
        df["CLAIM_FLAG"] = (df["CLAIM_PAID"] > 0).astype(int)
    if "INSR_BEGIN" in df.columns and "INSR_END" in df.columns:
        df["POLICY_MONTHS"] = ((df["INSR_END"] - df["INSR_BEGIN"]).dt.days / 30).clip(0, 36).fillna(12)
    # Fill remaining nulls
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")
    return df


def kpi_card(col, label: str, value: str, delta: str = "", icon: str = "📊", color: str = "#6366f1"):
    col.markdown(f"""
    <div style="background:linear-gradient(135deg,{color}22,{color}11);
                border-left:4px solid {color};
                padding:16px 20px;border-radius:12px;
                box-shadow:0 2px 8px rgba(0,0,0,.15);">
        <div style="font-size:28px;margin-bottom:4px;">{icon}</div>
        <div style="color:#9ca3af;font-size:12px;font-weight:600;letter-spacing:.5px;
                    text-transform:uppercase;">{label}</div>
        <div style="color:#f9fafb;font-size:26px;font-weight:800;margin:4px 0;">{value}</div>
        <div style="color:{color};font-size:12px;">{delta}</div>
    </div>
    """, unsafe_allow_html=True)
