"""
ETL Pipeline — Vehicle Insurance Risk Analytics Platform
Extracts from both CSVs → Transforms → Loads into SQLite data warehouse (Star Schema)
Run: python src/etl_pipeline.py
"""
import pandas as pd
import numpy as np
import sqlite3
import os
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "Dataset"
DB_PATH     = BASE_DIR / "data" / "insurance.db"
DB_PATH.parent.mkdir(exist_ok=True)

CAR_CSV   = DATASET_DIR / "Car_Insurance_Claim.csv"
MOTOR_CSV = DATASET_DIR / "motor_data14-2018.csv"

print("=" * 60)
print("  InsureAI ETL Pipeline")
print("=" * 60)

# ── STEP 1: EXTRACT ────────────────────────────────────────────────────────────
print("\n[1/5] EXTRACTING data from CSVs...")

car_df = pd.read_csv(CAR_CSV)
print(f"  ✓ Car Insurance data:  {len(car_df):,} rows × {car_df.shape[1]} cols")

motor_df = pd.read_csv(MOTOR_CSV, low_memory=False)
print(f"  ✓ Motor Policy data:   {len(motor_df):,} rows × {motor_df.shape[1]} cols")

# ── STEP 2: TRANSFORM — CAR DATA ───────────────────────────────────────────────
print("\n[2/5] TRANSFORMING car insurance data...")

car_df.columns = [c.strip().upper() for c in car_df.columns]

# Missing values
null_before = car_df.isnull().sum().sum()
for col in car_df.select_dtypes(include=np.number).columns:
    car_df[col] = car_df[col].fillna(car_df[col].median())
for col in car_df.select_dtypes(include="object").columns:
    if not car_df[col].mode().empty:
        car_df[col] = car_df[col].fillna(car_df[col].mode()[0])
print(f"  ✓ Filled {null_before} null values")

# Duplicates
before = len(car_df)
car_df = car_df.drop_duplicates()
print(f"  ✓ Removed {before - len(car_df)} duplicate rows")

# Type conversion
car_df["OUTCOME"]           = car_df["OUTCOME"].astype(int)
car_df["CREDIT_SCORE"]      = pd.to_numeric(car_df["CREDIT_SCORE"], errors="coerce").fillna(0.5)
car_df["ANNUAL_MILEAGE"]    = pd.to_numeric(car_df["ANNUAL_MILEAGE"], errors="coerce").fillna(12000)
car_df["SPEEDING_VIOLATIONS"] = pd.to_numeric(car_df["SPEEDING_VIOLATIONS"], errors="coerce").fillna(0)
car_df["DUIS"]              = pd.to_numeric(car_df["DUIS"], errors="coerce").fillna(0)
car_df["PAST_ACCIDENTS"]    = pd.to_numeric(car_df["PAST_ACCIDENTS"], errors="coerce").fillna(0)

# Feature engineering
age_map = {"16-25": 20, "26-39": 32, "40-64": 52, "65+": 70}
car_df["AGE_NUMERIC"]   = car_df["AGE"].map(age_map).fillna(35)
exp_map = {"0-9y": 4, "10-19y": 14, "20-29y": 24, "30y+": 35}
car_df["EXP_NUMERIC"]   = car_df["DRIVING_EXPERIENCE"].map(exp_map).fillna(10) if "DRIVING_EXPERIENCE" in car_df.columns else 10
inc_map = {"poverty": 0, "working class": 1, "middle class": 2, "upper class": 3}
car_df["INCOME_SCORE"]  = car_df["INCOME"].map(inc_map).fillna(1)
edu_map = {"none": 0, "high school": 1, "university": 2}
car_df["EDU_SCORE"]     = car_df["EDUCATION"].map(edu_map).fillna(1) if "EDUCATION" in car_df.columns else 1

# Risk score
car_df["RISK_SCORE"] = (
    car_df["PAST_ACCIDENTS"] * 15
    + car_df["SPEEDING_VIOLATIONS"] * 5
    + car_df["DUIS"] * 20
    + (1 - car_df["CREDIT_SCORE"]) * 20
    + car_df["AGE_NUMERIC"].apply(lambda a: 20 if a < 25 else 5 if a > 65 else 0)
    + car_df["INCOME_SCORE"].apply(lambda i: 10 if i == 0 else 0)
).clip(0, 100)

# Outlier detection (IQR method on ANNUAL_MILEAGE)
Q1 = car_df["ANNUAL_MILEAGE"].quantile(0.25)
Q3 = car_df["ANNUAL_MILEAGE"].quantile(0.75)
IQR = Q3 - Q1
outliers = ((car_df["ANNUAL_MILEAGE"] < Q1 - 1.5*IQR) | (car_df["ANNUAL_MILEAGE"] > Q3 + 1.5*IQR)).sum()
car_df["ANNUAL_MILEAGE"] = car_df["ANNUAL_MILEAGE"].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
print(f"  ✓ Capped {outliers} outliers in ANNUAL_MILEAGE")

print(f"  ✓ Engineered features: AGE_NUMERIC, EXP_NUMERIC, INCOME_SCORE, EDU_SCORE, RISK_SCORE")
print(f"  ✓ Car data ready: {len(car_df):,} rows")

# ── STEP 3: TRANSFORM — MOTOR DATA ────────────────────────────────────────────
print("\n[3/5] TRANSFORMING motor policy data...")

motor_df.columns = [c.strip().upper() for c in motor_df.columns]

# Date parsing
for dc in ["INSR_BEGIN", "INSR_END"]:
    if dc in motor_df.columns:
        motor_df[dc] = pd.to_datetime(motor_df[dc], format="%d-%b-%y", errors="coerce")

# Numeric coerce
for col in ["INSURED_VALUE", "PREMIUM", "CLAIM_PAID"]:
    if col in motor_df.columns:
        motor_df[col] = pd.to_numeric(motor_df[col], errors="coerce").fillna(0)

# Fill nulls
null_m = motor_df.isnull().sum().sum()
for col in motor_df.select_dtypes(include=np.number).columns:
    motor_df[col] = motor_df[col].fillna(motor_df[col].median())
for col in motor_df.select_dtypes(include="object").columns:
    if not motor_df[col].mode().empty:
        motor_df[col] = motor_df[col].fillna(motor_df[col].mode()[0])
print(f"  ✓ Filled {null_m} null values")

# Remove duplicates
before_m = len(motor_df)
motor_df = motor_df.drop_duplicates()
print(f"  ✓ Removed {before_m - len(motor_df)} duplicates")

# Derived features
motor_df["VEHICLE_AGE"]   = (2024 - pd.to_numeric(motor_df["PROD_YEAR"], errors="coerce").fillna(2010)).clip(0, 80)
motor_df["CLAIM_FLAG"]    = (motor_df["CLAIM_PAID"] > 0).astype(int)
motor_df["POLICY_MONTHS"] = ((motor_df["INSR_END"] - motor_df["INSR_BEGIN"]).dt.days / 30).clip(0, 36).fillna(12)
motor_df["YEAR"]          = motor_df["INSR_BEGIN"].dt.year.fillna(2017).astype(int)
motor_df["MONTH"]         = motor_df["INSR_BEGIN"].dt.month.fillna(1).astype(int)
motor_df["QUARTER"]       = motor_df["INSR_BEGIN"].dt.quarter.fillna(1).astype(int)

# Loss ratio per vehicle
loss = motor_df.groupby("TYPE_VEHICLE").apply(
    lambda g: (g["CLAIM_PAID"].sum() / g["PREMIUM"].sum() * 100) if g["PREMIUM"].sum() > 0 else 0
).reset_index()
loss.columns = ["TYPE_VEHICLE", "VEHICLE_LOSS_RATIO"]
motor_df = motor_df.merge(loss, on="TYPE_VEHICLE", how="left")

# Premium outlier clip
p99 = motor_df["PREMIUM"].quantile(0.99)
motor_df["PREMIUM"] = motor_df["PREMIUM"].clip(0, p99)

print(f"  ✓ Motor data ready: {len(motor_df):,} rows")

# ── STEP 4: LOAD TO SQLITE (STAR SCHEMA) ──────────────────────────────────────
print("\n[4/5] LOADING to SQLite data warehouse...")

conn = sqlite3.connect(DB_PATH)

# ── Dimension Tables ──────────────────────────────────────────────────────────
# dim_vehicle
dim_vehicle = motor_df[["TYPE_VEHICLE","VEHICLE_AGE","CCM_TON","SEATS_NUM","MAKE"]].drop_duplicates()
dim_vehicle = dim_vehicle.reset_index(drop=True)
dim_vehicle.index.name = "vehicle_id"
dim_vehicle.columns    = ["vehicle_type","vehicle_age","engine_cc","seats","make"]
dim_vehicle.to_sql("dim_vehicle", conn, if_exists="replace", index=True)
print(f"  ✓ dim_vehicle:   {len(dim_vehicle):,} rows")

# dim_customer (from car data)
dim_customer = car_df[["ID","AGE","GENDER","EDUCATION","INCOME","CREDIT_SCORE","MARRIED","CHILDREN"]].copy() if "GENDER" in car_df.columns else car_df[["ID","AGE","EDUCATION","INCOME","CREDIT_SCORE"]].copy()
dim_customer.columns = [c.lower() for c in dim_customer.columns]
dim_customer.to_sql("dim_customer", conn, if_exists="replace", index=False)
print(f"  ✓ dim_customer:  {len(dim_customer):,} rows")

# dim_time
years   = sorted(motor_df["YEAR"].dropna().unique().tolist())
months  = list(range(1, 13))
quarters = [1,1,1,2,2,2,3,3,3,4,4,4]
dim_time = pd.DataFrame([
    {"year": y, "month": m, "quarter": quarters[m-1], "year_month": f"{y}-{m:02d}"}
    for y in years for m in months
])
dim_time.to_sql("dim_time", conn, if_exists="replace", index=True)
print(f"  ✓ dim_time:      {len(dim_time):,} rows")

# dim_location
dim_location = pd.DataFrame({
    "postal_code": motor_df["EFFECTIVE_YR"].unique() if "EFFECTIVE_YR" in motor_df.columns else [0],
    "region": ["Unknown"] * motor_df["EFFECTIVE_YR"].nunique() if "EFFECTIVE_YR" in motor_df.columns else ["Unknown"]
})
dim_location.to_sql("dim_location", conn, if_exists="replace", index=True)
print(f"  ✓ dim_location:  {len(dim_location):,} rows")

# ── Fact Tables ───────────────────────────────────────────────────────────────
fact_motor = motor_df[[
    "OBJECT_ID","TYPE_VEHICLE","INSURED_VALUE","PREMIUM","CLAIM_PAID",
    "CLAIM_FLAG","VEHICLE_AGE","POLICY_MONTHS","YEAR","MONTH","QUARTER",
    "USAGE","MAKE","VEHICLE_LOSS_RATIO"
]].copy()
fact_motor.columns = [c.lower() for c in fact_motor.columns]
fact_motor.to_sql("fact_motor_claims", conn, if_exists="replace", index=True)
print(f"  ✓ fact_motor_claims: {len(fact_motor):,} rows")

fact_car = car_df[[
    "ID","AGE","VEHICLE_TYPE","ANNUAL_MILEAGE","CREDIT_SCORE",
    "SPEEDING_VIOLATIONS","DUIS","PAST_ACCIDENTS","OUTCOME",
    "RISK_SCORE","AGE_NUMERIC","EXP_NUMERIC","INCOME_SCORE"
]].copy()
fact_car.columns = [c.lower() for c in fact_car.columns]
fact_car.to_sql("fact_car_claims", conn, if_exists="replace", index=True)
print(f"  ✓ fact_car_claims: {len(fact_car):,} rows")

conn.close()

# ── STEP 5: DATA QUALITY REPORT ───────────────────────────────────────────────
print("\n[5/5] DATA QUALITY REPORT")
print(f"  Motor Dataset:")
print(f"    • Total policies:  {len(motor_df):,}")
print(f"    • Claim rate:      {motor_df['CLAIM_FLAG'].mean()*100:.2f}%")
print(f"    • Avg premium:     {motor_df['PREMIUM'].mean():,.2f}")
print(f"    • Avg claim paid:  {motor_df[motor_df['CLAIM_PAID']>0]['CLAIM_PAID'].mean():,.2f}")
print(f"    • Loss ratio:      {motor_df['CLAIM_PAID'].sum()/motor_df['PREMIUM'].sum()*100:.2f}%")
print(f"\n  Car Insurance Dataset:")
print(f"    • Total records:   {len(car_df):,}")
print(f"    • Claim rate:      {car_df['OUTCOME'].mean()*100:.2f}%")
print(f"    • Avg risk score:  {car_df['RISK_SCORE'].mean():.2f}")

print(f"\n{'='*60}")
print(f"  ✅ ETL COMPLETE! Database: {DB_PATH}")
print(f"{'='*60}\n")
