# 🚗 Vehicle Insurance Risk Analytics & Claim Prediction Platform
## Complete Project Flow Diagram & Build Order

---

## 📊 Two Datasets Available

| Dataset | Rows | Key Columns |
|---|---|---|
| [Car_Insurance_Claim.csv](file:///d:/kenexai/Dataset/Car_Insurance_Claim.csv) | 10,001 | ID, AGE, GENDER, CREDIT_SCORE, VEHICLE_TYPE, OUTCOME (0/1) |
| [motor_data14-2018.csv](file:///d:/kenexai/Dataset/motor_data14-2018.csv) | 508,502 | OBJECT_ID, INSURED_VALUE, PREMIUM, PROD_YEAR, TYPE_VEHICLE, CLAIM_PAID |

> **Strategy**: Use [motor_data14-2018.csv](file:///d:/kenexai/Dataset/motor_data14-2018.csv) as the **primary dataset** (has real PREMIUM & CLAIM_PAID amounts). Use [Car_Insurance_Claim.csv](file:///d:/kenexai/Dataset/Car_Insurance_Claim.csv) for **classification** (OUTCOME label ready).

---

## 🗺️ Full Project Architecture Flow

```mermaid
flowchart TD
    subgraph DATA["🗄️ PHASE 1 — DATA LAYER"]
        A1["📁 Car_Insurance_Claim.csv\n10K rows · OUTCOME label"]
        A2["📁 motor_data14-2018.csv\n508K rows · PREMIUM & CLAIM_PAID"]
        A3["🤖 Synthetic Data Simulator\n(Faker — streaming new policies)"]
    end

    subgraph ETL["⚙️ PHASE 2 — ETL PIPELINE (Python/Pandas)"]
        B1["Extract\nLoad CSVs + Simulated data"]
        B2["Transform\n• Fill nulls\n• Remove duplicates\n• Type conversion\n• IQR Outlier detection"]
        B3["Feature Engineering\n• Vehicle_Age = 2024 - PROD_YEAR\n• Claim_Flag = CLAIM_PAID > 0\n• Risk_Score formula\n• Policy_Duration months"]
        B4["Data Quality Checks\n• Schema validation\n• Range checks\n• Completeness report"]
        B5["Load → SQLite\ninsurance.db"]
    end

    subgraph DWH["🏛️ PHASE 3 — DATA WAREHOUSE (Star Schema)"]
        C1["📋 fact_claims\nPolicy_ID · Claim_Amount\nPremium · Risk_Score · Claim_Flag"]
        C2["🚗 dim_vehicle\nType · Age · Engine_CC\nMake · Seats"]
        C3["👤 dim_customer\nSex · Income · Education\nCredit_Score"]
        C4["📅 dim_time\nYear · Month · Quarter"]
        C5["📍 dim_location\nPostal_Code · Region"]
        C1 --> C2 & C3 & C4 & C5
    end

    subgraph ML["🤖 PHASE 4 — MACHINE LEARNING"]
        D1["Classification Model\nLogistic Reg + Random Forest\n+ Gradient Boosting\n→ Predict Claim Yes/No"]
        D2["Regression Model\nRandom Forest Regressor\n→ Predict Claim Amount"]
        D3["Clustering Model\nK-Means k=4\n→ Risk Segments\nLow/Med/High/Critical"]
        D4["📦 Saved Models\n.pkl files\nClassifier + Regressor + Scaler"]
        D1 & D2 & D3 --> D4
    end

    subgraph RAG["🧠 PHASE 5 — GenAI / RAG"]
        E1["Text Chunks from\nAggregated Insurance Stats"]
        E2["sentence-transformers\nEmbeddings"]
        E3["ChromaDB\nVector Store"]
        E4["NL Query Interface\n'Which vehicle has highest claims?'\n'Avg premium for trucks?'"]
        E1 --> E2 --> E3 --> E4
    end

    subgraph OPT["💰 PHASE 6 — PREMIUM OPTIMIZER"]
        F1["Input: Vehicle + Customer Profile"]
        F2["Get Claim Probability (Classifier)"]
        F3["Get Expected Claim Amount (Regressor)"]
        F4["Formula:\nPremium = Base × (1 + P×Severity)"]
        F5["Recommended Premium Output"]
        F1 --> F2 & F3 --> F4 --> F5
    end

    subgraph API["🔌 PHASE 7 — FastAPI BACKEND"]
        G1["POST /predict/claim\n→ Binary Classification"]
        G2["POST /predict/amount\n→ Claim Amount"]
        G3["POST /predict/cluster\n→ Risk Segment"]
        G4["GET /recommend/premium\n→ Optimal Price"]
        G5["POST /query/nl\n→ RAG Answer"]
        G6["GET /health"]
    end

    subgraph UI["🖥️ PHASE 8 — STREAMLIT DASHBOARD"]
        H1["📊 EDA Dashboard\nMissing data · Distributions\nCorrelation matrix"]
        H2["👔 Insurance Manager\nTotal Policies · Claim Rate\nLoss Ratio · Revenue Trends"]
        H3["⚠️ Risk Analyst\nHigh-Risk Vehicles · Heatmaps\nClaim Probability Matrix"]
        H4["🤝 Customer Advisor\nRisk Score Card\nPremium Suggestions"]
        H5["🤖 ML Prediction\nInteractive form → Live prediction\nProbability gauge chart"]
        H6["💬 GenAI Chat\nNatural language Q&A\nRAG-powered answers"]
        H7["💰 Premium Optimizer\nSliders → Recommended price"]
    end

    subgraph DEPLOY["🐳 PHASE 9 — DOCKER DEPLOYMENT"]
        I1["Dockerfile.api\n(FastAPI)"]
        I2["Dockerfile.app\n(Streamlit)"]
        I3["docker-compose.yml\napi:8000 + app:8501\nShared SQLite volume"]
    end

    DATA --> ETL --> DWH
    DWH --> ML
    DWH --> RAG
    ML --> OPT
    ML --> API
    RAG --> API
    OPT --> API
    API --> UI
    API --> DEPLOY
    UI --> DEPLOY
```

---

## 🏗️ Step-by-Step Build Order (What to Build First)

### ✅ STAGE 1 — Data Foundation (Days 1-2)
```
1. requirements.txt          ← Install all packages
2. src/etl_pipeline.py       ← Clean + transform both CSVs
3. src/warehouse_schema.sql  ← Create SQLite star schema
4. data/insurance.db         ← Populated from ETL
```
> ✔ Output: Clean database ready for everything else

---

### ✅ STAGE 2 — Machine Learning (Days 3-4)
```
5. src/models/train_models.py
   ├── Classification → Random Forest (predict OUTCOME)
   ├── Regression    → RF Regressor (predict CLAIM_PAID)  
   └── Clustering    → K-Means k=4 (risk segments)
6. src/models/saved/
   ├── classifier.pkl
   ├── regressor.pkl
   └── kmeans.pkl + scaler.pkl
```
> ✔ Output: Trained models saved to disk — **test these first with Python scripts before any web**

---

### ✅ STAGE 3 — Backend API (Day 5)
```
7. api/main.py (FastAPI)
   ├── /health
   ├── /predict/claim
   ├── /predict/amount
   ├── /predict/cluster
   ├── /recommend/premium
   └── /query/nl
```
> ✔ Output: Test with `curl` or Swagger UI at localhost:8000/docs

---

### ✅ STAGE 4 — GenAI / RAG (Day 6)
```
8. src/rag/rag_engine.py
   ├── Build ChromaDB vector store
   ├── Embed insurance stats as documents
   └── Query function for NL answers
```

---

### ✅ STAGE 5 — Web Dashboard (Days 7-9)
```
9. app/streamlit_app.py (Main entry)
   app/pages/
   ├── 1_EDA_Dashboard.py
   ├── 2_Manager_Dashboard.py
   ├── 3_Risk_Analyst.py
   ├── 4_Customer_Advisor.py
   ├── 5_ML_Prediction.py
   ├── 6_GenAI_Chat.py
   └── 7_Premium_Optimizer.py
```
> ✔ All pages call FastAPI for ML predictions

---

### ✅ STAGE 6 — Docker (Day 10)
```
10. Dockerfile.api
11. Dockerfile.app
12. docker-compose.yml   ← docker-compose up --build
```

---

## 📂 Final Folder Structure

```
d:\kenexai\
├── Dataset/
│   ├── Car_Insurance_Claim.csv     ← Classification dataset
│   └── motor_data14-2018.csv       ← Primary dataset (premium+claims)
├── data/
│   └── insurance.db                ← SQLite data warehouse
├── src/
│   ├── etl_pipeline.py
│   ├── data_simulator.py
│   ├── warehouse_schema.sql
│   ├── models/
│   │   ├── train_models.py
│   │   └── saved/                  ← pkl files
│   ├── rag/
│   │   └── rag_engine.py
│   └── optimizer/
│       └── premium_optimizer.py
├── api/
│   └── main.py                     ← FastAPI
├── app/
│   ├── streamlit_app.py
│   └── pages/
│       ├── 1_EDA_Dashboard.py
│       ├── 2_Manager_Dashboard.py
│       ├── 3_Risk_Analyst.py
│       ├── 4_Customer_Advisor.py
│       ├── 5_ML_Prediction.py
│       ├── 6_GenAI_Chat.py
│       └── 7_Premium_Optimizer.py
├── Dockerfile.api
├── Dockerfile.app
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run ETL (creates SQLite DB)
python src/etl_pipeline.py

# 3. Train ML models
python src/models/train_models.py

# 4. Start API
uvicorn api.main:app --reload --port 8000
# → Test at http://localhost:8000/docs

# 5. Start Dashboard (new terminal)
streamlit run app/streamlit_app.py
# → Visit http://localhost:8501

# OR — Docker all-in-one
docker-compose up --build
```

---

## 🎯 What Each Persona Sees

| Persona | Dashboard | Key Metrics |
|---|---|---|
| 👔 Insurance Manager | Manager Dashboard | Total Policies, Claim Rate %, Loss Ratio, Revenue |
| ⚠️ Risk Analyst | Risk Dashboard | High-Risk Vehicle Map, Claim Probability Heatmap, Trends |
| 🤝 Customer Advisor | Advisor Dashboard | Risk Score Card, Premium Recommendation, Customer Cluster |

---

> **Build Order Summary: Data → ML Models → API → RAG → Web UI → Docker**  
> **Do NOT build the web first — ML models must be saved before the API can serve them.**
