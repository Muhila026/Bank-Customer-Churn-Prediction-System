"""
FastAPI backend for Bank Customer Churn prediction.
Follows CI-master pattern: loads from Model/<name>/ (model.pkl, scaler.pkl, label_encoders.pkl, preprocessing_extra.pkl).
Falls back to legacy preprocessing.pkl + model_*.pkl if Model/ not present.
Exposes /predict, /model-metrics, /model-info.
"""

import json
import os
import warnings
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = PROJECT_ROOT / "Model"
MODEL_NAMES = ["XGBoost", "Gradient Boosting", "Decision Tree"]

# ----- CI-master style: load from Model/<name>/ -----
models = {}
model_metrics = {}
# Try Model/metrics.json first, then project root metrics.json (legacy)
for metrics_path in [MODEL_DIR / "metrics.json", PROJECT_ROOT / "metrics.json"]:
    try:
        with open(metrics_path) as f:
            model_metrics = json.load(f)
        break
    except Exception:
        pass

for name in MODEL_NAMES:
    folder = MODEL_DIR / name
    try:
        if (folder / "model.pkl").exists() and (folder / "preprocessing_extra.pkl").exists():
            extra = joblib.load(folder / "preprocessing_extra.pkl")
            models[name] = {
                "model": joblib.load(folder / "model.pkl"),
                "scaler": joblib.load(folder / "scaler.pkl"),
                "label_encoders": joblib.load(folder / "label_encoders.pkl"),
                "scaler_cluster": extra["scaler_cluster"],
                "kmeans": extra["kmeans"],
                "cluster_features": extra["cluster_features"],
                "feature_cols": extra["feature_cols"],
            }
        else:
            models[name] = None
    except Exception as e:
        print(f"Error loading {name}: {e}")
        models[name] = None

# ----- Fallback: legacy single preprocessing + 3 model files -----
legacy_loaded = False
if all(models.get(n) is None for n in MODEL_NAMES):
    def _pkl_path(n: str) -> Path:
        for base in (PROJECT_ROOT, BASE_DIR):
            p = base / n
            if p.exists():
                return p
        return PROJECT_ROOT / n

    PREPROCESSING_PATH = _pkl_path("preprocessing.pkl")
    MODEL_XGB_PATH = _pkl_path("model_xgb.pkl")
    MODEL_GBM_PATH = _pkl_path("model_gbm.pkl")
    MODEL_DT_PATH = _pkl_path("model_dt.pkl")

    if PREPROCESSING_PATH.exists() and MODEL_XGB_PATH.exists() and MODEL_GBM_PATH.exists() and MODEL_DT_PATH.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preprocessing = joblib.load(PREPROCESSING_PATH)
            scaler = preprocessing["scaler"]
            scaler_cluster = preprocessing["scaler_cluster"]
            kmeans = preprocessing["kmeans"]
            cluster_features = preprocessing["cluster_features"]
            feature_cols = preprocessing["feature_cols"]
            label_encoders = preprocessing["label_encoders"]
            for i, name in enumerate(MODEL_NAMES):
                path = [MODEL_XGB_PATH, MODEL_GBM_PATH, MODEL_DT_PATH][i]
                models[name] = {
                    "model": joblib.load(path),
                    "scaler": scaler,
                    "label_encoders": label_encoders,
                    "scaler_cluster": scaler_cluster,
                    "kmeans": kmeans,
                    "cluster_features": cluster_features,
                    "feature_cols": feature_cols,
                }
            legacy_loaded = True
    if not legacy_loaded and not model_metrics:
        print("No models loaded. Run bank_customer_churn_ict_u_ai.py to create Model/ and Model/metrics.json.")

models_loaded = any(models.get(n) for n in MODEL_NAMES)


def add_engineered_features_single(row: dict) -> dict:
    out = dict(row)
    balance = out.get("Balance", 0) or 0
    salary = out.get("EstimatedSalary", 0) or 0
    age = out.get("Age", 0) or 0
    tenure = out.get("Tenure", 0) or 0
    credit = out.get("CreditScore", 0) or 0
    num_prod = out.get("NumOfProducts", 0) or 0
    active = out.get("IsActiveMember", 0) or 0
    card = out.get("HasCrCard", 0) or 0
    out["BalanceToSalary"] = balance / (salary + 1)
    out["AgeTenure"] = age * (tenure + 1)
    out["CreditScoreNorm"] = (credit - 350) / 500
    out["AgeNumProducts"] = age * num_prod
    out["BalanceNumProducts"] = balance * (num_prod + 1)
    out["ActiveWithCard"] = active * card
    return out


def prepare_input(req, ref: dict) -> pd.DataFrame:
    row = {
        "CreditScore": req.CreditScore,
        "Geography": req.Geography,
        "Gender": req.Gender,
        "Age": req.Age,
        "Tenure": req.Tenure,
        "Balance": req.Balance,
        "NumOfProducts": req.NumOfProducts,
        "HasCrCard": req.HasCrCard,
        "IsActiveMember": req.IsActiveMember,
        "EstimatedSalary": req.EstimatedSalary,
    }
    label_encoders = ref["label_encoders"]
    for col in label_encoders:
        le = label_encoders[col]
        val = str(row[col])
        if val not in le.classes_:
            val = le.classes_[0]
        row[col] = le.transform([val])[0]
    cluster_row = pd.DataFrame([{c: row[c] for c in ref["cluster_features"]}])
    cluster_scaled = ref["scaler_cluster"].transform(cluster_row)
    row["Cluster"] = ref["kmeans"].predict(cluster_scaled)[0]
    row = add_engineered_features_single(row)
    return pd.DataFrame([row])[ref["feature_cols"]]


app = FastAPI(
    title="Bank Customer Churn API",
    description="API for predicting customer churn with XGBoost, Gradient Boosting, and Decision Tree",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    CreditScore: float = Field(..., ge=0, le=1000, description="Credit score (0-1000)")
    Geography: str = Field(..., description="Country: France, Spain, or Germany")
    Gender: str = Field(..., description="Male or Female")
    Age: float = Field(..., ge=0, le=120)
    Tenure: float = Field(..., ge=0, le=20, description="Years with the bank")
    Balance: float = Field(..., ge=0)
    NumOfProducts: float = Field(..., ge=0, le=10)
    HasCrCard: int = Field(..., ge=0, le=1)
    IsActiveMember: int = Field(..., ge=0, le=1)
    EstimatedSalary: float = Field(..., ge=0)


class ModelResult(BaseModel):
    exited_probability: float
    prediction: str


class PredictResponse(BaseModel):
    XGBoost: ModelResult
    Gradient_Boosting: ModelResult
    Decision_Tree: ModelResult
    message: str = Field(..., description="Interpretation based on churn risk")


@app.get("/")
def root():
    return {
        "message": "Bank Customer Churn API",
        "status": "running",
        "version": "1.0.0",
        "models": MODEL_NAMES,
    }


@app.get("/health")
def health():
    loaded = sum(1 for n in MODEL_NAMES if models.get(n))
    if loaded == 0:
        raise HTTPException(status_code=503, detail="No models loaded")
    return {"status": "healthy", "models_loaded": loaded, "model_names": MODEL_NAMES}


@app.get("/model-info")
def get_model_info():
    return {
        "models": MODEL_NAMES,
        "description": "XGBoost, Gradient Boosting, and Decision Tree trained on bank customer churn data.",
    }


@app.get("/model-metrics")
def get_model_metrics():
    """Return accuracy % and ROC-AUC per model (for frontend display)."""
    out = {}
    for name in MODEL_NAMES:
        m = model_metrics.get(name)
        if m is None:
            out[name] = {}
            continue
        m = dict(m)
        # Ensure accuracy_pct is present for frontend (0–100)
        if "accuracy_pct" in m:
            pass
        elif "accuracy" in m and m["accuracy"] is not None:
            m["accuracy_pct"] = round(float(m["accuracy"]) * 100, 2)
        out[name] = m
    return {"models": out}


@app.get("/models/status")
def models_status():
    return JSONResponse(
        status_code=200,
        content={
            "models_loaded": models_loaded,
            "model_names": MODEL_NAMES,
            "loaded": [n for n in MODEL_NAMES if models.get(n)],
        },
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not models_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Models not loaded. Run bank_customer_churn_ict_u_ai.py to create Model/ and Model/metrics.json (or legacy preprocessing.pkl and model_*.pkl).",
                "models_loaded": False,
            },
        )
    ref = models[MODEL_NAMES[0]]
    try:
        X_df = prepare_input(req, ref)
        results = {}
        for name in MODEL_NAMES:
            m = models[name]
            X_scaled = m["scaler"].transform(X_df)
            proba = float(m["model"].predict_proba(X_scaled)[0, 1])
            results[name] = proba
        def to_result(prob: float) -> ModelResult:
            p = round(prob, 4)
            pred = "Churned" if p >= 0.5 else "Stayed"
            return ModelResult(exited_probability=p, prediction=pred)
        avg_prob = (results["XGBoost"] + results["Gradient Boosting"] + results["Decision Tree"]) / 3
        if avg_prob >= 0.6:
            message = "High churn risk. Consider retention actions."
        elif avg_prob >= 0.5:
            message = "Moderate churn risk. Monitor and engage."
        else:
            message = "Low churn risk. Keep engaging positively."
        return PredictResponse(
            XGBoost=to_result(results["XGBoost"]),
            Gradient_Boosting=to_result(results["Gradient Boosting"]),
            Decision_Tree=to_result(results["Decision Tree"]),
            message=message,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
