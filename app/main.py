"""
main.py — Churn Intelligence · FastAPI Application

Endpoints:
  GET  /              → API info
  GET  /health        → model readiness check
  POST /predict/single → single customer prediction (JSON body)
  POST /predict/batch  → batch prediction (CSV / Excel upload)
"""

import io
import logging
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from app.predict import predict_churn, predict_batch_df, _get_artifacts

ALLOWED_MIME_TYPES = {
    "text/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

# Internal imports
# All inference goes through predict.py — single source of truth.
# main.py never touches the pipeline, le, or engineer_features directly.


log = logging.getLogger(__name__)


# LIFESPAN — model loaded once on startup, not at module level
# This is the FastAPI-native replacement for putting joblib.load() at the top.
# If the model file is missing, the server logs a clear error on boot and the
# /health endpoint reports it — the server does NOT crash silently.

_model_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm the model. Shutdown: nothing needed for joblib."""
    global _model_ready
    log.info("Warming model...")
    try:
        _get_artifacts()  # triggers lazy load and caches it
        _model_ready = True
        log.info("Model ready ✓")
    except FileNotFoundError as e:
        log.error("Model load failed: %s", e)
        _model_ready = False  # server stays up but health reports not ready
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Churn Intelligence API",
    description="Telco customer churn prediction · XGBoost pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve frontend static files — index.html lives at app/static/
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# REQUEST / RESPONSE SCHEMAS
# Defining these explicitly does three things:
#   1. FastAPI auto-generates accurate Swagger docs at /docs
#   2. Pydantic validates and rejects bad input before it hits the model
#   3. Your frontend team knows exactly what to send and what to expect


class CustomerInput(BaseModel):
    """
    All 19 raw feature fields the model expects.
    Fields that feed engineer_features() are required — not optional —
    because the model cannot run without them.
    """

    # Numeric (required)
    tenure: int = Field(..., ge=0, le=100, description="Months as customer (0–100)")
    MonthlyCharges: float = Field(
        ...,
        gt=0,
        le=200.0,
        description="Current monthly bill ($) — capped at $200. Values above this are uncommon in commercial telco pricing.($)",
    )
    TotalCharges: float = Field(..., ge=0, description="Cumulative charges ($)")

    # Demographics
    gender: str = Field(..., pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(
        ..., ge=0, le=1, description="1 = senior citizen, 0 = not"
    )
    Partner: str = Field(..., pattern="^(Yes|No)$")
    Dependents: str = Field(..., pattern="^(Yes|No)$")

    # Services (required — used in has_support_services feature)
    PhoneService: str = Field(..., pattern="^(Yes|No)$")
    MultipleLines: str = Field(..., pattern="^(Yes|No|No phone service)$")
    InternetService: str = Field(..., pattern="^(DSL|Fiber optic|No)$")
    OnlineSecurity: str = Field(..., pattern="^(Yes|No|No internet service)$")
    OnlineBackup: str = Field(..., pattern="^(Yes|No|No internet service)$")
    DeviceProtection: str = Field(..., pattern="^(Yes|No|No internet service)$")
    TechSupport: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingTV: str = Field(..., pattern="^(Yes|No|No internet service)$")
    StreamingMovies: str = Field(..., pattern="^(Yes|No|No internet service)$")

    # Contract & billing
    Contract: str = Field(..., pattern="^(Month-to-month|One year|Two year)$")
    PaperlessBilling: str = Field(..., pattern="^(Yes|No)$")
    PaymentMethod: str = Field(
        ...,
        pattern="^(Electronic check|Mailed check|Bank transfer \\(automatic\\)|Credit card \\(automatic\\))$",
    )

    @model_validator(mode="after")
    def total_charges_consistent(self):
        """
        Validates TotalCharges is consistent with tenure and MonthlyCharges.
        model_validator runs after ALL fields are validated — no field ordering risk.
        """
        if self.tenure > 0 and self.TotalCharges == 0:
            raise ValueError("TotalCharges cannot be 0 for a customer with tenure > 0.")
        if self.tenure == 0 and self.TotalCharges > self.MonthlyCharges:
            raise ValueError(
                f"New customer (tenure=0) has TotalCharges={self.TotalCharges} "
                f"which exceeds MonthlyCharges={self.MonthlyCharges}. "
                "TotalCharges should be 0 or at most one month's charge."
            )
        return self


class SinglePredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Churn probability 0–100")
    churn_risk: str = Field(..., description="Low | Medium | High")
    churn_risk_score: int = Field(..., description="1–10 risk score")
    recommended_action: str = Field(..., description="Suggested retention action")


class BatchSummaryResponse(BaseModel):
    total_customers: int
    high_risk: int
    medium_risk: int
    low_risk: int
    avg_churn_probability: float


# ROUTES


@app.get("/", tags=["Info"], include_in_schema=False)
def root():
    """Serve the dashboard frontend."""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"name": "Churn Intelligence API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health", tags=["Info"])
def health():
    """
    Real health check — verifies the model is loaded and ready.
    Returns 503 if the model failed to load on startup.
    The frontend / load balancer should poll this before sending predictions.
    """
    if not _model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs.",
        )
    return {"status": "healthy", "model": "ready"}


@app.post(
    "/predict/single",
    response_model=SinglePredictionResponse,
    tags=["Prediction"],
    summary="Predict churn for one customer",
)
def predict_single(customer: CustomerInput):
    """
    Accepts a single customer's features as JSON.
    Called by the dashboard form when the user clicks 'Run Prediction'.
    All inference logic lives in predict.py — this endpoint just handles
    HTTP concerns (parsing, validation, error formatting).
    """
    df = pd.DataFrame([customer.model_dump()])

    try:
        result = predict_churn(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


@app.post(
    "/predict/batch",
    tags=["Prediction"],
    summary="Predict churn for a batch of customers (CSV / Excel)",
)
async def predict_batch_endpoint(file: UploadFile = File(...)):
    """
    Accepts a CSV or Excel file upload.
    Returns a downloadable CSV with columns:
      customerID (if present) | churn_probability | churn_risk | churn_risk_score | recommended_action

    Also returns a JSON summary in the response headers:
      X-Total-Customers, X-High-Risk, X-Medium-Risk, X-Low-Risk
    so the frontend can show a summary card without parsing the CSV.
    """
    # Validate file type
    filename = file.filename or ""
    if not filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{filename}'. Upload a .csv, .xlsx, or .xls file.",
        )

    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected content type '{file.content_type}'. Upload a valid CSV or Excel file.",
        )

    # Read uploaded bytes into DataFrame
    contents = await file.read()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file contains no rows.")

    #  Run inference directly from DataFrame — no disk I/O needed
    try:
        results = predict_batch_df(df)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    #  Build summary
    risk_counts = results["churn_risk"].value_counts()
    summary = {
        "total_customers": len(results),
        "high_risk": int(risk_counts.get("High", 0)),
        "medium_risk": int(risk_counts.get("Medium", 0)),
        "low_risk": int(risk_counts.get("Low", 0)),
        "avg_churn_probability": round(float(results["churn_probability"].mean()), 1),
    }

    #  Stream results CSV back to client
    output = io.StringIO()
    results.to_csv(output, index=False)
    output.seek(0)

    download_name = filename.rsplit(".", 1)[0] + "_predictions.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={download_name}",
            # Summary in headers so frontend can show stats without parsing CSV
            "X-Total-Customers": str(summary["total_customers"]),
            "X-High-Risk": str(summary["high_risk"]),
            "X-Medium-Risk": str(summary["medium_risk"]),
            "X-Low-Risk": str(summary["low_risk"]),
            "X-Avg-Churn-Probability": str(summary["avg_churn_probability"]),
        },
    )
