from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional
from app.utils import engineer_features
import joblib
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────

# create a logger instance for this module
# Basically tells from which module/file any error or info is coming from
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Defining paths dynamically so that when uploaded to server code doesn't breakdown.

# os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# This line gets the absolute path of the current file (predict.py), then goes up one level (to app/), then up another level (to churn/).
# This ensures that no matter where the script is run from, it always finds its model and data relative to its own location.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# os.path.join is a "smart" string builder. Its only job is to
# combine folder names and file names into a single,
# valid path that your computer's operating system (OS) can understand.

MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_pipeline.joblib")
# This line points to the model file

DATA_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
# This line points to the dataset file

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
# This line points to the output directory

# ── Required columns ─────────────────────────────────────────────────────────
REQUIRED_COLUMNS = {
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
}


# RISK HELPERS
# probability  → risk label, risk score (1–10), recommended action
#
# Thresholds:
#   < 30%  → Low    score 1–3   standard retention comms
#   30–60% → Medium score 4–6   proactive outreach
#   > 60%  → High   score 7–10  immediate intervention
#
# Adjust thresholds after reviewing precision/recall tradeoffs on your data.


def _get_risk_label(probability: float):
    if probability >= 60:
        return "High"
    elif probability >= 30:
        return "Medium"
    else:
        return "Low"


def _get_risk_score(probability: float):

    # Map 0–100 probability to a 1–10 integer score.
    # Gives analysts a gradient within each risk tier:
    # Low    → 1–3   Medium → 4–6   High → 7–10
    # Formula: simple linear scale clamped to 1–10.
    return max(1, min(10, round(probability / 10)))


def _get_recommended_action(probability: float):
    # Translate churn probability into actionable insights.
    if probability >= 80:
        return "Escalate to account manager, immediate retention call"
    elif probability >= 60:
        return "Offer personalised retention discount or upgrade"
    elif probability >= 30:
        return "Schedule proactive check-in within 7 days"
    else:
        return "Standard engagement , monitor next billing cycle"


# LAZY MODEL LOADER

# Initally model is not loaded and so we avoid wasting time by loading it only when needed.
# This is called lazy loading.
_artifacts: Optional[dict] = None


def _get_artifacts() -> dict:
    global _artifacts
    # If model is not loaded, load it
    if _artifacts is None:
        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}\n"
                "Run train.py first to generate the model artifact."
            )
        log.info("Loading model from %s", MODEL_PATH)
        # time the model loading process
        t0 = time.perf_counter()
        # Load the model
        _artifacts = joblib.load(MODEL_PATH)
        # Calculate the time taken to load the model
        elapsed = (time.perf_counter() - t0) * 1000
        # Log the time taken to load the model and the pipeline steps
        log.info(
            "Model loaded in %.1f ms  |  Pipeline steps: %s",
            elapsed,
            [s[0] for s in _artifacts["pipeline"].steps],
        )
    return _artifacts


# SCHEMA VALIDATION


def _validate_schema(df: pd.DataFrame, context: str = "input"):
    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError(f"The {context} DataFrame is empty — no rows to predict.")
    # Check if the required columns are present in the DataFrame
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        # Raise an error if the required columns are not present
        raise ValueError(
            f"The {context} data is missing {len(missing)} required column(s):\n"
            f"  {sorted(missing)}\n"
            f"Check that your file matches the expected schema."
        )


# CORE INFERENCE


def _run_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    engineer features → pipeline → attach churn_probability + churn_risk.
    Intentionally drops all intermediate feature columns from output —
    callers only ever need the two result columns, not the engineered features.
    """
    artifacts = _get_artifacts()
    pipeline = artifacts["pipeline"]
    monthly_median = artifacts["training_monthly_median"]
    # label_encoder not needed: predict_proba returns class-1 probability directly

    # Copy the DataFrame to avoid modifying the original DataFrame
    df_engineered = engineer_features(df.copy(), monthly_median=monthly_median)

    t0 = time.perf_counter()

    # Predict the probability of churn
    probas = pipeline.predict_proba(df_engineered)[:, 1]
    # Calculate the time taken to predict the probability of churn
    elapsed = (time.perf_counter() - t0) * 1000

    # Log the time taken to predict the probability of churn
    log.info(
        "Inference complete — %d row(s) in %.1f ms  (%.3f ms/row)",
        len(df),
        elapsed,
        elapsed / len(df),
    )

    # Build clean results — only the columns that matter
    probs_pct = (probas * 100).astype(float).round(1)
    results = pd.DataFrame(
        {
            "churn_probability": probs_pct,
            "churn_risk": [_get_risk_label(p) for p in probs_pct],
            "churn_risk_score": [_get_risk_score(p) for p in probs_pct],
            "recommended_action": [_get_recommended_action(p) for p in probs_pct],
        }
    )
    return results


# PUBLIC API


def predict_churn(customer_data: pd.DataFrame) -> dict:
    """
    Predict churn for a SINGLE customer row.
    Called by FastAPI when the dashboard form is submitted.

    Returns
    -------
    {
        "churn_probability":  73.4,                                    ← 0–100 float
        "churn_risk":         "High",                                  ← "Low" | "Medium" | "High"
        "churn_risk_score":   7,                                       ← 1–10 integer
        "recommended_action": "Offer personalised retention discount"  ← plain-English next step
    }
    """
    if len(customer_data) != 1:
        raise ValueError(
            f"predict_churn expects exactly 1 row; got {len(customer_data)}. "
            "Use predict_batch() for multiple customers."
        )
    _validate_schema(customer_data, context="single-customer")
    result = _run_inference(customer_data)

    # dictionaries cannot hold dataframes , so we extract data from the
    # df and return it as a dictionary
    return {
        "churn_probability": round(float(result["churn_probability"].iloc[0]), 1),
        "churn_risk": result["churn_risk"].iloc[0],
        "churn_risk_score": int(result["churn_risk_score"].iloc[0]),
        "recommended_action": result["recommended_action"].iloc[0],
    }


def predict_batch_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict churn for a DataFrame directly.
    Called by FastAPI batch endpoint — no file I/O, no disk writes.
    DataFrame is already parsed from the uploaded file bytes in memory.

    Returns
    -------
    pd.DataFrame with columns: [customerID,] churn_probability, churn_risk,
                                churn_risk_score, recommended_action
    """
    id_col = None
    if "customerID" in df.columns:
        id_col = df["customerID"].copy()
        df = df.drop(columns=["customerID"])

    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    _validate_schema(df, context="batch DataFrame")
    results = _run_inference(df)

    if id_col is not None:
        results.insert(0, "customerID", id_col.reset_index(drop=True))

    dist = results["churn_risk"].value_counts()
    log.info(
        "Risk distribution — High: %d  Medium: %d  Low: %d",
        dist.get("High", 0),
        dist.get("Medium", 0),
        dist.get("Low", 0),
    )

    return results


def predict_batch(file_path: str, output_path: str | None = None) -> pd.DataFrame:
    """
    Predict churn for ALL customers in a CSV or Excel file.
    Called by the CLI — reads from disk, saves results to disk.
    Output CSV contains: customerID (if present) + churn_probability +
                         churn_risk + churn_risk_score + recommended_action

    Returns
    -------
    pd.DataFrame with columns: [customerID,] churn_probability, churn_risk,
                                churn_risk_score, recommended_action
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Use .csv, .xlsx, or .xls.")

    log.info("Loaded %d rows from %s", len(df), os.path.basename(file_path))

    # Run inference via the DataFrame path — no duplication of logic
    results = predict_batch_df(df)

    # Save to outputs/
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base}_predictions.csv")

    results.to_csv(output_path, index=False)
    log.info("Results saved → %s", output_path)

    return results


# This is a command-line interface (CLI) for the churn prediction model.
# It allows you to run predictions on a batch of customers from a CSV or Excel file.
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Churn Intelligence — Batch Inference Tool",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the input CSV or Excel file.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Path to save results. Defaults to the 'outputs' folder.",
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args()

    # Since we only do batch via CLI now, we just call it directly
    try:
        predict_batch(file_path=args.file, output_path=args.output)
    except Exception as e:
        log.error("Batch processing failed: %s", e)


# ── Risk thresholds ───────────────────────────────────────────────────────────
# TODO: Thresholds are currently applied to raw XGBoost probabilities.
# XGBoost + SMOTE outputs are not guaranteed to be calibrated.
# Add CalibratedClassifierCV (isotonic) to train.py and validate
# thresholds against actual precision/recall before using in production.
#
#   < 30%  → Low    score 1–3   standard retention comms
#   30–60% → Medium score 4–6   proactive outreach
#   > 60%  → High   score 7–10  immediate intervention
