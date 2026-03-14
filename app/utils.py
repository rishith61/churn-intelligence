import pandas as pd


def engineer_features(df: pd.DataFrame, monthly_median: float = 70.35) -> pd.DataFrame:

    df = df.copy()

    # 1. Clean TotalCharges (Handle blanks/strings)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # 2. Engineer features
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # New customer logic
    df["is_new_customer"] = (df["tenure"] <= 12).astype(int)

    # Support services logic
    df["has_support_services"] = (df["OnlineSecurity"] == "Yes").astype(int) + (
        df["TechSupport"] == "Yes"
    ).astype(int)

    # 3. FIX: Use fixed median to prevent single-row inference failure
    df["is_high_value"] = (df["MonthlyCharges"] > monthly_median).astype(int)

    # 4. Final Type Casting (Ensures everything is an int for the model)
    binary_to_cast = [
        "SeniorCitizen",
        "is_new_customer",
        "is_high_value",
        "has_support_services",
    ]
    for col in binary_to_cast:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df
