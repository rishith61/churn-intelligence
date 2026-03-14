import joblib
import os
import datetime
import hashlib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from app.utils import engineer_features

# Load the dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_pipeline.joblib")

df = pd.read_csv(DATA_PATH)

# Fingerprint the raw dataset so you can verify later what data this model was trained on
dataset_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:8]


df.drop(columns=["customerID"], inplace=True)

# feature engineering
df = engineer_features(df)
training_median = float(df["MonthlyCharges"].median())

# Categorizing columns
cols_to_exclude = ["Churn", "SeniorCitizen"]

binary_cols = [
    col
    for col in df.select_dtypes(include="O").columns
    if df[col].nunique() == 2 and col not in cols_to_exclude
]
binary_to_cast = [
    "SeniorCitizen",
    "is_new_customer",
    "is_high_value",
    "has_support_services",
]

binary_cols = binary_cols + binary_to_cast
for col in binary_to_cast:
    df[col] = df[col].astype(int)

multi_cat_cols = [
    col for col in df.select_dtypes(include="O").columns if df[col].nunique() > 2
]
numerical_cols = [
    col
    for col in df.select_dtypes(include=["int64", "float64"]).columns
    if col
    not in cols_to_exclude
    + ["is_new_customer", "has_support_services", "is_high_value"]
]
print("Numerical:", numerical_cols)
print("Binary:", binary_cols)
print("Multi-cat:", multi_cat_cols)

X = df.drop(columns=["Churn"])
y = df["Churn"]

# Check if all features are sorted according to their datatypes and if any missing features
assert len(numerical_cols + binary_cols + multi_cat_cols) == X.shape[1], (
    "Missing or duplicate columns!"
)
print("Sanity check passed")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# LabelEncoder used only to convert "Yes"/"No" → 1/0 for XGBoost.
# Not needed at inference (predict_proba handles it directly).
# Saved in artifacts for potential future multi-class use.
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Preprocessing that handles null/missing values and scales the data
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Preprocessing that handles null/missing values and encodes binary data
binary_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder()),
    ]
)

# Preprocessing that handles null/missing values and encodes multiple categories data
multi_cat_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ]
)

# Combining all the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("binary", binary_transformer, binary_cols),
        ("multi_cat", multi_cat_transformer, multi_cat_cols),
        ("numerical", numerical_transformer, numerical_cols),
    ]
)

best_params = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
}

full_pipeline = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(**best_params, random_state=42, n_jobs=-1)),
    ]
)

full_pipeline.fit(X_train, y_train_encoded)
print("Model trained")

# Evaluate Model and get a classification report
y_pred = full_pipeline.predict(X_test)
report = classification_report(y_test_encoded, y_pred, output_dict=True)
print(classification_report(y_test_encoded, y_pred))

# collect artifacts from the complete pipeline
artifacts = {
    "pipeline": full_pipeline,
    "label_encoder": le,
    "feature_names": binary_cols + multi_cat_cols + numerical_cols,
    "training_monthly_median": training_median,
    "trained_at": datetime.datetime.utcnow().isoformat(),
    "dataset_hash": dataset_hash,
    "test_metrics": {
        "precision": round(report["1"]["precision"], 3),
        "recall": round(report["1"]["recall"], 3),
        "f1": round(report["1"]["f1-score"], 3),
    },
}

# Save Pipeline
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(artifacts, MODEL_PATH)
