from __future__ import annotations
import pandas as pd
from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# numeric columns we will coerce; anything else stays categorical
NUMERIC_COLS: List[str] = [
    "Age",
    "Number_of_Referrals",
    "Tenure_in_Months",
    "Monthly_Charge",
    "Total_Charges",
    "Total_Refunds",
    "Total_Extra_Data_Charges",
    "Total_Long_Distance_Charges",
    "Total_Revenue",
]

# columns never used for training/inference
DROP_ALWAYS: List[str] = [
    "Customer_ID",           # high-cardinality identifier
    "Churn_Category",        # leakage/explanation text
    "Churn_Reason",          # leakage/explanation text
    "Customer_Status_Predicted",  # appears in sample pred file
]

def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # fix obvious bad values
    if "Monthly_Charge" in df.columns:
        df.loc[df["Monthly_Charge"] < 0, "Monthly_Charge"] = pd.NA
    return df

def prepare_training_frame(df: pd.DataFrame):
    """
    Filters rows to 'Stayed' vs 'Churned', builds y, and returns (X, y, preprocessor)
    """
    df = df[df["Customer_Status"].isin(["Stayed", "Churned"])].copy()
    y = (df["Customer_Status"] == "Churned").astype(int)

    X = df.drop(columns=["Customer_Status"] + [c for c in DROP_ALWAYS if c in df.columns], errors="ignore")
    X = _coerce_numerics(X)

    # infer column types
    num_cols = [c for c in X.columns if c in NUMERIC_COLS and c in X.columns]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop", sparse_threshold=1.0)

    return X, y, pre

def prepare_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops label/leakage columns, coerces numerics, returns X for prediction.
    Works even if CSV contains 'Joined' or label columns.
    """
    X = df.drop(columns=["Churn", "Customer_Status"] + DROP_ALWAYS, errors="ignore").copy()
    X = _coerce_numerics(X)
    return X
