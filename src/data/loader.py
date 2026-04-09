"""
src/data/loader.py
------------------
Data loading and cleaning pipeline for Amazon product dataset.
"""

import pandas as pd
import numpy as np


def load_and_clean(file_path: str) -> pd.DataFrame:
    """
    Load raw Amazon CSV and return a fully cleaned, analysis-ready DataFrame.

    Steps
    -----
    1. Parse price strings (₹ symbol, commas) → float
    2. Normalize discount_percentage to 0-1
    3. Fix rating / rating_count formatting
    4. Standardize category names
    5. Impute missing rating_count with median
    6. Engineer derived columns
    7. Drop rows with critical missing values
    """
    df = pd.read_csv(file_path)

    # --- Price fields ---
    for col in ["discounted_price", "actual_price"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Discount percentage → 0-1 ---
    df["discount_percentage"] = (
        df["discount_percentage"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce") / 100

    # --- Rating ---
    df["rating"] = pd.to_numeric(
        df["rating"].astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )

    # --- Rating count ---
    df["rating_count"] = (
        df["rating_count"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")
    median_rc = df["rating_count"].median()
    df["rating_count"] = df["rating_count"].fillna(median_rc)

    # --- Category ---
    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.strip().str.lower()

    # --- Derived columns ---
    df["price_discount_amount"] = df["actual_price"] - df["discounted_price"]
    df["discount_pct_100"] = df["discount_percentage"] * 100.0
    df["discount_ratio"] = df["discounted_price"] / df["actual_price"]

    # --- Drop rows missing critical fields ---
    required = ["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]
    df_clean = df.dropna(subset=required).copy()

    # --- Log transform for variance stabilisation ---
    df_clean["log_rating_count"] = np.log1p(df_clean["rating_count"])

    print(f"[loader] Raw rows: {len(df):,} → Clean rows: {len(df_clean):,}")
    return df_clean
