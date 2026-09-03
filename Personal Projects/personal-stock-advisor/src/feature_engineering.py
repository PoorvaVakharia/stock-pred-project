import pandas as pd
import numpy as np
import json
import os
import re

# Map normalized column names to canonical names
CANONICALS = {
    "company": ["company", "stock", "name", "ticker", "stock name"],
    "current_price": ["current price", "price", "share price", "market price"],
    "price_as_of": ["stock price as of", "price as of", "as of"],
    "pe_ratio": ["p/e ratio", "p/e", "pe ratio", "pe"],
    "eps": ["earnings per share", "earnigs per share", "eps"],
    "roe": ["return on equity", "roe"],
    "dividend_per_share": ["dividend per share", "dividend/share"],
    "dividend_yield": ["dividend yield"],
    "net_income": ["net income", "profit", "earnings"],
    "operating_cash_flow": [
        "operating cashflow",
        "operating cash flow",
        "cash flow",
        "ocf"
    ],
    "sharpe_ratio": ["sharpe ratio", "sharpe"],
    "beta": ["beta"],
    "expense_ratio": ["expense ratio"],
    "alpha": ["alpha"],
    "sector": ["sector", "industry"],
    "type": ["type"],
    "location": ["location"]
}

def normalize(col):
    return re.sub(r"[^a-z0-9]", "", col.lower())

def infer_feature_definitions(df):
    col_map = {}

    norm_cols = {
        normalize(str(c)): c
        for c in df.columns
    }

    for key, aliases in CANONICALS.items():
        for alias in aliases:
            alias_norm = normalize(alias)

            for norm_col, original_col in norm_cols.items():
                if norm_col == alias_norm:
                    col_map[key] = original_col
                    break

            if key in col_map:
                break

    # Numerical financial features
    numeric_features = [
        "current_price",
        "pe_ratio",
        "eps",
        "roe",
        "dividend_per_share",
        "dividend_yield",
        "net_income",
        "operating_cash_flow",
        "sharpe_ratio",
        "beta",
        "expense_ratio",
        "alpha"
    ]

    input_features = [
        col_map[key]
        for key in numeric_features
        if key in col_map
    ]

    # Categorical features
    cat_features = [
        col_map[key]
        for key in ["sector", "type"]
        if key in col_map
    ]

    # IMPORTANT:
    # Do not automatically turn "return", "alpha", "ROE", etc.
    # into prediction targets.
    target = None

    future_target_candidates = []

    for column in df.columns:
        normalized = normalize(str(column))

        if any(token in normalized for token in [
            "futureprice",
            "futureprice",
            "futurereturn",
            "forwardreturn",
            "nextreturn",
            "pricechange",
            "futureperformance"
        ]):
            future_target_candidates.append(column)

    return {
        "col_map": col_map,
        "input_features": input_features,
        "cat_features": cat_features,
        "target": target,
        "future_target_candidates": future_target_candidates,
        "engineered": [],
        "processed_features": []
    }
    return feature_defs

def engineer_features(df, feature_defs):
    # Copy only relevant columns
    cols = feature_defs["input_features"] + feature_defs["cat_features"]
    X = df[cols].copy()
    # Engineered features
    col_map = feature_defs["col_map"]
    # EPS proxy
    if "current_price" in col_map and "pe_ratio" in col_map:
        with np.errstate(divide="ignore", invalid="ignore"):
            eps = df[col_map["current_price"]] / df[col_map["pe_ratio"]]
        X["eps_proxy"] = eps
    # dividend_payout_proxy
    if "dividend_per_share" in col_map and "current_price" in col_map and "pe_ratio" in col_map:
        with np.errstate(divide="ignore", invalid="ignore"):
            eps = df[col_map["current_price"]] / df[col_map["pe_ratio"]]
            payout = df[col_map["dividend_per_share"]] / eps
        X["dividend_payout_proxy"] = payout
    # earnings_yield_proxy
    if "pe_ratio" in col_map:
        with np.errstate(divide="ignore", invalid="ignore"):
            X["earnings_yield_proxy"] = 1.0 / df[col_map["pe_ratio"]]
    # price_to_cash_flow_proxy
    if "current_price" in col_map and "operating_cash_flow" in col_map:
        with np.errstate(divide="ignore", invalid="ignore"):
            X["price_to_cash_flow_proxy"] = df[col_map["current_price"]] / df[col_map["operating_cash_flow"]]
    # Log transforms for positive numeric features
    for col in X.select_dtypes(include=[np.number]).columns:
        if (X[col] > 0).all():
            X[f"log_{col}"] = np.log1p(X[col])
    # Robust normalization (z-score)
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        med = X[col].median()
        mad = np.median(np.abs(X[col] - med))
        if mad > 0:
            X[f"{col}_robustnorm"] = (X[col] - med) / mad
    # Track new features
    feature_defs["engineered"] = [c for c in X.columns if c not in cols]
    feature_defs["processed_features"] = list(X.columns)
    return X

def save_feature_definitions(feature_defs, path):
    with open(path, "w") as f:
        json.dump(feature_defs, f, indent=2)

def load_feature_definitions(path):
    with open(path, "r") as f:
        return json.load(f)
