import os
import sys
import pandas as pd
import json
import config
from src import data_loader, feature_engineering

def main():
    # Find Excel file in data/
    excel_files = [f for f in os.listdir("data") if f.lower().endswith(".xlsx") or f.lower().endswith(".xls")]
    print("10.***", excel_files)
    if not excel_files:
        print("No Excel file found in 'data/'.", file=sys.stderr)
        sys.exit(1)
    excel_path = os.path.join("data", excel_files[2]) ### changed to 2 from 0
    print("9. **** excel path =", excel_path)
    sheets = data_loader.load_workbook(excel_path)
    print(f"Loaded workbook: {excel_path}")
    for name, df in sheets.items():
        print(f"Sheet: {name}  Shape: {df.shape}")
    # Profile
    profiles = data_loader.profile_workbook(sheets)
    # Save summary to outputs/reports/
    with open("outputs/reports/dataset_profile.json", "w") as f:
        json.dump(profiles, f, indent=2)
    # For main sheet, infer features
    main_sheet = list(sheets.keys())[0]
    df = sheets[main_sheet]
    feat_defs = feature_engineering.infer_feature_definitions(df)
    print("\n=== DATASET SUMMARY ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Detected columns:", ", ".join(df.columns))
    print("Selected input features:", feat_defs["input_features"])
    print("Selected categorical features:", feat_defs["cat_features"])
    print("Unavailable canonical columns:", [k for k in feature_engineering.CANONICALS if k not in feat_defs["col_map"]])
    # Task discovery
    # If no future/label columns, it's unsupervised profiling
    future_cols = feat_defs["future_target_candidates"]

    if future_cols:
        problem_type = "Potential supervised learning"
        learning_approach = "Requires verification of a genuine future outcome"
        outputs = future_cols
        loss = "Determined after target validation"
        optimizer = "Determined after target validation"
        metrics = "Determined after target validation"
        arch = "Determined after target validation"
        why = (
            "The workbook contains a candidate future outcome, "
            "but it must be verified before being treated as a target."
        )
    else:
        problem_type = "Unsupervised cross-sectional financial profiling"
        learning_approach = (
            "Self-supervised reconstruction learning "
            "+ unsupervised clustering/anomaly detection"
        )
        outputs = (
            "Latent financial representation, reconstruction error, "
            "anomaly score, and company profile"
        )
        loss = "Mean Squared Reconstruction Error"
        optimizer = "Adam with weight decay"
        metrics = (
            "MAE, MSE, RMSE, latent-space silhouette score, "
            "and anomaly diagnostics"
        )
        arch = (
            "Small fully-connected autoencoder: "
            "Input → 32 → 16 → Latent → 16 → 32 → Output"
        )
        why = (
            "The spreadsheet contains current financial characteristics "
            "but no observed future outcome. A supervised forecasting "
            "target would require fabricated labels."
        )
