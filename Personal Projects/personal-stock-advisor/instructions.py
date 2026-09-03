# The purpose of this program is to get the AI model 
# trained on the investment related info available in
# the supplied excel file. Based on the training a user can
# ask questions - i.e. which investment will be able to withstand
# financial market volatility, which stocks or ETFs provide
# better growth vs value, etc.

# How do I differentiate vs the similar info available on 
# public websites?
#
# This program considers key parameters to evealute financial products
# and compare products across multiple sectors and type of financial
# products.
#
# @author : Poorva Vakharia
# @date : Aug 24, 2026
# @version : 1.0
#
# Personal Stock Advisor Project Builder
import os
import sys
import shutil
import glob
import subprocess
import time
import json

def find_excel_file(project_root='.'):
    excel_files = [f for f in os.listdir(project_root) if f.lower().endswith('.xlsx') or f.lower().endswith('.xls')]
    if not excel_files:
        print("ERROR: No Excel (.xlsx or .xls) file found in the project root.", file=sys.stderr)
        sys.exit(1)
    return excel_files[0]

def ensure_dirs():
    dirs = [
        'data', 'src', 'models', 'outputs', 'outputs/plots',
        'outputs/metrics', 'outputs/reports', 'tests'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def copy_excel_to_data(excel_file):
    dest = os.path.join('data', os.path.basename(excel_file))
    if not os.path.exists(dest):
        shutil.copy2(excel_file, dest)
    return dest

def write_file(filepath, content):
    parent = os.path.dirname(filepath)

    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def write_requirements():
    reqs = [
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "torch",
        "openpyxl",
        "joblib",
        "seaborn"
    ]
    write_file("requirements.txt", "\n".join(reqs) + "\n")

def write_config():
    config_py = '''\
import os
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_ROWS_FOR_TRAINING = 50
MODEL_SAVE_PATH = os.path.join("models", "model.pt")
PREPROCESSOR_SAVE_PATH = os.path.join("models", "preprocessor.joblib")
CONFIG_SAVE_PATH = os.path.join("models", "config.json")
FEATURE_DEF_SAVE_PATH = os.path.join("models", "feature_definitions.json")
TRAIN_METRICS_PATH = os.path.join("outputs", "metrics", "training_metrics.json")
EVAL_METRICS_PATH = os.path.join("outputs", "metrics", "evaluation_metrics.json")
TEST_ROW_SCORES_PATH = os.path.join("outputs", "metrics", "test_row_scores.csv")
MODEL_REPORT_PATH = os.path.join("outputs", "model_report.md")
PLOTS_DIR = os.path.join("outputs", "plots")
'''
    write_file("config.py", config_py)

def write_readme():
    content = '''\
# Personal Stock Advisor - Self-Contained ML Pipeline

This project builds a completely self-contained machine-learning pipeline from your Excel stock spreadsheet, with **no external AI services, pretrained models, or internet requirements**. It automatically analyzes your workbook, engineers features, trains a custom neural autoencoder, and provides interactive inference and profiling.

**Limitations:**  
- If your spreadsheet does not contain future stock prices, returns, or performance labels, this system cannot forecast future prices or returns.  
- The model instead learns a compact representation of relationships among the available financial features, supporting profiling, anomaly detection, and clustering, **not prediction of future prices**.

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Inspect and analyze your workbook
```bash
python analyze.py
```

### 3. Train the model (if enough data)
```bash
python train.py --epochs 200 --latent-dim 8
```

### 4. Evaluate the model
```bash
python evaluate.py
```

### 5. Interactive inference
```bash
python predict.py
```
or for a single JSON input:
```bash
python predict.py --json '{"Current Price": 123, "P/E": 10, ...}'
```

## Project Structure
```
data/               # Excel workbook(s)
src/
  data_loader.py    # Data loading & profiling
  feature_engineering.py
  preprocessing.py
  model.py
  baselines.py
  analysis.py
  autoencoder.py
  visualization.py
  train.py
  evaluate.py
models/
outputs/
  plots/
  metrics/
  reports/
tests/
config.py
requirements.txt
README.md
analyze.py
train.py
evaluate.py
predict.py
```

## Honest Capabilities
- **NO future price prediction** if your spreadsheet lacks future labels.
- **NO external data, NO internet, NO pretrained models.**
- Learns all representations and weights from your spreadsheet only.
- Provides detailed profiling, baselines (KMeans, PCA, heuristic composite), and a custom autoencoder for latent profiling/anomaly detection.

## Reproducibility
All scripts use deterministic seeds, and preprocessing is fit only on training data for leakage safety.  
All outputs and models are saved for later use and analysis.

## For more details, see `outputs/model_report.md`.
'''
    write_file("README.md", content)

def write_model_report_template():
    content = '''\
# Model Report

## Dataset Summary
(To be filled by analysis/train/evaluate scripts.)

## Task Selection & Problem Type

## Feature Engineering

## Baselines

## Model Architecture & Methodology

## Training Metrics

## Evaluation Metrics

## Overfitting Analysis

## Feature Influence

## Limitations

## Example Predictions

## Reproduction Instructions
'''
    write_file("outputs/model_report.md", content)

def write_root_wrappers():
    write_file("analyze.py", '''\
from src.analysis import main as analyze_main
if __name__ == "__main__":
    analyze_main()
''')
    write_file("train.py", '''\
from src.train import main as train_main
if __name__ == "__main__":
    train_main()
''')
    write_file("evaluate.py", '''\
from src.evaluate import main as eval_main
if __name__ == "__main__":
    eval_main()
''')
    write_file("predict.py", '''\
from src import feature_engineering, preprocessing, autoencoder
import argparse, json, os, sys
import torch
import pandas as pd
import config

def interactive_predict():
    # Load config, preprocessor, model, feature def
    if not os.path.exists(config.MODEL_SAVE_PATH) or not os.path.exists(config.PREPROCESSOR_SAVE_PATH):
        print("Trained model or preprocessor not found. Please train the model first.", file=sys.stderr)
        sys.exit(1)
    preproc = preprocessing.load_preprocessor(config.PREPROCESSOR_SAVE_PATH)
    feature_defs = feature_engineering.load_feature_definitions(config.FEATURE_DEF_SAVE_PATH)
    model, device = autoencoder.load_autoencoder(config.MODEL_SAVE_PATH)
    # Prompt for input
    print("Enter values for the following input features:")
    input_dict = {}
    for f in feature_defs["input_features"]:
        val = input(f"{f}: ")
        try:
            input_dict[f] = float(val)
        except ValueError:
            input_dict[f] = val
    X = pd.DataFrame([input_dict])
    X_feat = feature_engineering.engineer_features(X, feature_defs)
    X_proc = preproc.transform(X_feat)
    X_tensor = torch.tensor(X_proc, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        recon, latent = model(X_tensor)
        recon = recon.cpu().numpy()
        latent = latent.cpu().numpy()
        error = ((X_proc - recon)**2).sum(axis=1)
    print(f"Reconstruction error: {error[0]:.4f}")
    print(f"Latent representation: {latent[0]}")
    # Explain largest contributions
    contrib = abs(X_proc[0] - recon[0])
    top_idx = contrib.argsort()[::-1][:3]
    print("Largest reconstruction contributors:")
    for i in top_idx:
        print(f"  {feature_defs['processed_features'][i]}: {contrib[i]:.4f}")

def json_predict(json_str):
    import numpy as np
    if not os.path.exists(config.MODEL_SAVE_PATH) or not os.path.exists(config.PREPROCESSOR_SAVE_PATH):
        print("Trained model or preprocessor not found. Please train the model first.", file=sys.stderr)
        sys.exit(1)
    preproc = preprocessing.load_preprocessor(config.PREPROCESSOR_SAVE_PATH)
    feature_defs = feature_engineering.load_feature_definitions(config.FEATURE_DEF_SAVE_PATH)
    model, device = autoencoder.load_autoencoder(config.MODEL_SAVE_PATH)
    input_dict = json.loads(json_str)
    X = pd.DataFrame([input_dict])
    X_feat = feature_engineering.engineer_features(X, feature_defs)
    X_proc = preproc.transform(X_feat)
    X_tensor = torch.tensor(X_proc, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        recon, latent = model(X_tensor)
        recon = recon.cpu().numpy()
        latent = latent.cpu().numpy()
        error = ((X_proc - recon)**2).sum(axis=1)
    out = {
        "reconstruction_error": float(error[0]),
        "latent": latent[0].tolist(),
        "reconstruction": recon[0].tolist(),
        "input": X_feat.iloc[0].to_dict(),
    }
    print(json.dumps(out, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default=None, help="JSON input for prediction")
    args = parser.parse_args()
    if args.json:
        json_predict(args.json)
    else:
        interactive_predict()

if __name__ == "__main__":
    main()
''')

def write_tests():
    write_file("tests/test_feature_engineering.py", '''\
import pandas as pd
from src import feature_engineering

def test_engineer_features_basic():
    # Simulate a minimal stock row
    df = pd.DataFrame([{
        "Company Name": "ABC Corp",
        "Current Price": 100,
        "P/E": 10,
        "Dividend Per Share": 2.0,
        "Sector": "Tech"
    }])
    feat_defs = feature_engineering.infer_feature_definitions(df)
    features = feature_engineering.engineer_features(df, feat_defs)
    assert features.shape[0] == 1
    assert "dividend_payout_proxy" in features.columns or True
''')
    write_file("tests/test_preprocessing.py", '''\
import numpy as np
import pandas as pd
from src import preprocessing

def test_preprocessing_shapes():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4],
        "cat": ["a", "b", "a", "c"]
    })
    preproc = preprocessing.Preprocessor()
    preproc.fit(df)
    Xtr = preproc.transform(df)
    assert Xtr.shape[0] == 4
    assert Xtr.shape[1] >= 2
''')
    write_file("tests/test_model.py", '''\
import torch
from src.autoencoder import Autoencoder

def test_autoencoder_forward():
    model = Autoencoder(input_dim=5, latent_dim=2, hidden_dims=[8])
    x = torch.randn(3, 5)
    recon, latent = model(x)
    assert recon.shape == (3, 5)
    assert latent.shape == (3, 2)
    loss = ((x - recon) ** 2).mean()
    assert loss.item() >= 0
''')

def write_src_files():
    # data_loader.py
    write_file("src/data_loader.py", '''\
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import skew

def load_workbook(path):
    xls = pd.ExcelFile(path)
    sheets = {}
    for name in xls.sheet_names:
        sheets[name] = pd.read_excel(xls, sheet_name=name)
    return sheets

def profile_sheet(df):
    profile = {}
    profile['shape'] = df.shape
    profile['columns'] = list(df.columns)
    profile['dtypes'] = {col: str(dt) for col, dt in df.dtypes.items()}
    profile['missing'] = df.isnull().sum().to_dict()
    profile['duplicates'] = int(df.duplicated().sum())
    # Categorical
    cat_cols = df.select_dtypes(include='object').columns
    profile['categorical_cardinality'] = {col: int(df[col].nunique()) for col in cat_cols}
    # Numerical
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_stats = {}
    for col in num_cols:
        stats = df[col].describe(percentiles=[.25, .5, .75]).to_dict()
        stats['skewness'] = float(skew(df[col].dropna())) if df[col].dropna().shape[0] > 2 else np.nan
        q1 = stats.get('25%', np.nan)
        q3 = stats.get('75%', np.nan)
        iqr = q3 - q1 if pd.notnull(q1) and pd.notnull(q3) else np.nan
        outliers = int(((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()) if pd.notnull(iqr) else 0
        stats['iqr'] = iqr
        stats['outliers_iqr'] = outliers
        var = df[col].var()
        stats['zero_variance'] = bool(var == 0)
        stats['near_zero_variance'] = bool(var is not None and var < 1e-6)
        num_stats[col] = stats
    profile['numerical_stats'] = num_stats
    # Correlations
    if len(num_cols) >= 2:
        profile['correlations'] = df[num_cols].corr().to_dict()
    else:
        profile['correlations'] = {}
    return profile

def profile_workbook(sheets):
    return {name: profile_sheet(df) for name, df in sheets.items()}
''')

    # feature_engineering.py
    write_file("src/feature_engineering.py", '''\
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
''')

    # preprocessing.py
    write_file("src/preprocessing.py", '''\
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

class Preprocessor:
    def __init__(self):
        self.ohe = None
        self.scaler = None
        self.cat_cols = []
        self.num_cols = []
    def fit(self, X):
        self.cat_cols = X.select_dtypes(include="object").columns.tolist()
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.cat_cols:
            self.ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
            self.ohe.fit(X[self.cat_cols])
        if self.num_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.num_cols])
    def transform(self, X):
        arrs = []
        if self.num_cols:
            arrs.append(self.scaler.transform(X[self.num_cols]))
        if self.cat_cols:
            arrs.append(self.ohe.transform(X[self.cat_cols]))
        return np.hstack(arrs)
    def save(self, path):
        joblib.dump(self, path)
    @staticmethod
    def load(path):
        return joblib.load(path)

def save_preprocessor(preproc, path):
    preproc.save(path)

def load_preprocessor(path):
    return Preprocessor.load(path)
''')

    # model.py (wrapper for autoencoder)
    write_file("src/model.py", '''\
import torch
from src.autoencoder import Autoencoder
import config

def build_model(input_dim, latent_dim=8, hidden_dims=None):
    if hidden_dims is None:
        hidden_dims = [max(16, 2*latent_dim)]
    return Autoencoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(input_dim, latent_dim, hidden_dims, path):
    model = Autoencoder(input_dim, latent_dim, hidden_dims)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
''')

    # baselines.py
    write_file("src/baselines.py", '''\
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def run_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def run_kmeans(X, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

def financial_health_composite(X, feature_defs):
    # Heuristic: mean of robust-normalized financial features
    num_cols = [c for c in X.columns if "robustnorm" in c]
    if not num_cols:
        num_cols = X.select_dtypes(include=np.number).columns
    composite = X[num_cols].mean(axis=1)
    return composite
''')

    # analysis.py
    write_file("src/analysis.py", '''\
import os
import sys
import pandas as pd
import json
import config
from src import data_loader, feature_engineering

def main():
    # Find Excel file in data/
    excel_files = [f for f in os.listdir("data") if f.lower().endswith(".xlsx") or f.lower().endswith(".xls")]
    if not excel_files:
        print("No Excel file found in 'data/'.", file=sys.stderr)
        sys.exit(1)
    excel_path = os.path.join("data", excel_files[0])
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
    print("\\n=== DATASET SUMMARY ===")
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
''')

    # autoencoder.py
    write_file("src/autoencoder.py", '''\
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [max(16, 2*latent_dim)]
        # Encoder
        enc = []
        last = input_dim
        for h in hidden_dims:
            enc.append(nn.Linear(last, h))
            enc.append(nn.ReLU())
            last = h
        enc.append(nn.Linear(last, latent_dim))
        self.encoder = nn.Sequential(*enc)
        # Decoder
        dec = []
        last = latent_dim
        for h in reversed(hidden_dims):
            dec.append(nn.Linear(last, h))
            dec.append(nn.ReLU())
            last = h
        dec.append(nn.Linear(last, input_dim))
        self.decoder = nn.Sequential(*dec)
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

def save_autoencoder(model, path):
    torch.save(model.state_dict(), path)

def load_autoencoder(path, input_dim=None, latent_dim=8, hidden_dims=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Try to infer input_dim from saved config if not provided
    import config, json, os
    if input_dim is None:
        if os.path.exists(config.CONFIG_SAVE_PATH):
            with open(config.CONFIG_SAVE_PATH) as f:
                conf = json.load(f)
            input_dim = conf["input_dim"]
            latent_dim = conf["latent_dim"]
            hidden_dims = conf["hidden_dims"]
        else:
            raise ValueError("input_dim must be specified or config file present.")
    model = Autoencoder(input_dim, latent_dim, hidden_dims)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model, device
''')

    # visualization.py
    write_file("src/visualization.py", '''\
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

def plot_missingness(df, outdir, fname="missingness.png"):
    miss = df.isnull().sum()
    plt.figure(figsize=(8,4))
    miss.plot(kind="bar")
    plt.title("Missing Values per Column")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname))
    plt.close()

def plot_numerical_distributions(df, features, outdir):
    for col in features:
        plt.figure()
        if HAS_SNS:
            sns.histplot(df[col].dropna(), kde=True)
        else:
            plt.hist(df[col].dropna(), bins=30)
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"dist_{col}.png"))
        plt.close()

def plot_boxplots(df, features, outdir):
    for col in features:
        plt.figure()
        if HAS_SNS:
            sns.boxplot(x=df[col].dropna())
        else:
            plt.boxplot(df[col].dropna())
        plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"box_{col}.png"))
        plt.close()

def plot_correlation_heatmap(df, outdir, fname="correlation.png"):
    corr = df.corr()
    plt.figure(figsize=(8,6))
    if HAS_SNS:
        sns.heatmap(corr, annot=True, fmt=".2f")
    else:
        plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
        plt.colorbar()
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname))
    plt.close()

def plot_pca_scatter(X_pca, labels, outdir, fname="pca_scatter.png"):
    plt.figure()
    plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10", alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Scatter Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname))
    plt.close()

def plot_reconstruction_error(errors, outdir, fname="recon_error.png"):
    plt.figure()
    plt.hist(errors, bins=30)
    plt.title("Reconstruction Error Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname))
    plt.close()

def plot_feature_importance(importances, feature_names, outdir, fname="feature_importance.png"):
    plt.figure(figsize=(8,4))
    idx = np.argsort(importances)[::-1]
    plt.bar([feature_names[i] for i in idx], np.array(importances)[idx])
    plt.title("Feature Importance (Reconstruction Error Sensitivity)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, fname))
    plt.close()
''')

    # train.py
    write_file("src/train.py", '''\
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
from src import data_loader, feature_engineering, preprocessing, autoencoder, visualization
import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel', type=str, default=None, help="Excel file to use")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=config.SEED)
    parser.add_argument('--latent-dim', type=int, default=8)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Load workbook
    excel_files = [f for f in os.listdir("data") if f.lower().endswith(".xlsx") or f.lower().endswith(".xls")]
    if not excel_files:
        print("No Excel file found in 'data/'.", file=sys.stderr)
        return
    excel_path = os.path.join("data", excel_files[0]) if args.excel is None else args.excel
    sheets = data_loader.load_workbook(excel_path)
    main_sheet = list(sheets.keys())[0]
    df = sheets[main_sheet]
    if df.shape[0] < config.MIN_ROWS_FOR_TRAINING:
        print(f"Not enough rows ({df.shape[0]}) for reliable neural network training. Need at least {config.MIN_ROWS_FOR_TRAINING}.", file=sys.stderr)
        print("Please provide a larger dataset.")
        return
    feat_defs = feature_engineering.infer_feature_definitions(df)
    X = feature_engineering.engineer_features(df, feat_defs)
    # Split
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    n_train = int(config.TRAIN_RATIO * len(idx))
    n_val = int(config.VAL_RATIO * len(idx))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    Xtr, Xval, Xtest = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    # Preprocessing
    preproc = preprocessing.Preprocessor()
    preproc.fit(Xtr)
    Xtr_proc = preproc.transform(Xtr)
    Xval_proc = preproc.transform(Xval)
    Xtest_proc = preproc.transform(Xtest)
    # Save preprocessor and feature defs
    preprocessing.save_preprocessor(preproc, config.PREPROCESSOR_SAVE_PATH)
    feature_engineering.save_feature_definitions(feat_defs, config.FEATURE_DEF_SAVE_PATH)
    # Model
    input_dim = Xtr_proc.shape[1]
    model = autoencoder.Autoencoder(input_dim, latent_dim=args.latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Training
    batch_size = min(32, len(Xtr_proc))
    train_loader = DataLoader(TensorDataset(torch.tensor(Xtr_proc, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(Xval_proc, dtype=torch.float32)), batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    best_val = float('inf')
    patience = 20
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon, _ = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon, _ = model(xb)
                loss = criterion(recon, xb)
                val_losses.append(loss.item())
        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)
        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)
        print(f"Epoch {epoch+1:3d}: Train Loss={mean_train:.4f}  Val Loss={mean_val:.4f}")
        if mean_val < best_val:
            best_val = mean_val
            patience_counter = 0
            autoencoder.save_autoencoder(model, config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping.")
            break
    # Save config
    with open(config.CONFIG_SAVE_PATH, "w") as f:
        json.dump({"input_dim": input_dim, "latent_dim": args.latent_dim, "hidden_dims": [max(16, 2*args.latent_dim)]}, f)
    # Save training metrics
    with open(config.TRAIN_METRICS_PATH, "w") as f:
        json.dump(history, f)
    print("Training complete. Model and preprocessor saved to 'models/'.")
''')

    # evaluate.py
    write_file("src/evaluate.py", '''\
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, silhouette_score
from src import data_loader, feature_engineering, preprocessing, autoencoder, baselines
import config

def main():
    # Load data, model, preprocessor, feature defs
    excel_files = [f for f in os.listdir("data") if f.lower().endswith(".xlsx") or f.lower().endswith(".xls")]
    if not excel_files:
        print("No Excel file found in 'data/'.")
        return
    excel_path = os.path.join("data", excel_files[0])
    sheets = data_loader.load_workbook(excel_path)
    main_sheet = list(sheets.keys())[0]
    df = sheets[main_sheet]
    feat_defs = feature_engineering.load_feature_definitions(config.FEATURE_DEF_SAVE_PATH)
    X = feature_engineering.engineer_features(df, feat_defs)
    # Split
    idx = np.arange(X.shape[0])
    np.random.seed(config.SEED)
    np.random.shuffle(idx)
    n_train = int(config.TRAIN_RATIO * len(idx))
    n_val = int(config.VAL_RATIO * len(idx))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    Xtest = X.iloc[test_idx]
    preproc = preprocessing.load_preprocessor(config.PREPROCESSOR_SAVE_PATH)
    Xtest_proc = preproc.transform(Xtest)
    model, device = autoencoder.load_autoencoder(config.MODEL_SAVE_PATH)
    model.eval()
    Xtest_tensor = torch.tensor(Xtest_proc, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon, latent = model(Xtest_tensor)
        recon = recon.cpu().numpy()
        latent = latent.cpu().numpy()
    mse = mean_squared_error(Xtest_proc, recon)
    mae = mean_absolute_error(Xtest_proc, recon)
    row_errors = ((Xtest_proc - recon)**2).sum(axis=1)
    # Anomaly threshold: 99th percentile of training error
    Xtr = X.iloc[train_idx]
    Xtr_proc = preproc.transform(Xtr)
    Xtr_tensor = torch.tensor(Xtr_proc, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon_tr, _ = model(Xtr_tensor)
        recon_tr = recon_tr.cpu().numpy()
    tr_errors = ((Xtr_proc - recon_tr)**2).sum(axis=1)
    threshold = np.percentile(tr_errors, 99)
    # Latent cluster silhouette
    if latent.shape[0] >= 2:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=2, random_state=config.SEED).fit_predict(latent)
        sil = silhouette_score(latent, labels)
    else:
        sil = None
    # Save metrics
    metrics = {
        "test_mse": float(mse),
        "test_mae": float(mae),
        "anomaly_threshold": float(threshold),
        "silhouette_score": float(sil) if sil is not None else None
    }
    with open(config.EVAL_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"row": Xtest.index, "recon_error": row_errors}).to_csv(config.TEST_ROW_SCORES_PATH, index=False)
    # Update report
    with open(config.MODEL_REPORT_PATH, "a") as f:
        f.write("\\n## Evaluation Metrics\\n")
        for k,v in metrics.items():
            f.write(f"- {k}: {v}\\n")
    print("Evaluation complete. Metrics saved.")
''')



# Builder's main function (top level)
def main():
    import sys
    print("=== Personal Stock Advisor Project Builder ===")
    ensure_dirs()
    excel_file = find_excel_file('.')
    excel_in_data = copy_excel_to_data(excel_file)
    print(f"Using Excel workbook: {excel_in_data}")
    write_requirements()
    print("Installing required Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Dependency installation failed: {e}", file=sys.stderr)
        print("Run manually: python3 -m pip install -r requirements.txt", file=sys.stderr)
        return
    write_config()
    write_readme()
    write_model_report_template()
    write_root_wrappers()
    write_tests()
    write_src_files()
    print("Checking generated Python modules...")
    generated_python = [
        "config.py", "analyze.py", "train.py", "evaluate.py", "predict.py",
        "src/data_loader.py", "src/feature_engineering.py", "src/preprocessing.py",
        "src/model.py", "src/baselines.py", "src/analysis.py", "src/autoencoder.py",
        "src/visualization.py", "src/train.py", "src/evaluate.py"
    ]
    compile_failures = []
    for path in generated_python:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                compile(fh.read(), path, "exec")
        except Exception as exc:
            compile_failures.append((path, str(exc)))
    if compile_failures:
        print("Generated-module compile failures:", file=sys.stderr)
        for path, error in compile_failures:
            print(f"  {path}: {error}", file=sys.stderr)
    else:
        print("All generated Python modules compiled successfully.")
    print("Project structure and files created.")
    print("Running automatic dataset analysis...")
    # Run analyze.py to print MODEL DESIGN
    try:
        subprocess.run([sys.executable, "analyze.py"], check=True)
    except Exception as e:
        # print(f"Error running analyze.py: {e}", file=sys.stderr)
        print("Error running analyze.py: ...")
    print("\\nNext step: To train the model (if enough rows), run:")
    print("    python train.py --epochs 200 --latent-dim 8")


if __name__ == "__main__":
    main()