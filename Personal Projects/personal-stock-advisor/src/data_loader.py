import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import skew

def load_workbook(path):
    print("7. **** path =", path)
    path = "data/additional_stocks_combinations.xlsx"
    print("8. **** new path = ", path)
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
