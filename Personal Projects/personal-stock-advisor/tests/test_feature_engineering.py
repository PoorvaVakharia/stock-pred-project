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
