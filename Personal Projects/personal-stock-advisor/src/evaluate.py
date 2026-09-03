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
        f.write("\n## Evaluation Metrics\n")
        for k,v in metrics.items():
            f.write(f"- {k}: {v}\n")
    print("Evaluation complete. Metrics saved.")
