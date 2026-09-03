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
