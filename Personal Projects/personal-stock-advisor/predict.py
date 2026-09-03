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
