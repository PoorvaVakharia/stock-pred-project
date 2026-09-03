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
