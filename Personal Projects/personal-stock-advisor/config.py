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
