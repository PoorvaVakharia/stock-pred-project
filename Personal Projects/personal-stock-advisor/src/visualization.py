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
