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
