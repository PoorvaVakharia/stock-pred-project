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
