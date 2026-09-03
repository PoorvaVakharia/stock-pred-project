import numpy as np
import pandas as pd
from src import preprocessing

def test_preprocessing_shapes():
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4],
        "cat": ["a", "b", "a", "c"]
    })
    preproc = preprocessing.Preprocessor()
    preproc.fit(df)
    Xtr = preproc.transform(df)
    assert Xtr.shape[0] == 4
    assert Xtr.shape[1] >= 2
