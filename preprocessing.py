import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def preprocess_data(df, target_col, categorised):
    y_train = df[target_col]
    
    drop_cols = categorised['drop']
    x_cols = [col for col in df.columns if col not in drop_cols and col != target_col]
    x_train = df[x_cols].copy()
    
    for col in categorised['numerical']:
        if col in x_train.columns:
            median_val = x_train[col].median()
            x_train[col] = x_train[col].fillna(median_val)
            
    categorical = ['ohe', 'target_encode', 'text']
    for cat in categorical:
        for col in categorised[cat]:
            if col in x_train.columns:
                x_train[col] = x_train[col].fillna('missing_value')

    target_mean = y_train.mean()
    for col in categorised['target_encode']:
        if col in x_train.columns:
            group_y = y_train.groupby(x_train[col])
            cumsum = group_y.cumsum() - y_train
            cumcount = group_y.cumcount()
            x_train[col] = (cumsum + target_mean) / (cumcount + 1)
    
    for col in categorised['ohe']:
        if col in x_train.columns:
            unique_values = x_train[col].unique()
            for val in unique_values:
                x_train[f"{col}_{val}"] = np.where(x_train[col] == val, 1, 0)
            x_train = x_train.drop(columns=[col])
    
    for col in categorised['text']:
        text_ft = x_train[col]
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(text_ft)
        n_comp = min(10, tfidf_matrix.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_comp)
        svd_matrix = svd.fit_transform(tfidf_matrix)
        for i in range(n_comp):
            x_train[f"{col}_svd_{i}"] = svd_matrix[:, i]
        x_train = x_train.drop(columns=[col])

    return x_train, y_train
