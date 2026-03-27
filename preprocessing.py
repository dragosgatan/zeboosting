import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
def preprocess_data(df, target_col, categorised, test_df=None):
    y_train = df[target_col]
    y_train_encoded = y_train
    if categorised.get('target_categorical'):
        y_train_encoded = pd.Series(pd.factorize(y_train)[0], index=y_train.index)

    drop_cols = categorised['drop']
    x_cols = [col for col in df.columns if col not in drop_cols and col != target_col]
    x_train = df[x_cols].copy()
    x_test = test_df[x_cols].copy() if test_df is not None else None

    for col in categorised['numerical']:
        if col in x_train.columns:
            median_val = x_train[col].median()
            x_train[col] = x_train[col].fillna(median_val)
            if x_test is not None:
                x_test[col] = x_test[col].fillna(median_val)
            
    categorical = ['ohe', 'target_encode', 'text']
    for cat in categorical:
        for col in categorised[cat]:
            if col in x_train.columns:
                x_train[col] = x_train[col].fillna('missing_value')
                if x_test is not None:
                    x_test[col] = x_test[col].fillna('missing_value')

    target_mean = y_train_encoded.mean()
    for col in categorised['target_encode']:
        if col in x_train.columns:
            if x_test is not None:
                stats = y_train_encoded.groupby(x_train[col]).agg(['sum', 'count'])
                lookup = (stats['sum'] + target_mean) / (stats['count'] + 1)
                x_test[col] = x_test[col].map(lookup).fillna(target_mean)
            
            group_y = y_train_encoded.groupby(x_train[col])
            x_train[col] = (group_y.cumsum() - y_train_encoded + target_mean) / (group_y.cumcount() + 1)
    
    for col in categorised['ohe']:
        if col in x_train.columns:
            unique_values = x_train[col].unique()
            for val in unique_values:
                col_name = f"{col}_{val}"
                x_train[col_name] = np.where(x_train[col] == val, 1, 0)
                if x_test is not None:
                    x_test[col_name] = np.where(x_test[col] == val, 1, 0)
            x_train = x_train.drop(columns=[col])
            if x_test is not None:
                x_test = x_test.drop(columns=[col])
    
    for col in categorised['text']:
        if col in x_train.columns:
            tfidf = TfidfVectorizer()
            tfidf_train = tfidf.fit_transform(x_train[col])
            n_comp = min(10, tfidf_train.shape[1] - 1)
            svd = TruncatedSVD(n_components=n_comp)
            svd_train = svd.fit_transform(tfidf_train)
            
            for i in range(n_comp):
                x_train[f"{col}_svd_{i}"] = svd_train[:, i]
                if x_test is not None:
                    tfidf_test = tfidf.transform(x_test[col])
                    svd_test = svd.transform(tfidf_test)
                    x_test[f"{col}_svd_{i}"] = svd_test[:, i]
            
            x_train = x_train.drop(columns=[col])
            if x_test is not None:
                x_test = x_test.drop(columns=[col])

    return (x_train, y_train, x_test) if x_test is not None else (x_train, y_train)
