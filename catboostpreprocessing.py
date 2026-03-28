import pandas as pd
import numpy as np

def preprocess_data_cat(df, target_col, categorised, test_df):
    y_train = df[target_col]
    y_train_encoded = y_train
    drop_cols = categorised['drop']
    x_cols = [col for col in df.columns if col not in drop_cols and col != target_col]
    x_train = df[x_cols].copy()
    x_test = test_df[x_cols].copy()

    cat_features = [col for col in categorised['ohe'] + categorised['target_encode'] if col in x_train.columns]
    text_features = [col for col in categorised['text'] if col in x_train.columns]

    for col in cat_features + text_features:
        x_train[col] = x_train[col].fillna('missing_value')
        x_test[col] = x_test[col].fillna('missing_value')

    return x_train, y_train, x_test, cat_features, text_features
