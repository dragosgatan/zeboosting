from catboost import CatBoostRegressor, CatBoostClassifier
from .categorise import categorise_features
from .catboostpreprocessing import preprocess_data_cat
import pandas as pd

def model_cat(df, target_col, test_df, verbose=0):
    categorised = categorise_features(df, target_col)
    x_train, y_train, x_test, cat_features, text_features = preprocess_data_cat(df, target_col, categorised, test_df)
    if categorised['task'] == 'classification':
        model = CatBoostClassifier(cat_features=cat_features, text_features=text_features, random_seed=42,verbose=verbose)
    else:
        model = CatBoostRegressor(cat_features=cat_features, text_features=text_features, random_seed=42,verbose=verbose)
    model.fit(x_train, y_train)
    return model.predict(x_test)
