from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from .preprocessing import preprocess_data
from .categorise import categorise_features
from .parameters import get_params
import pandas as pd

def model_hgbr(df, target_col, test_df, verbose=False):
    categorised = categorise_features(df, target_col)
    params = get_params(df)
    params['verbose'] = 1 if verbose else 0
    
    x_train, y_train, x_test = preprocess_data(df, target_col, categorised, test_df)
    
    if categorised['task'] == 'classification':
        model = HistGradientBoostingClassifier(**params)
    else:
        model = HistGradientBoostingRegressor(**params)
        
    model.fit(x_train, y_train)
    return model.predict(x_test)
