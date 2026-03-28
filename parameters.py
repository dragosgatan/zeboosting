def get_params(df):
    len_df = len(df)
    #temporary way of determining lr, will change eventually
    if len_df < 1000:
        lr = 0.1
    elif len_df < 5000:
        lr = 0.05
    elif len_df < 10000:
        lr = 0.02
    else:
        lr = 0.01
    params = {
        'learning_rate': lr,
        'max_iter': 1000,
        'early_stopping': True,
        'n_iter_no_change': 30,
        'random_state': 42
    }
    return params
