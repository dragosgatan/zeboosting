import pandas as pd 
import numpy as np 

def categorise_features(df, target_col, ohe_limit = 5, word_threshold = 5):
    categorised = {
        'task': 'regression',
        'target_categorical': False,
        'drop': [],
        'numerical': [],
        'ohe': [],
        'target_encode': [],
        'text': []
    }
    rows = len(df)
    object_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    #handling target 
    if target_col in object_cols:
        categorised['task'] = 'classification'
        categorised['target_categorical'] = True
    elif not np.issubdtype(df[target_col].dtype, np.floating):
        if df[target_col].nunique() / len(df) < 0.10 and df[target_col].nunique() <= 10: #ik 10% is kind of a lot 
            categorised['task'] = 'classification'
    for col in df.columns:
        if col == target_col:
            continue

        nunique = df[col].nunique()
        if nunique == rows:
            if col in object_cols:
                word_counts = df[col].astype(str).str.split().str.len()
                #if more than half the values exceed the word threshold, it's most likely a nlp column
                if (word_counts > word_threshold).mean() > 0.5: 
                    categorised['text'].append(col)
                #else, the column is probably an id column or noise
                else:
                    categorised['drop'].append(col)
            else:
                categorised['drop'].append(col)
        #noise
        elif nunique == 1:
            categorised['drop'].append(col)

        elif col in object_cols:
            if nunique <= ohe_limit:
                categorised['ohe'].append(col)
            else:
                categorised['target_encode'].append(col)
        else:
            categorised['numerical'].append(col)
            
    return categorised
