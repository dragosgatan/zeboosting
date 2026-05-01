# ZeBoosting
(short for Zero-effort Boosting) is a gradient boosting model that focuses on minimal user effort.

## Features
- Auto-detects task type (classification / regression)
- Automatically preprocesses data

## Requirements
- scikit-learn
- pandas
- numpy
- catboost
Dependencies are installed automatically when you run `pip install .`.

## Usage
```python
from zeboosting import ZeBoosting
predictions = ZeBoosting(df_train, 'target', df_test)
```
### CatBoost version (more accurate)
```python
from zeboosting import ZeCatBoosting
predictions = ZeCatBoosting(df_train, 'target', df_test)
```
## Installation
```bash
git clone https://github.com/dragosgatan/zeboosting.git
cd zeboosting
pip install -e .
```

If you prefer a non-editable install:
```bash
pip install .
```


