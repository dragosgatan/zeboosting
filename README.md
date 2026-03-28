# ZeBoosting
(short for Zero-effort Boosting) is a gradient boosting model that focuses on minimal user effort.

## Features
- Auto-detects task type (classification / regression)
- Automatically preprocesses data

## Requirements
- scikit-learn
- pandas
- numpy
```bash
pip install -r requirements.txt
```

## Usage
```python
from zeboosting import ZeBoosting
predictions = ZeBoosting(df_train, 'target', df_test)
```

## Installation
```bash
git clone https://github.com/dragosgatan/zeboosting.git
cd zeboosting
pip install -r requirements.txt
```


