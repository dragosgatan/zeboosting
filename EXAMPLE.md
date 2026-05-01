## Complete usage examples
### [SmartCargo - Optimizing deliveries in Romania](https://platform.olimpiada-ai.ro/ro/problems/75)
- This problem was featured at Romania's county stage of the National Olympiad of AI in 2025.
- It's a classic regression problem; the dataset includes categorical features.
```python
import pandas as pd; from zeboosting import ZeBoosting as ze
df_train = pd.read_csv('train_data.csv'); df_test = pd.read_csv('test_data.csv')

s1=pd.DataFrame({'subtaskID':[1],'datapointID':1, 'answer':len(df_test[(df_test['City A'] == 'Barlad') & (df_test['Weather'] == 'Fog')])})
s2=pd.DataFrame({'subtaskID':2,'datapointID':df_test['ID'],'answer':ze(df_train, 'deliver_time', df_test)})

pd.concat([s1,s2]).to_csv('submission.csv', index=False)
```
- This solution obtains 100p in just 5 lines of code.
- The  ML model part is only one line of code: `ze(df_train, 'deliver_time', df_test)`.