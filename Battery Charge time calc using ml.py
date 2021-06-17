#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from sklearn.linear_model import LinearRegression


timeCharged = float(input())
file = 'C:/Users/sharm/Downloads/trainingdata.txt'
data = pd.read_csv(file, names=['charged', 'lasted'])
train = data[data['lasted'] < 8]
model = LinearRegression()
model.fit(train['charged'].values.reshape(-1, 1), train['lasted'].values.reshape(-1, 1))
ans = model.predict([[timeCharged]])
print(min(ans[0][0], 8))


# In[ ]:




