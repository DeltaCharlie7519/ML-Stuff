#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dataset = pd.read_csv("C:/Users/sharm/Downloads/Review Inputs/Inputs/weight-height.csv")
dataset.head()


# In[3]:


dataset.isnull().sum()


# In[4]:


x1 = dataset.iloc[:, 0].values
y1 = dataset.iloc[:, 2].values
plt.scatter(x1,y1,label='Gender',color='Green',s=50)
plt.xlabel('Gender')
plt.ylabel('Weight')
plt.title('Gender vs Weight')
plt.legend()
plt.show()


# In[5]:


x2 = dataset.iloc[:, 1].values
y2 = dataset.iloc[:, 2].values
plt.scatter(x2,y2,label = 'Height',color = 'Orange',s = 50)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Height vs Weight')
plt.legend(loc = "lower right")
plt.show()


# In[6]:


X = dataset.iloc[:, 1:2].values
print(X)


# In[7]:


y = dataset.iloc[:, 2].values
print(y)


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[9]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[10]:


y_pred = regressor.predict(X_test)


# In[14]:


plt.scatter(X_train, y_train, color = 'Yellow')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Height vs Weight (Training set)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()


# In[15]:


plt.scatter(X_test, y_test, color = 'Yellow')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Height vs Weight (Test set)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()


# In[24]:


y_pred = regressor.predict(X_test)
print('Coefficients: ', regressor.coef_)

print("Mean squared error: %.2f" % np.mean((regressor.predict(X_test) - y_test)**2 ))

print('Variance score: %.2f' % regressor.score(X_test, y_test))
print('Accuracy: %d '% (regressor.score(X_test, y_test)*100))


# In[ ]:




