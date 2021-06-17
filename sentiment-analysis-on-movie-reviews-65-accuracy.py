#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('C:/Users/sharm/Downloads/Review Inputs'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('C:/Users/sharm/Downloads/Review Inputs/train.tsv.zip',sep='\t')
test = pd.read_csv('C:/Users/sharm/Downloads/Review Inputs/test.tsv.zip',sep='\t')
train.head()
test.head()


# In[2]:


train.Sentiment.value_counts()
train.SentenceId.value_counts()
train.SentenceId.nunique()
train.shape
test.shape
train.info()
train['text_len'] = train['Phrase'].apply(len)
test['text_len'] = test['Phrase'].apply(len)
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
ps = PorterStemmer()

message = []

for i in range(0, train.shape[0]):
    
    review = re.sub('[^a-zA-Z]', ' ', train['Phrase'][i])
   
    review = review.lower()
    
    review = review.split()
    #
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
   
    review = ' '.join(review)
    message.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=100)
X = cv.fit_transform(message).toarray()
Y=np.array(train['Sentiment'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,criterion='gini',random_state=42)
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
ps1 = PorterStemmer()

message1 = []

for i in range(0, test.shape[0]):
    review1 = re.sub('[^a-zA-Z]', ' ', test['Phrase'][i])
    review1 = review1.lower()
    review1 = review1.split()
    #
    review1 = [ps1.stem(word) for word in review1 if not word in stopwords.words('english')]
    review1 = ' '.join(review1)
message1.append(review1)
cv1 = CountVectorizer(max_features=100)
X1 = cv.fit_transform(message1).toarray()
Y1 = rfc.predict(X1)
sub = pd.read_csv('C:/Users/sharm/Downloads/Review Inputs/train.tsv.zip')
sub.head()
submission = pd.DataFrame()
submission['PhraseId'] = test.PhraseId
submission['Sentiment'] = Y1
submission.to_csv('Random_Forest_V1.csv', index=False)


# In[ ]:




