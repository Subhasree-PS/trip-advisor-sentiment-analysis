#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore',category=UserWarning)


# In[2]:


pd.set_option('display.max_rows',5000000)
pd.set_option('display.max_columns',5000000)


# In[3]:


import re
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


# In[4]:


df=pd.read_csv('tripadvisor_hotel_reviews.csv')


# In[5]:


df.shape


# In[6]:


df.head()


# In[38]:


df['Rating'].value_counts()


# In[7]:


corpus=[]

for i in range(0, 20491):
  review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)


# In[8]:


corpus


# # Data transformation

# In[9]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1420)


# In[10]:


X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values


# In[11]:


import pickle
bow_path = 'c1_BoW_Sentiment_Model.pkl'
pickle.dump(cv, open(bow_path, "wb"))


# # Training set and testing set

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# # Model fitting (Naive Bayes)

# In[13]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[14]:


# Exporting NB Classifier to later use in prediction
import joblib
joblib.dump(classifier, 'c2_Classifier_Sentiment_Model') 


# In[15]:


y_pred = classifier.predict(X_test)


# # Model perfomance

# In[29]:


y_pred = classifier.predict(X_test)


# In[30]:


y_pred.shape


# In[17]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[18]:


accuracy_score(y_test, y_pred)


# In[19]:


from sklearn.metrics import precision_score, recall_score, f1_score


# In[31]:


precision = precision_score(y_test, y_pred, average='macro')


# In[32]:


precision


# In[33]:


recall = recall_score(y_test, y_pred, average='macro')
recall


# In[34]:


f1 = f1_score(y_test, y_pred, average='macro')
f1


# In[21]:


import joblib
classifier = joblib.load('c2_Classifier_Sentiment_Model')


# In[22]:


# Loading BoW dictionary
from sklearn.feature_extraction.text import CountVectorizer
import pickle
cvFile='c1_BoW_Sentiment_Model.pkl'
# cv = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open('./drive/MyDrive/Colab Notebooks/2 Sentiment Analysis (Basic)/3.1 BoW_Sentiment Model.pkl', "rb")))
cv = pickle.load(open(cvFile, "rb"))


# In[23]:


X_fresh = cv.transform(corpus).toarray()
X_fresh.shape


# In[24]:


y_pred = classifier.predict(X_fresh)
print(y_pred)


# In[25]:


df['predicted_label'] = y_pred.tolist()


# In[35]:


class_counts = np.bincount(y_pred)


# In[36]:


class_counts


# In[37]:


import matplotlib.pyplot as plt

labels = ['Poor', 'Below Average','Fair','Average', 'Good', 'Excellent']
plt.pie(class_counts, labels=labels, autopct='%1.1f%%')
plt.title('Predicted Class Distribution')
plt.axis('equal')
plt.show()


# In[ ]:




