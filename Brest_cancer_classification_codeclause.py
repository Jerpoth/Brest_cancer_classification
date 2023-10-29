#!/usr/bin/env python
# coding: utf-8

# Importing the Dependencies

# In[4]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[6]:


# Data collection and preprocessing
breast_cancer_dataset = load_breast_cancer()
print(breast_cancer_dataset)


# In[9]:


# loading the data to dataframe

data_frame=pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names)
data_frame.head()


# In[10]:


# Adding the traget column

data_frame['label']=breast_cancer_dataset.target
data_frame.tail()


# In[11]:


#number of Row and Column
data_frame.shape


# In[12]:


#Information about the Data
data_frame.info()


# In[13]:


#checking for the missing value
data_frame.isnull().sum()


# In[14]:


# statistical measure
data_frame.describe()


# In[18]:


# checking the distribution of target variable
data_frame['label'].value_counts()


# In[19]:


data_frame.groupby('label').mean()


# In[20]:


# separating the feature and target

x=data_frame.drop(columns='label',axis=1)
y=data_frame['label']


# In[21]:


print(x)


# In[22]:


print(y)


# In[24]:


#splitting the data into training data and testing data

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[25]:


print(x.shape,x_train.shape,x_test.shape)


# Model Training

# In[30]:


model=LogisticRegression()


# In[31]:


model.fit(x_train,y_train)


# In[32]:


# Accuracy_score on training data

x_train_pred=model.predict(x_train)


# In[33]:


training_data_acc=accuracy_score(y_train,x_train_pred)


# In[34]:


print('Accuracy score of training Data is =',training_data_acc)


# In[35]:


# Accuracy_score on testing data

x_test_pred=model.predict(x_test)


# In[36]:


testing_data_acc=accuracy_score(y_test,x_test_pred)


# In[37]:


print('Accuracy score of testing Data is =',testing_data_acc)


# # Building a Predicitive system
# 

# In[38]:


input_data=(17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)
#input data into numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the numpy array for one data_point
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


# In[41]:


prediction=model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print('The Breast Cancer is Malignant')
else:
    print('The Breast Cancer is Benign')


# In[ ]:




