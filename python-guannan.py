#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_excel("intern_ml_inputs.xlsx",header=[1])

data.head()


# In[2]:


data=data.drop(columns=['date','Unnamed: 0'])


# In[3]:


#Handle the missing values
print('Number of instances = %d' % (data.shape[0]))
print('Number of attributes = %d' % (data.shape[1]))

print('Number of missing values:')
for col in data.columns:
    print('\t%s: %d' % (col,data[col].isna().sum()))


# In[4]:


print('Number of rows before discarding missing values = %d' % (data.shape[0]))
data=data.dropna()
print('Number of rows before discarding missing values = %d' % (data.shape[0]))


# In[5]:


#Start building the model
from sklearn import linear_model
import numpy as np
X=data.drop(columns=['sp_sed_avg_turb'])
Y=data['sp_sed_avg_turb']


# In[6]:


#Evaluate the performance of the model
#Divide the dataset into train set and test set
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# In[7]:


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The intercept
print('Intercept: \n', regr.intercept_)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))


# In[8]:


#Predict one single line of data:
# "0" means the first line of the data in the excel sheet.
predict = regr.predict(X.loc[0].values.reshape(-1,35))
print(predict[0])


# In[9]:


#Predict the whole data in the excel sheet
predict_all = regr.predict(X)
print(predict_all)

