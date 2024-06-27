#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv('Housing Price.csv')
data


# # Getting basic information about Data

# In[6]:


data.head()


# In[7]:


data.tail()


# In[10]:


data.info()


# In[12]:


data.shape


# In[13]:


data.size


# In[14]:


data.columns


# In[15]:


data.describe()


# # Check missing values

# In[19]:


print(data.isnull().sum())


# #  Drop rows with NaN values

# In[67]:


data = data.dropna(subset=['bedrooms', 'square footage'])
data['price'] = data['bedrooms'].astype(int)


# # Check the unique values in the bedrooms and square_footage column 

# In[70]:


print(data['bedrooms'].unique())
print(data['square footage'].unique())


# #  Remove any non-numeric characters (if any) and convert to numeric, coercing errors to NaN

# In[69]:


data['bedrooms'] = pd.to_numeric(data['bedrooms'], errors='coerce')
data['square footage'] = pd.to_numeric(data['square footage'], errors='coerce')



# #  Split the data into training and testing sets

# In[54]:


X = data[['bedrooms', 'square footage']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Create a Linear Regression

# In[61]:


model = LinearRegression()
model.fit(X_train, y_train)





# #  Make predictions on the testing set

# In[60]:


y_pred = model.predict(X_test)


# # Calculate Mean Squared Error

# In[66]:


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')



# # Calculate R-squared

# In[64]:


r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')


# 
# # Scatter plot of actual vs predicted values

# In[59]:


plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()


# In[ ]:




