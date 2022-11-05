#!/usr/bin/env python
# coding: utf-8

# # Smartphone Price model  using Linear Regression

# ## Credit 
# 1) The Dataset was downloaded from Kaggle *Mobile Phones Specifications and Prices in Kenya*  https://www.kaggle.com/datasets/lyraxvinns/mobile-phones-specifications-and-prices-in-kenya
# 

# # Step 1: Import the basic libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Import all the necessary libraries for Models
# We need to import the following libraries from sklearn
#       
#       --> LinearRegression
# 
#  Import the test split function for spliting the training data from the test data
#       --> train_test_split

# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ## Import pandas library for data manipulation

# In[3]:


import pandas as pd


# # Step 2: Load the dataset to the platform

# The Pandas library has an easy way to load in data, read_csv():

# In[4]:


phoneprice = pd.read_csv("PhonesPriceInKenya.csv")


# In order to view the data, how they are arranged we can use the head() function which shows the first 5 rows of data. To see the last five we use iris.head(-5)

# In[5]:


phoneprice.head()


# ## Step2: Preprocessing and Data Cleaning 
# This data has comma separator in the **Price(Kshs)** and % sign in the **Specs Score** which is not regognized as interger,rather the dataframe sees the entry as a string. We use DataFrame.replace with regex=True for substrings replacement
# 1)We need to remove all the commas from the **Price(Kshs)**  
# 
# 2)Remove the % sign from **Specs Score**

# In[6]:


phoneprice['Price(Kshs)'] = phoneprice['Price(Kshs)'].replace(',','', regex=True)
phoneprice['Specs Score'] = phoneprice['Specs Score'].replace('%','', regex=True)
phoneprice


# select all rows with NaN values in Pandas DataFrame  Using isnull() to select all rows with NaN under a single DataFrame column

# In[7]:


phoneprice[phoneprice['Price(Kshs)'].isnull()].count()


# Convert all the numeric series to float datatype. This is to avoid situations that the entries are read as string

# In[8]:


all_numeric_series = ['Price(Kshs)','Rating','Specs Score','Likes']
phoneprice[all_numeric_series] = phoneprice[all_numeric_series].astype(float)


# Find the mean of the enries and use it to fill the NAN entries

# In[9]:


mean_value_for_all = phoneprice[all_numeric_series].mean()
mean_value_for_all


# In[10]:


[phoneprice.fillna(value=mean_value_for_all, inplace=True)]
phoneprice.isna().any() # Check if there are any null values


# 

# Check the statistics of the dataset with the **describe()** function e.g the mean, max etc

# In[11]:


phoneprice.describe()


# 
# # Step 3: Define the inputs and outputs

# We now need to define the **features(inputs** and **labels(outputs)**. We can do this easily with pandas by slicing the data table and choosing certain rows/columns.
# 
# **input/features** are those characteristics of the data that determines the class that they belong e.g color, lenght, weight etc
# 
# **output** is the class that they belong e.g positive/negative, 0/1, hot/cold, present/absent etc
# 
# You can achieve this by naming all the column names within a double square parenthesis.
# 
# Another way to do this using **iloc()**
# 

# In[12]:


# Let's tell the dataframe which column we want for the imputs/features.  
X = phoneprice[['Rating','Specs Score','Likes']]

# Let's tell the dataframe which column we want for the target/labels/output.  
y = phoneprice['Price(Kshs)']


# In[13]:


inputs.head()


# # Step 4: Split the data to training and testing sets. 
# Remember the default percentage is 80% for training and 20% for testing but you can change the percentages using test_size

# Assign the split data into diffent arrays

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(inputs, output,test_size=0.2,random_state=42)


# # Step 5: Apply the model

# In[ ]:


regressor = LinearRegression()


# In[ ]:


# Fit the model to the training data 
regressor.fit(X_train, y_train) 


# In[ ]:


# Make the prediction
y_prediction = regressor.predict(X_test)


# # Step 6: Metrics and Scores
# 
# There are a number of metrics that can be used to measure the perfomance of a model depending on the type c.f https://scikit-learn.org/stable/modules/model_evaluation.html
# 
# Regression model can be measured using several tools, we are going to use only three namely:
#     
#     1)max_error
#     2)Root Mean Square Error(RMSE)
#     3)$r^2$ score: — the proportion of variance in y that can be explained by X
# 

# In[ ]:


from sklearn.metrics import mean_squared_error,max_error,r2_score


# In[ ]:


#confusion matrix
def prediction_metrics(y_test, y_pred, plt_title=None):
    # r2  test
    r_2 = r2_score(y_test, y_pred)
    #mean squared test
    mse = np.sqrt(mean_squared_error(y_test, y_pred))
    #max error test
    me = max_error(y_test, y_pred)
    return r_2, mse,me


# In[ ]:


scores=prediction_metrics(y_test, y_prediction)
print('r2 score | mean squared|  max error, ')
print([ '%.5f' % elem for elem in scores ])


# # Step 7: Get the statistics of the predicted values

# In[ ]:


#Convert the array to pandas series
prediction_pd = pd.Series(y_prediction)


# In[ ]:


prediction_pd.describe()


# # Step 8: Make the model interactive

# In[ ]:


Rating,SpecsScore,Likes = input('Enter the ratings, specs score and likes ').split(',')
Rating,SpecsScore,Likes = list(map(float, [Rating,SpecsScore,Likes]))


# In[ ]:


my_price = int(regressor.predict([[Rating,SpecsScore,Likes]]))
print(my_price)


# In[ ]:




