
# coding: utf-8

# In[18]:

# imports
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# allow plots to appear directly in the notebook
get_ipython().magic(u'matplotlib inline')


# In[19]:

# read data into a DataFrame
data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data", delim_whitespace = True,
                   keep_default_na = True, na_values = ["?"])
data.columns = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin", "Car Name"]
print(data.shape)
data.head(10)


# In[20]:

data = data.dropna()
print(data.shape)


# In[21]:

# create a fitted model
lm1 = smf.ols(formula="MPG ~ Displacement", data=data).fit()

# print the coefficients
print(lm1.params)

# print the R-squared value for the model
print(lm1.rsquared)


# In[22]:

# create a fitted model
lm2 = smf.ols(formula="MPG ~ Horsepower", data=data).fit()

# print the coefficients
print(lm2.params)

# print the R-squared value for the model
print(lm2.rsquared)


# In[23]:

# create a fitted model
lm3 = smf.ols(formula="MPG ~ Weight", data=data).fit()

# print the coefficients
print(lm3.params)

# print the R-squared value for the model
print(lm3.rsquared)


# In[24]:

# create a fitted model
lm4 = smf.ols(formula="MPG ~ Acceleration", data=data).fit()

# print the coefficients
print(lm4.params)

# print the R-squared value for the model
print(lm4.rsquared)


# In[25]:

# create a fitted model with all four features
lm5 = smf.ols(formula="MPG ~ Displacement + Horsepower + Weight + Acceleration", data=data).fit()

# print the coefficients
print(lm5.params)

# print the R-squared value for the model
print(lm5.rsquared)


# In[26]:

# create a fitted model
lm6 = smf.ols(formula="MPG ~ Cylinders", data=data).fit()

# print the coefficients
print(lm6.params)

# print the R-squared value for the model
print(lm6.rsquared)


# In[27]:

# create a fitted model with all four features
lm7 = smf.ols(formula="MPG ~ Cylinders + Displacement + Horsepower + Weight + Acceleration", data=data).fit()

# print the coefficients
print(lm7.params)

# print the R-squared value for the model
print(lm7.rsquared)

