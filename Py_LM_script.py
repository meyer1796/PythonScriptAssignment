#!/usr/bin/env python
# coding: utf-8

# # Python Linear Regression

# ## Here we read a CSV file to import the data

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import os

print("Hello")

if len(sys.argv) > 1:
    input_file = sys.argv[1]
    basename = os.path.basename(input_file)
    base, ext = os.path.splitext(basename)
else:
    print("Please input a file in the command terminal")
    sys.exit(-1)

# In[28]:
print()
print(f"loading{input_file}")
print()
df = pd.read_csv(input_file)


# In[29]:





# In[30]:

print("File loaded, now looking at the head of the file.")
print()
print(df['x'].head())
print()

# In[31]:


print(df['y'].head())
print()

# Here we plot the data

# In[32]:

print("Plotting Data")
print()
plt.scatter(df['x'], df['y'])
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(f"{base}_Raw_Scatter_Plot.png")


# In[33]:





# In[34]:





# In[35]:

print("Modeling Data")
print()
x=np.array(df['x']).reshape((-1,1))


# In[36]:


y=np.array(df['y'])


# Here we form the linear model, display its key details, and form predictions from it

# In[37]:


model = LinearRegression()


# In[38]:

print("Creating Fit Line")
print()
md=model.fit(x,y)


# In[39]:

print("Calculating r squared")
print()
print(f" r_sqd: {model.score(x,y)}")
print()

# In[40]:

print("Calculating Slope of Fit")
print()
print(f" Slope: {md.coef_}")
print()

# In[41]:

print("Calculating Y Intercept of Fit")
print()
print(f"y_int: {md.intercept_}")
print()

# In[42]:


yp = md.predict(x)


# In[43]:

print("Calculating Predicted Values")
print()
print(yp)
print()

# Here we plot the linear model, using the predicted y values, by itself and with the original scatter plot

# In[44]:


plt.plot(x, yp)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(f"{base}_Linear_Model.png")

# In[46]:


plt.scatter(x,y)
plt.plot(x,yp)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(f"{base}_Linear_Model_and_Scatter_Plot.png")


# In[ ]:




