#!/usr/bin/env python
# coding: utf-8

# # Step 1 : Load Dataset

# In[3]:


import pandas as pd


# In[10]:


dataset = pd.read_csv("data.csv")


# In[14]:


dataset.head(8)


# In[15]:


X = dataset[["Age", "EstimatedSalary"]]


# In[16]:


y= dataset["Purchased"]


# # Step 2 : Graph

# In[17]:


import seaborn as sns 


# In[20]:


sns.set()


# In[37]:


sns.scatterplot(x='Age', y="EstimatedSalary", data = dataset, hue="Purchased",palette="Set1")


# # Splitting Data into Test data and Train data

# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Step 4 : Create Model

# In[41]:


from sklearn.neighbors import KNeighborsClassifier


# In[44]:


model = KNeighborsClassifier(n_neighbors = 5)


# In[49]:


model.fit(X_train, y_train)


# In[50]:


y_pred = model.predict(X_test)


# # Step 5 : Accuracy 

# In[48]:


from sklearn.metrics import accuracy_score


# In[52]:


Accuracy = accuracy_score(y_test, y_pred)


# In[57]:


print("Accuracy = {} %".format(Accuracy*100))


# # Prediction

# In[72]:


result = []


# In[81]:


result = [model.predict([[20,20000]]), model.predict([[60, 1000000]])]


# In[92]:


for i in range(len(result)):
    if(result[i][0]==0):
        print("Not Purchased")
    else:
        print("Purchased")


# In[ ]:




