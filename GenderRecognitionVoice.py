#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
#For confusion matrixes
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[54]:


df = pd.read_csv('/Users/sadegh/Desktop/DataSet GitHub/Logostic Regression/gender_voice_weka_dataset.csv')


# In[55]:


df.head()


# In[56]:


df.info()


# In[57]:


df.corr()


# In[58]:


df.label = [1 if each == "female" else 0 for each in df.label]
#We assign 1 to female, 0 to male.


# In[59]:


df.head()


# In[60]:


#corelation matrix.
cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)


# In[52]:


g = sns.PairGrid(df[['meanfreq','sd','median','Q25','IQR','sp.ent','sfm','meanfun','label']], hue = "label")
g = g.map(plt.scatter).add_legend()


# In[19]:


X = df.iloc[:, :-1].values


# In[20]:


Y = df.iloc[:,20].values


# In[23]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test =train_test_split(X,Y,random_state = 0,test_size = 0.25)


# In[26]:


from sklearn.preprocessing import StandardScaler


# In[27]:


sc_X = StandardScaler()


# In[28]:


X_train = sc_X.fit_transform(X_train)


# In[29]:


X_test = sc_X.transform(X_test)


# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


classifier = LogisticRegression(random_state=0)


# In[32]:


classifier.fit(X_train, y_train)


# In[33]:


y_pred = classifier.predict(X_test)


# In[34]:


cm = confusion_matrix(y_test,y_pred)


# In[35]:


cm


# In[43]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[39]:


from sklearn.metrics import classification_report


# In[40]:


print(classification_report(y_test,y_pred))


# In[47]:


y_pred_proba = classifier.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:




