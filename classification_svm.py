#!/usr/bin/env python
# coding: utf-8

# In[128]:


import numpy as np
import pandas as pd
import io
import requests
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
import os
from pandas.api.types import CategoricalDtype

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[134]:


columns = ["age", "workClass", "fnlwgt", "education", "education-num","marital-status", "occupation", "relationship",
          "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
train_data = pd.read_csv('adult.data', names=columns, 
             sep=' *, *', na_values='?')
test_data  = pd.read_csv('adult.test', names=columns, 
             sep=' *, *', skiprows=1, na_values='?')


# In[ ]:





# In[135]:


train_data.head()


# In[136]:


train_data.info()


# In[137]:


num_attributes = train_data.select_dtypes(include=['int'])
print(num_attributes.columns)


# In[138]:


num_attributes.hist(figsize=(10,10))


# In[139]:


train_data.describe()


# In[140]:


cat_attributes = train_data.select_dtypes(include=['object'])
print(cat_attributes.columns)


# In[141]:


sns.countplot(y='workClass', hue='income', data = cat_attributes)


# In[142]:


sns.countplot(y='occupation', hue='income', data = cat_attributes)


# In[143]:


class ColumnsSelector(BaseEstimator, TransformerMixin):
  
  def __init__(self, type):
    self.type = type
  
  def fit(self, X, y=None):
    return self
  
  def transform(self,X):
    return X.select_dtypes(include=[self.type])


# In[144]:


num_pipeline = Pipeline(steps=[
    ("num_attr_selector", ColumnsSelector(type='int')),
    ("scaler", StandardScaler())
])


# In[145]:


class CategoricalImputer(BaseEstimator, TransformerMixin):
  
  def __init__(self, columns = None, strategy='most_frequent'):
    self.columns = columns
    self.strategy = strategy
    
    
  def fit(self,X, y=None):
    if self.columns is None:
      self.columns = X.columns
    
    if self.strategy is 'most_frequent':
      self.fill = {column: X[column].value_counts().index[0] for 
        column in self.columns}
    else:
      self.fill ={column: '0' for column in self.columns}
      
    return self
      
  def transform(self,X):
    X_copy = X.copy()
    for column in self.columns:
      X_copy[column] = X_copy[column].fillna(self.fill[column])
    return X_copy


# In[146]:


class CategoricalEncoder(BaseEstimator, TransformerMixin):
  
  def __init__(self, dropFirst=True):
    self.categories=dict()
    self.dropFirst=dropFirst
    
  def fit(self, X, y=None):
    join_df = pd.concat([train_data, test_data])
    join_df = join_df.select_dtypes(include=['object'])
    for column in join_df.columns:
      self.categories[column] = join_df[column].value_counts().index.tolist()
    return self
    
  def transform(self, X):
    X_copy = X.copy()
    X_copy = X_copy.select_dtypes(include=['object'])
    for column in X_copy.columns:
      X_copy[column] = X_copy[column].astype({column:CategoricalDtype(self.categories[column])})
    return pd.get_dummies(X_copy, drop_first=self.dropFirst)


# In[147]:


cat_pipeline = Pipeline(steps=[
    ("cat_attr_selector", ColumnsSelector(type='object')),
    ("cat_imputer", CategoricalImputer(columns=
          ['workClass','occupation', 'native-country'])),
    ("encoder", CategoricalEncoder(dropFirst=True))
])


# In[148]:


full_pipeline = FeatureUnion([("num_pipe", num_pipeline), 
                ("cat_pipeline", cat_pipeline)])


# In[149]:


train_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)
test_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)


# In[150]:


train_copy = train_data.copy()
train_copy["income"] = train_copy["income"].apply(lambda x:0 if 
                        x=='<=50K' else 1)
X_train = train_copy.drop('income', axis =1)
Y_train = train_copy['income']


# In[151]:


X_train_processed=full_pipeline.fit_transform(X_train)
model = LogisticRegression(random_state=0)
model.fit(X_train_processed, Y_train)


# In[153]:


test_copy = test_data.copy()
test_copy["income"] = test_copy["income"].apply(lambda x:0 if 
                      x=='<=50K.' else 1)
X_test = test_copy.drop('income', axis =1)
Y_test = test_copy['income']


# In[154]:


X_test_processed = full_pipeline.fit_transform(X_test)
predicted_classes = model.predict(X_test_processed)


# In[155]:


accuracy_score(predicted_classes, Y_test.values)


# confuse matrix

# In[156]:


cfm = confusion_matrix(predicted_classes, Y_test.values)
sns.heatmap(cfm, annot=True)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')


# In[163]:


cross_val_model = LogisticRegression(random_state=0)
scores = cross_val_score(cross_val_model, X_train_processed,Y_train, cv=5)
print(np.mean(scores))


# In[160]:


penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
random_state=[0]
hyperparameters = dict(C=C, penalty=penalty,random_state=random_state)


# In[161]:


clf = GridSearchCV(estimator = model, param_grid = hyperparameters, 
                   cv=5)
best_model = clf.fit(X_train_processed, Y_train)
print('Best Penalty:', best_model.best_estimator_.get_params() ['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])


# In[164]:


best_predicted_values = best_model.predict(X_test_processed)
accuracy_score(best_predicted_values, Y_test.values)


# In[181]:


filename = 'final_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[182]:


saved_model = pickle.load(open(filename, 'rb')) 
print(saved_model)


# In[185]:


y_pred = model.predict(X_test_processed)


# In[187]:


accuracy = accuracy_score(Y_test,y_pred)
accuracy


# In[189]:


# Confusion Matrix
conf_mat = confusion_matrix(Y_test,y_pred)
conf_mat


# In[190]:


true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# In[192]:


#  formula breaking down for accuracy
Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy 


# In[193]:


# Precison
Precision = true_positive/(true_positive+false_positive)
Precision


# In[194]:


# Recall
Recall = true_positive/(true_positive+false_negative)
Recall


# In[195]:


# F1 Score
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score


# In[197]:


# Area Under Curve
auc = roc_auc_score(Y_test, y_pred)
auc


# ROC

# In[198]:


fpr, tpr, thresholds = roc_curve(Y_test, y_pred)


# In[200]:


plt.plot(fpr, tpr, color='green', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# SVM karnal

# In[230]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train_processed, Y_train)


# In[232]:


y_pred = svclassifier.predict(X_test_processed)


# In[235]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


# In[234]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train_processed, Y_train)


# In[236]:


y_pred = svclassifier.predict(X_test_processed)


# In[237]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


# In[238]:


from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train_processed, Y_train)


# In[239]:


y_pred = svclassifier.predict(X_test_processed)


# In[240]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))


# In[ ]:




