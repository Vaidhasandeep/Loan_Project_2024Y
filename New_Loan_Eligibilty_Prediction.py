#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('C:\\Users\\v.omsai\\Downloads\\LoanData.csv')
df.head()


# In[3]:


df.drop('Loan_ID',axis=1,inplace=True)


# In[4]:


df['Gender'].unique()


# In[5]:


df['Married'].value_counts()


# In[6]:


df['Dependents'].value_counts()


# In[7]:


df['Education'].value_counts()


# In[8]:


df['Self_Employed'].value_counts()


# In[9]:


df['ApplicantIncome'].value_counts()


# In[10]:


df['CoapplicantIncome'].value_counts()


# In[11]:


df['LoanAmount'].value_counts()


# In[12]:


df['Loan_Amount_Term'].value_counts()


# In[13]:


df['Credit_History'].value_counts()


# In[14]:


df['Property_Area'].value_counts()


# In[15]:


df['Loan_Status'].value_counts()


# In[16]:


df['Credit_History'] = df['Credit_History'].replace({1.0:'good',0.0:'bad'})


# In[17]:


df['Dependents'] = df['Dependents'].replace({'3+':3})


# In[18]:


continuos = ['ApplicantIncome','CoapplicantIncome','LoanAmount']

Discret_Count = ['Dependents','Loan_Amount_Term','Credit_History']

Discret_Catogorical = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status'] 


# In[19]:


df[continuos].describe()


# In[20]:


plt.rcParams['figure.figsize'] = (18,8)

plt.subplot(1,3,1)
sns.histplot(df['ApplicantIncome'],kde=True)

plt.subplot(1,3,2)
sns.histplot(df['CoapplicantIncome'],kde=True)

plt.subplot(1,3,3)
sns.histplot(df['LoanAmount'],kde=True)
plt.show()


# In[21]:


df[continuos].skew()


# In[22]:


sns.pairplot(df[continuos])


# In[23]:


sns.heatmap(df[continuos].corr(),annot=True)


# In[24]:


plt.subplot(1,5,1)
sns.boxplot(df['ApplicantIncome'])
plt.show()

plt.subplot(1,5,2)
sns.boxplot(df['CoapplicantIncome'])

plt.subplot(1,5,3)
sns.boxplot(df['LoanAmount'])
plt.show()


# In[25]:


df[Discret_Catogorical].describe()


# In[26]:


plt.rcParams['figure.figsize'] = (18,8)

plt.subplot(2,3,1)
sns.countplot(x = df['Gender'],width = 0.5)

plt.subplot(2,3,2)
sns.countplot(x = df['Married'],width = 0.5)

plt.subplot(2,3,3)
sns.countplot(x = df['Education'],width = 0.5)

plt.subplot(2,3,4)
sns.countplot(x = df['Self_Employed'],width = 0.5)

plt.subplot(2,3,5)
sns.countplot(x = df['Property_Area'],width = 0.6)

plt.subplot(2,3,6)
sns.countplot(x = df['Loan_Status'],width = 0.5)
plt.show()


# In[27]:


df['Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']

df = df.drop(columns = ['ApplicantIncome','CoapplicantIncome'])


# In[28]:


df.isnull().sum()/len(df)*100


# In[29]:


df = df.dropna(subset=['Income','LoanAmount','Loan_Amount_Term','Credit_History'])


# In[30]:


df['Dependents'] = df['Dependents'].fillna(0)


# In[31]:


df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])


# In[32]:


df['Gender'].unique()


# In[33]:


df.isnull().sum()


# In[34]:


df['Property_Area'].value_counts()


# In[35]:


df['Gender'] = df['Gender'].map({'Male':1,'Female':0}).astype('int')
df['Married'] = df['Married'].map({'Yes':1,'No':0}).astype('int')
df['Education'] = df['Education'].map({'Graduate':1,'Not Graduate':0}).astype('int')
df['Self_Employed'] = df['Self_Employed'].map({'Yes':1,'No':0}).astype('int')
df['Credit_History'] = df['Credit_History'].map({'good':1,'bad':0}).astype('int')
df['Property_Area'] = df['Property_Area'].map({'Rural':0,'Semiurban':1,'Urban':2}).astype('int')
df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0}).astype('int')


# In[36]:


df['Dependents'] = df['Dependents'].astype('int')


# In[37]:


df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('int')


# In[38]:


df[['Income','LoanAmount']].skew()


# In[39]:


from scipy.stats import boxcox
df['Income'],a = boxcox(df['Income'])
df['LoanAmount'],c = boxcox(df['LoanAmount'])


# In[40]:


df[['Income','LoanAmount']].skew()


# In[41]:


df['Loan_Amount_Term'] = df['Loan_Amount_Term']/12


# In[42]:


X = df.drop('Loan_Status',axis=1)
y = df['Loan_Status']


# In[43]:


Train=[]
Test=[]
cv=[]
for i in range(0,101):
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=i)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train,y_train)

    ypred_train = model.predict(X_train)
    ypred_test = model.predict(X_test)

    from sklearn.metrics import accuracy_score
    Train.append(accuracy_score(y_train,ypred_train))
    Test.append(accuracy_score(y_test,ypred_test))

    from sklearn.model_selection import cross_val_score
    cv.append(cross_val_score(model,X_train,y_train,cv=5,scoring='accuracy').mean())

em = pd.DataFrame({'Train':Train,'Test':Test,'CV':cv})
gm = em[(abs(em['Train']- em['Test'])<=0.05) & (abs(em['Test']- em['CV'])<=0.05)]
rs = gm[gm['CV']==gm['CV'].max()].index.to_list()[0]
print('Random_State',rs)


# In[44]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=70)


# In[45]:


from sklearn.linear_model import LogisticRegression
model_1 = LogisticRegression()
model_1.fit(X_train,y_train)

ypred_train = model_1.predict(X_train)
ypred_test = model_1.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,ypred_train))
print(accuracy_score(y_test,ypred_test))

from sklearn.model_selection import cross_val_score
print(cross_val_score(model_1,X_train,y_train,cv=5,scoring='accuracy').mean())


# In[46]:


from sklearn.neighbors import KNeighborsClassifier

estimator = KNeighborsClassifier()
param_grid = {'n_neighbors' :list(range(1,50))}

from sklearn.model_selection import GridSearchCV
knn_grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')
knn_grid.fit(X_train,y_train)

knn_model = knn_grid.best_estimator_

ypred_train = knn_model.predict(X_train)
ypred_test = knn_model.predict(X_test)

print(accuracy_score(y_train,ypred_train))
print(accuracy_score(y_test,ypred_test))
print(cross_val_score(knn_model,X_train,y_train,cv=5,scoring='accuracy').mean())


# In[47]:


from sklearn.svm import SVC

estimator = SVC()
param_grid = {'C':[0.01,0.1,1], 'kernel':['linear','rbf','sigmoid','poly']}

from sklearn.model_selection import GridSearchCV
svm_grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')
svm_grid.fit(X_train,y_train)

svm_model = svm_grid.best_estimator_

ypred_train = svm_model.predict(X_train)
ypred_test = svm_model.predict(X_test)

print(accuracy_score(y_train,ypred_train))
print(accuracy_score(y_test,ypred_test))
print(cross_val_score(svm_model,X_train,y_train,cv=5,scoring='accuracy').mean())


# In[52]:


from sklearn.tree import DecisionTreeClassifier

estimator = DecisionTreeClassifier()
param_grid = {'criterion':['gini','entropy'],
              'max_depth':list(range(1,16))}

from sklearn.model_selection import GridSearchCV
tree_grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')
tree_grid.fit(X_train,y_train)

tree_model = tree_grid.best_estimator_
tree_features = tree_model.feature_importances_

index = [i for i,x in enumerate(tree_features)if x>0]

X_train_dt = X_train.iloc[:,index]
X_test_dt = X_test.iloc[:,index]

tree_model.fit(X_train_dt,y_train)

ypred_train = tree_model.predict(X_train_dt)
ypred_test = tree_model.predict(X_test_dt)

print(accuracy_score(y_train,ypred_train))
print(accuracy_score(y_test,ypred_test))
print(cross_val_score(tree_model,X_train_dt,y_train,cv=5,scoring='accuracy').mean())


# In[65]:


from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier(random_state=70)
param_grid = {'n_estimators':list(range(1,61))}

from sklearn.model_selection import GridSearchCV
rf_grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')
rf_grid.fit(X_train,y_train)


rf_model = rf_grid.best_estimator_

rf_feat = rf_model.feature_importances_

index = [i for i,x in enumerate(rf_feat) if x>0]

X_train_rf = X_train.iloc[:,index]
X_test_rf = X_test.iloc[:,index]

rf_model.fit(X_train_rf,y_train)

ypred_train = rf_model.predict(X_train_rf)
ypred_test = rf_model.predict(X_test_rf)

print(accuracy_score(y_train,ypred_train))
print(accuracy_score(y_test,ypred_test))
print(cross_val_score(rf_model,X_train_rf,y_train,cv=5,scoring='accuracy').mean())


# In[71]:


from sklearn.ensemble import AdaBoostClassifier

estimator = AdaBoostClassifier(random_state=rs)
param_grid = {'n_estimators':list(range(1,51))}

from sklearn.model_selection import GridSearchCV
adc_grid = GridSearchCV(estimator,param_grid,scoring='accuracy',cv=5)
adc_grid.fit(X_train,y_train)

adc_model = adc_grid.best_estimator_
adc_feat = adc_model.feature_importances_

index = [i for i,x in enumerate(adc_feat) if x>0]

X_train_adc = X_train.iloc[:,index]
X_test_adc = X_test.iloc[:,index]

adc_model.fit(X_train_adc,y_train)

ypred_train = adc_model.predict(X_train_adc)
ypred_test = adc_model.predict(X_test_adc)

print(accuracy_score(y_train,ypred_train))
print(accuracy_score(y_test,ypred_test))
print(cross_val_score(adc_model,X_train_adc,y_train,scoring='accuracy',cv=5).mean())


# In[80]:


from sklearn.ensemble import GradientBoostingClassifier

estimator = GradientBoostingClassifier(random_state=rs)
param_grid = {'n_estimators':list(range(1,10)),
              'learning_rate':[0,0.1,0.2,0.3,0.4,0.5,0.9,1]}

from sklearn.model_selection import GridSearchCV
gbc_grid = GridSearchCV(estimator,param_grid,scoring='accuracy',cv=5)
gbc_grid.fit(X_train,y_train)

gbc_model = gbc_grid.best_estimator_
gbc_feat = gbc_model.feature_importances_

index = [i for i,x in enumerate(gbc_feat) if x>0]

X_train_gbc = X_train.iloc[:,index]
X_test_gbc = X_test.iloc[:,index]

gbc_model.fit(X_train_gbc,y_train)

ypred_train = gbc_model.predict(X_train_gbc)
ypred_test = gbc_model.predict(X_test_gbc)

print(accuracy_score(y_train,ypred_train))
print(accuracy_score(y_test,ypred_test))
print(cross_val_score(gbc_model,X_train_gbc,y_train,scoring='accuracy',cv=5).mean())


# In[83]:


from xgboost import XGBClassifier

estimator = XGBClassifier(random_state=rs)
param_grid = {'n_estimators':[10,20,40,100],
              'max_depth':[3,4,5],
              'gamma':[0,0.15,0.35,0.5,1]}

from sklearn.model_selection import GridSearchCV
xgb_grid = GridSearchCV(estimator,param_grid,scoring='accuracy',cv=5)
xgb_grid.fit(X_train,y_train)

xgb_model = xgb_grid.best_estimator_
xgb_feat = xgb_model.feature_importances_

index = [i for i,x in enumerate(xgb_feat) if x>0]

X_train_xgb = X_train.iloc[:,index]
X_test_xgb= X_test.iloc[:,index]

xgb_model.fit(X_train_xgb,y_train)

ypred_train = xgb_model.predict(X_train_xgb)
ypred_test = xgb_model.predict(X_test_xgb)

print(accuracy_score(y_train,ypred_train))
print(accuracy_score(y_test,ypred_test))
print(cross_val_score(xgb_model,X_train_xgb,y_train,scoring='accuracy',cv=5).mean())


# In[ ]:




