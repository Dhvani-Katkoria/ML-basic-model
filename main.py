import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from sklearn.preprocessing import *#StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("Pima_Indian_diabetes.csv")
#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome

#filling the missing values
# mean and median values almost same, we prefer median
data.Pregnancies.fillna(value = data.Pregnancies.median(), inplace = True)
data.Glucose.fillna(value = data.Glucose.median(), inplace = True)
data.SkinThickness.fillna(value = data.SkinThickness.median(), inplace = True)
data.BMI.fillna(value = data.BMI.median(), inplace = True)
data.Age.fillna(value = data.Age.median(), inplace = True)


# Split dataset into training and testing datasets 80%-20%
TRAIN, TEST = train_test_split(data, test_size=0.2, random_state=42)
train_data=TRAIN[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
test_data=TEST[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
#seggregation of output labels
train_output=TRAIN['Outcome']
test_output=TEST['Outcome']

#plotting correlation matrix
sns.heatmap(data.corr(), annot = True)
plt.show()

#scaling using min max scaler
norm = MinMaxScaler()
norm.fit(train_data)
train_norm= norm.transform(train_data)
test_norm = norm.transform(test_data)
print("train data set shape: ",train_data.shape)
print("test data set shape: ",test_data.shape)


#trying different models for classification

#no.of rows of test data
n=test_data.shape[0]

print("\nlogistic regression")
log_reg=LogisticRegression()
log_reg.fit(train_norm,train_output)
log_predict=log_reg.predict(test_norm)
actual_output=list(test_output)
c=0
for i in range(n):
    if(log_predict[i]!=actual_output[i]):
        c+=1
print("\nAccuracy Score ",1-c/n)
print('Confusion Matrix : ',confusion_matrix(actual_output,log_predict))
# print('Report : ',classification_report(actual_output,log_predict))

print("\nnavie bayesian")
nv_by=GaussianNB()
nv_by.fit(train_norm,train_output)
nvby_predict=nv_by.predict(test_norm)
actual_output=list(test_output)
c=0
for i in range(n):
    if(nvby_predict[i]!=actual_output[i]):
        c+=1
print("\nAccuracy Score ",1-c/n)
print('Confusion Matrix : ',confusion_matrix(actual_output,nvby_predict))
# print('Report : ',classification_report(actual_output,nvby_predict))

print("\nStochastic Gradient Descent")
stoc_grad=SGDClassifier(loss="log", shuffle=True, random_state=101)
stoc_grad.fit(train_norm,train_output)
stocgrad_predict=stoc_grad.predict(test_norm)
actual_output=list(test_output)
c=0
for i in range(n):
    if(stocgrad_predict[i]!=actual_output[i]):
        c+=1
print("\nAccuracy Score ",1-c/n)
print('Confusion Matrix : ',confusion_matrix(actual_output,stocgrad_predict))
# print('Report : ',classification_report(actual_output,stocgrad_predict))

print("\nK nearest Neighbours")
knn=KNeighborsClassifier()
knn.fit(train_norm,train_output)
knn_predict=knn.predict(test_norm)
actual_output=list(test_output)
c=0
for i in range(n):
    if(knn_predict[i]!=actual_output[i]):
        c+=1
print("\nAccuracy Score ",1-c/n)
print('Confusion Matrix : ',confusion_matrix(actual_output,knn_predict))
# print('Report : ',classification_report(actual_output,knn_predict))

print("\nDecision Tree classifier")
deci_tree=DecisionTreeClassifier(max_depth=10,random_state=101,min_samples_leaf=15)
deci_tree.fit(train_norm,train_output)
deci_tree_predict=deci_tree.predict(test_norm)
actual_output=list(test_output)
c=0
for i in range(n):
    if(deci_tree_predict[i]!=actual_output[i]):
        c+=1
print("\nAccuracy Score ",1-c/n)
print('Confusion Matrix : ',confusion_matrix(actual_output,deci_tree_predict))
# print('Report : ',classification_report(actual_output,deci_tree_predict))

print("\nRandomForestClassifier")
rand_forest=RandomForestClassifier(n_estimators=70,oob_score=True,n_jobs=1,min_samples_leaf=30,random_state=101)
rand_forest.fit(train_norm,train_output)
rand_forest_predict=rand_forest.predict(test_norm)
actual_output=list(test_output)
c=0
for i in range(n):
    if(rand_forest_predict[i]!=actual_output[i]):
        c+=1
print("\nAccuracy Score ",1-c/n)
print('Confusion Matrix : ',confusion_matrix(actual_output,rand_forest_predict))
# print('Report : ',classification_report(actual_output,rand_forest_predict))

print("\nSupport Vector Clasification")
svmachine=SVC(kernel="linear")
svmachine.fit(train_norm,train_output)
svmachine_predict=svmachine.predict(test_norm)
actual_output=list(test_output)
c=0
for i in range(n):
    if(svmachine_predict[i]!=actual_output[i]):
        c+=1
print("\nAccuracy Score ",1-c/n)
print('Confusion Matrix : ',confusion_matrix(actual_output,svmachine_predict))
# print('Report : ',classification_report(actual_output,svmachine_predict))

