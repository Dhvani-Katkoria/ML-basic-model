import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("Pima_Indian_diabetes.csv")
#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome

print("Missing values:\n",data.isnull().sum())

print("deleting rows :")
print("\nold length= ",len(data))
data = data.dropna(axis = 0, how ='any')
print("new length= ",len(data))

print("\nranking features: ")
input = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
input.to_numpy()
model = ExtraTreesClassifier()
model.fit(input, list(data['Outcome']))
# display the relative importance of each attribute
print("\nsignificance of exsisting features:")
print(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
print(model.feature_importances_)

print("\n\n\nMean values:\n",data.mean())
print("\nMedian values:\n",data.median())
