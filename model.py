import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('liver_data.csv')


df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean(), inplace = True)
df.rename(columns = {'Dataset': 'output'}, inplace = True)

df = pd.get_dummies(df)
df.head()

df.drop( ['Gender_Female'], axis =1 , inplace  = True)

X = df.drop('output', axis = 1)
Y = df['output']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 12)

from sklearn.ensemble import RandomForestClassifier
RC = RandomForestClassifier()
RCmodel = RC.fit(X_train, Y_train)
Y_pred = RCmodel.predict(X_test)
cmr = confusion_matrix(Y_pred,Y_test)
print(cmr)
acr = accuracy_score(Y_pred, Y_test)
print(acr)

import pickle
file = open("Random_forest.pkl",'wb')
pickle.dump(RCmodel, file)
model = pickle.load(open('Random_forest.pkl', 'rb'))

