#imports
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#importing Dataset
df = pd.read_csv('https://raw.githubusercontent.com/jsambhav5/size-prediction/main/dataset/Body_Measurements_original.csv')
df = df.iloc[0:399, :]

data = df
X = data.drop(columns=['Size'])  # Features
y = data['Size']  # Target

le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])  # Label encoding for 'Gender'

#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and train the model
rf = RandomForestClassifier(random_state=42)
gnb = GaussianNB()
mnb = MultinomialNB()

#Training the Models
rf.fit(X_train, y_train)
gnb.fit(X_train, y_train)
mnb.fit(X_train, y_train)

#Predictions
y_pred_rf = rf.predict(X_test)
y_pred_gnb = gnb.predict(X_test)
y_pred_mnb = mnb.predict(X_test)

#Evaluation
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf * 100, "%")
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
print("Gaussian Naive Bayes Accuracy:", accuracy_gnb * 100, "%")
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
print("Multinomial Naive Bayes Accuracy:", accuracy_mnb * 100, "%")

#Dumping to Pickle
pickle.dump(rf, open('rf.sav','wb'))
pickle.dump(gnb, open('gnb.sav','wb'))
pickle.dump(mnb, open('mnb.sav','wb'))