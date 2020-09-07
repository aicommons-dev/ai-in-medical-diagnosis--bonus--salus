# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Importing the dataset
dataset = pd.read_csv('Data/Diabetes.csv')
col = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X = dataset[list(col)].values
y = dataset['Outcome'].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)
X_test = sc.transform(X)

from sklearn.ensemble import RandomForestClassifier


classifier = RandomForestClassifier(n_estimators=10)
classifier = classifier.fit(X_train, y)
acc = classifier.score(X_train, y)


# save the model to disk
filename = 'Models/Diabetes.pickle'
pickle.dump(classifier, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Predicting the Test set results
result = loaded_model.score(X_test, y)
print(result)
print("Test score: {0:.2f} %".format(100 * result))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y, y_pred)
print(cm)
