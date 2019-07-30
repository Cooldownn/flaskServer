import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn as sk

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

data = pd.read_csv("/Users/cooldown/Desktop/Cooldown/CapA/data/data.csv")

y = data['Satisfaction']
X = data.drop('Satisfaction', axis = 1)


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Predict
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# Get headers for payload
headers = ['EnvironmentSatisfaction','JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction','MonthlyIncome']

# Test model
#input_variables = pd.DataFrame([[3, 3, 3, 1, 2, 4, 3, 5, 4, 8000, 1]],
#                                columns=headers, 
#                                dtype=float,
#                                index=['input'])

# Get the model's prediction
#prediction = clf.predict(input_variables)
#if (prediction == 1):
#    print("Result: Yes")
#elif (prediction == 0) :
#    print("Result: No")
#print("Prediction: ", prediction)                                

# Saving model
import pickle
pickle.dump(classifier,open('model_satisfaction.pkl','wb'))
