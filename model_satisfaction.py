import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn as sk

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

data = pd.read_csv("/Users/cooldown/Desktop/Cooldown/CapA/data/data_satisfaction.csv")

y = data['Satisfaction']
X = data.drop('Satisfaction', axis = 1)

# Encode
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X.iloc[:, 10] = le.fit_transform(X.iloc[:,10])
y = le.fit_transform(y)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
#print(y_pred)


# Get headers for payload
headers = ['DistanceFromHome', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating',
 'RelationshipSatisfaction', 'WorkLifeBalance', 'YearsAtCompany', 'YearsSinceLastPromotion', 'MonthlyIncome', 'OverTime']

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
pickle.dump(clf,open('model_satisfaction.pkl','wb'))
