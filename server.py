#importing libraries
import os
import numpy as np
import pandas as pd
import flask
import pickle
import json
from flask import Flask, request, jsonify

#creating instance of the class
app=Flask(__name__)

@app.route('/')


#prediction function
@app.route('/satisfaction',methods = ['POST'])
def result():
    headers = ['DistanceFromHome', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction', 'PerformanceRating',
 'RelationshipSatisfaction', 'WorkLifeBalance', 'YearsAtCompany', 'YearsSinceLastPromotion', 'MonthlyIncome', 'OverTime']
    payload = request.json['data']
    values = [float(i) for i in payload.split(',')]
    input_variables = pd.DataFrame([values],
                                columns=headers, 
                                dtype=float,
                                index=['input'])
    model = pickle.load(open("model_satisfaction.pkl","rb"))
    prediction = model.predict(input_variables)
    return jsonify("Prediction", int(prediction))
#    ret = "prediction: " + str(prediction)
#    if (prediction == 1):
#        ret = "Yes"
#    elif (prediction == 0) :
#        ret = "No"
# {"data": "2, 2, 3, 4, 3, 4, 3, 5, 4, 5001, 0 "}

@app.route('/promotion', methods = ['POST'])
def ret():
    headers = ['Education', 'JobInvolvement', 'MonthlyIncome', 'NumCompaniesWorked',
 'PercentSalaryHike', 'PerformanceRating', 'TotalWorkingYears', 'YearsAtCompany']
    payload = request.json['data']
    values = [float(i) for i in payload.split(',')]
    input_variables = pd.DataFrame([values],
                                columns=headers, 
                                dtype=float,
                                index=['input'])
    model = pickle.load(open("model_promotion.pkl","rb"))
    prediction = model.predict(input_variables)
    pd.Series(prediction).to_json(orient='values')
#    ret = "prediction: " + str(prediction)
#    if (prediction == 1):
#        ret = "Yes"
#    elif (prediction == 0) :
#        ret = "No"
    return prediction

# {"data": "4, 3, 5561, 0, 16, 3, 6, 5"}
