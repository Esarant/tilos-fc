import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor
import os.path
from flask import jsonify,request
import calendar

months = {month: index for index, month in enumerate(calendar.month_name) if month}
days = {
    "Friday":       0,
    "Saturday":     1,
    "Sunday":       2,
    "Monday":       3,
    "Tuesday":      4,
    "Wednesday":    5,
    "Thursday":     6 
}
# Load the model
def load_model(_filename):
    this_file_path = os.path.dirname(__file__)
    filename = _filename
    filepath = os.path.join(this_file_path, filename)

    if os.path.isfile(filepath):
        loaded_model = pickle.load(open(filepath, 'rb'))
        return loaded_model
    else: 
        print(str("Model" + _filename +  "Error"))
        return None

# Generate the X_test from the model
def get_params_X_test(params):
    print(params)
    
   
    # The input has horizon and data
    horizon = int(params["horizon"])
    data = params["data"]
    data_list = []
    for record in data:
        #User input 5 variables: Month / Hour / Day / Yesterday Load / Last week Load
        Month = months.get(str(record["Month"]), None)
        
        if Month is None:
            return None
        Hour = int(record["Hour"])
        Day = days.get(str(record["Day"]), None)
        
        
        if Day is None:
            return None
        Yesterday_Load= float(record["Yesterday Load"])
        Last_Week_Load = float(record["Last Week Load"])
        data_list.append([Month, Hour, Day, Yesterday_Load, Last_Week_Load])
    
    #check for horizon 
    if (len(data_list)!=horizon):
        return None
    # X_test = np.array([Month, Hour, Day, Yesterday_Load, Last_Week_Load]).reshape(1, -1)
    X_test = np.asarray(data_list)
    
    return X_test

# Prediction from Decisiton tree model
def decision_tree_prediction(params):
    loaded_model = load_model('model1.sav')
    
    if loaded_model:
        try:
            
            X_test = get_params_X_test(params)
           
            if X_test is not None:
                # y_pred = loaded_model.predict(X_test)[0]
                
                y_pred = loaded_model.predict(X_test)
                return y_pred
            else:
                return None
        except:
            return None
    else:
        return None

# Prediction for lightgbm model
def lightgbm_prediction(params):
    loaded_model = load_model('model2.sav')
    if loaded_model:
        try:
            X_test = get_params_X_test(params)
            if X_test is not None:
                # y_pred = loaded_model.predict(X_test)[0]
                y_pred = loaded_model.predict(X_test)
                return y_pred
            else:
                return None
        except:
            return None
    else:
        return None

# Prediction for xgboost model
def xgboost_prediction(params):
    loaded_model = load_model('model3.sav')
    if loaded_model:
        try:
            X_test = get_params_X_test(params)
            if X_test is not None:
                # y_pred = loaded_model.predict(X_test)[0]
                y_pred = loaded_model.predict(X_test)
                return y_pred
            else:
                return None
        except:
            return None
    else:
        return None

# Prediction from all the models
def ensemble_prediction(params):
    y_pred1 = decision_tree_prediction(params)
    y_pred2 = lightgbm_prediction(params)
    y_pred3 = xgboost_prediction(params)
    if y_pred1 is not None and y_pred2 is not None and y_pred3 is not None:
        y_ensemble = (y_pred1 + y_pred2 + y_pred3) / 3
        return y_ensemble
    else:
        return None
# --------------------------------------------
def decision_tree_rest(params):
    #params = request.get_json()
    y_pred = decision_tree_prediction(params)
    if y_pred is not None:
        #return jsonify({"Consumption": str(y_pred), "message":""}), 200
        return {"Consumption": str(y_pred), "message":""}
    else:
        message="decision_tree Error"
        return {"Consumption": "", "message":message}


def lightgbm_rest():
    params = request.get_json()
    y_pred = lightgbm_prediction(params)
    if y_pred is not None:
        return jsonify({"Consumption": str(y_pred), "message":""}), 200
    else:
        message="lightgbm Error"
        return jsonify({"Consumption": "", "message":message}), 400


def xgboost_rest():
    params = request.get_json()
    y_pred = xgboost_prediction(params)
    if y_pred is not None:
        return jsonify({"Consumption": str(y_pred), "message":""}), 200
    else:
        message="xgboost Error"
        return jsonify({"Consumption": "", "message":message}), 400


def ensemble_rest():
    params = request.get_json()
    y_pred = ensemble_prediction(params)
    if y_pred is not None:
        return jsonify({"Consumption": str(y_pred), "message":""}), 200
    else:
        message="enseble Error"
        return jsonify({"Consumption": "", "message":message}), 400
