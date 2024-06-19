import numpy as np
import pickle

#User input 5 variables: Month / Hour / Day / Yesterday Load / Last week Load

# Example: The 5 values inserted by the user in the GUI must be converted into numpy ndarray and then reshaped in order to be transformed with the scaler
X_test = np.array([4, 15, 3, 214, 232]).reshape(1, -1)

X_test = [[4, 15, 3, 214, 232], [7, 15, 6, 214, 232], [4, 22, 3, 158, 170]]

'''
Days Encoding
0: Friday
1: Saturday
2: Sunday
3: Monday
4: Tuesday
5: Wednesday
6: Thursday
'''

from sklearn.tree import DecisionTreeRegressor

# Import pre-trained model
filename = 'model1.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#Prediction of the production with the loaded model from SKLearn
y_pred1 = loaded_model.predict(X_test)
print("PV Production:", y_pred1)


import lightgbm as lgb

# Import pre-trained model
filename = 'model2.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#Prediction of the production with the loaded model from LightGBM
y_pred2 = loaded_model.predict(X_test)
print("PV Production:", y_pred2)


import xgboost as xgb

# Import pre-trained model
filename = 'model3.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#Prediction of the production with the loaded model from XGBoost
y_pred3 = loaded_model.predict(np.asarray(X_test))
print("PV Production:", y_pred3)


def ensemble_forecast(y_pred1, y_pred2, y_pred3):
	y_ensemble = (y_pred1 + y_pred2 + y_pred3) / 3
	return y_ensemble

ensemble = ensemble_forecast(y_pred1, y_pred2, y_pred3)
print("PV Production:", ensemble)