import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

number_of_hours = 19008


# Function that implements decision tree ML
def Decision_Tree(X_train, y_train, X_test, y_test, criterion, random_state, max_depth, min_samples_split, min_samples_leaf):
# Training the Decision Tree Regression model on the whole dataset
	from sklearn.tree import DecisionTreeRegressor
	regressor = DecisionTreeRegressor(criterion=criterion, random_state = random_state, max_depth = max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
	regressor.fit(X_train, y_train)
	y_pred = regressor.predict(X_test)

# # Parameters
# 	print("random_state:", random_state, ", max_depth:", max_depth, ", min_samples_split:", min_samples_split, ", min_samples_leaf:", min_samples_leaf)

# # Metrics
# 	print("r_square", format(r2_score(y_pred, y_test), ".3f"))
# 	print("MAE", format(mean_absolute_error(y_pred, y_test), ".3f"))
# 	print("RMSE", format(mean_squared_error(y_pred, y_test, squared=False), ".3f"))

# # Plots
# 	# plt.scatter(y_pred, y_test)
# 	plt.plot(y_pred, label="Prediction")
# 	plt.plot(y_test, label = "Real")
# 	plt.legend()
# 	plt.show()

	return(y_pred)

# ====================================================================

def Decision_Tree_XGBoost(dtrain, dtest, learning_rate, min_split_loss, max_depth):
# Set up parameters dictionary
	params = {"objective":"reg:squarederror", "learning_rate":learning_rate, "min_split_loss":min_split_loss, "max_depth":max_depth}

# Train the model and make predictions
	model = xgb.train(params = params, dtrain = dtrain, num_boost_round = 10)
	y_pred = model.predict(dtest)

# # Parameters
# 	print("learning_rate:", learning_rate, ", min_split_loss:", min_split_loss, ", max_depth:", max_depth)

# # Metrics
# 	print("r_square", format(r2_score(y_pred, y_test), ".3f"))
# 	print("MAE", format(mean_absolute_error(y_pred, y_test), ".3f"))
# 	print("RMSE", format(mean_squared_error(y_pred, y_test, squared=False), ".3f"))

# # Plots
# 	# xgb.plot_importance(model, max_num_features=10) # top 10 most important features
# 	# plt.scatter(y_pred, y_test)
# 	plt.plot(y_pred, label="Prediction")
# 	plt.plot(y_test, label = "Real")
# 	plt.legend()
# 	plt.show()

	return(y_pred)


# ====================================================================

def Decision_Tree_LightGBM(train_data, X_test, y_test, objective, metric, num_leaves, learning_rate, bagging_fraction, feature_fraction, verbosity):
# Set up parameters dictionary
	params = {"objective": objective,
    "metric": metric,
    "num_leaves": num_leaves,
    "learning_rate": learning_rate,
    "bagging_fraction": bagging_fraction,
    "feature_fraction": feature_fraction,
    "verbosity": verbosity}

# Train the model and make predictions
	model = lgb.train(params, train_data)
	y_pred = model.predict(X_test)

# # Parameters
# 	print("num_leaves:", num_leaves, ", learning_rate:", learning_rate, ", feature_fraction:", feature_fraction)

# # Metrics
# 	print("r_square", format(r2_score(y_pred, y_test), ".3f"))
# 	print("MAE", format(mean_absolute_error(y_pred, y_test), ".3f"))
# 	print("RMSE", format(mean_squared_error(y_pred, y_test, squared=False), ".3f"))

# # Plots
# 	# plt.scatter(y_pred, y_test)
# 	plt.plot(y_pred, label="Prediction")
# 	plt.plot(y_test, label = "Real")
# 	plt.legend()
# 	plt.show()

	return(y_pred)

initial_dataset = pd.read_excel('island_consumption.xlsx')
initial_dataset['Month'] = initial_dataset.apply(lambda row: int(str(row.DateTime)[4] + str(row.DateTime)[5]), axis=1)
initial_dataset['Hour'] = initial_dataset.apply(lambda row: int(str(row.DateTime)[8] + str(row.DateTime)[9]), axis=1)

days = [0 for i in range(number_of_hours)]
for i in range(number_of_hours):
	days[i] = (i // 24) % 7 


demand_hourly_array = initial_dataset["Consumption"].to_numpy()

yesterday_demand = [0 for i in range(number_of_hours)]
last_week_demand = [0 for i in range(number_of_hours)]
for i in range(number_of_hours):
	if i <= 24:
		yesterday_demand[i] = demand_hourly_array[i]
	else:
		yesterday_demand[i] = demand_hourly_array[i-24]
	if i <= 168:
		last_week_demand[i] = demand_hourly_array[i]
	else:
		last_week_demand[i] = demand_hourly_array[i-168]

initial_dataset['Day'] = days
initial_dataset['Yesterday_demand'] = yesterday_demand
initial_dataset['Last_week_demand'] = last_week_demand

print(initial_dataset)

X = initial_dataset.iloc[:, [2,3,4,5,6]].values
y = initial_dataset.iloc[:, 1].values

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


num_of_weeks = 52
initial_training = 10248
observations_test = 24*7
predictions1 = []
predictions2 = []
predictions3 = []
predictions4 = []
real = y[initial_training:initial_training+num_of_weeks*observations_test]
print(real)

for i in range(num_of_weeks):

	train_end = initial_training+i*observations_test

	#Create Training Set
	X_train = X[:train_end]
	y_train = y[:train_end]

	#Create Test Set
	X_test = X[train_end:train_end+observations_test]
	y_test = y[train_end:train_end+observations_test]


#MLP Implementation
	regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
	mlp_pred = regr.predict(X_test)

	for j in range(len(mlp_pred)):
		predictions4.append(mlp_pred[j])

	print("MLP")
	print("r_square", format(r2_score(mlp_pred, y_test), ".3f"))
	print("MAE", format(mean_absolute_error(mlp_pred, y_test), ".3f"))
	print("RMSE", format(mean_squared_error(mlp_pred, y_test, squared=False), ".3f"))

	# pred_sklearn1 = Decision_Tree(X_train, y_train, X_test, y_test, 'mse', 0, 5, 10, 2)
	# pred_sklearn2 = Decision_Tree(X_train, y_train, X_test, y_test, 'mse', 0, 10, 10, 2)
	# pred_sklearn3 = Decision_Tree(X_train, y_train, X_test, y_test, 'mse', 0, 20, 10, 10)
	# ensemble_sklearn = [(i + j + k) / 3 for i, j, k in zip(pred_sklearn1, pred_sklearn2, pred_sklearn3)]
	# print("Ensemble-skLearn")
	# print("r_square", format(r2_score(ensemble_sklearn, y_test), ".3f"))
	# print("MAE", format(mean_absolute_error(ensemble_sklearn, y_test), ".3f"))
	# print("RMSE", format(mean_squared_error(ensemble_sklearn, y_test, squared=False), ".3f"))

	dtrain = xgb.DMatrix(X_train, label=y_train)
	dtest = xgb.DMatrix(X_test, label=y_test)
	pred_xgb1 = Decision_Tree_XGBoost(dtrain, dtest, 0.5, 2, 5)
	pred_xgb2 = Decision_Tree_XGBoost(dtrain, dtest, 0.3, 2, 5)
	pred_xgb3 = Decision_Tree_XGBoost(dtrain, dtest, 0.3, 2, 10)
	ensemble_xgb = [(i + j + k) / 3 for i, j, k in zip(pred_xgb1, pred_xgb2, pred_xgb3)]
	# print("Ensemble-XGB")
	# print("r_square", format(r2_score(ensemble_xgb, y_test), ".3f"))
	# print("MAE", format(mean_absolute_error(ensemble_xgb, y_test), ".3f"))
	# print("RMSE", format(mean_squared_error(ensemble_xgb, y_test, squared=False), ".3f"))

	train_data = lgb.Dataset(X_train, label=y_train)
	pred_lightgbm1 = Decision_Tree_LightGBM(train_data, X_test, y_test, "regression", "mse", 30, 0.1, 0.7, 0.9, -1)
	pred_lightgbm2 = Decision_Tree_LightGBM(train_data, X_test, y_test, "regression", "mse", 50, 0.1, 0.7, 0.5, -1)
	pred_lightgbm3 = Decision_Tree_LightGBM(train_data, X_test, y_test, "regression", "mse", 30, 0.3, 0.7, 0.7, -1)
	pred_lightgbm4 = Decision_Tree_LightGBM(train_data, X_test, y_test, "regression", "mse", 10, 0.1, 0.7, 0.3, -1)
	ensemble_lightgbm = [(i + j + k + l) / 4 for i, j, k, l in zip(pred_lightgbm1, pred_lightgbm2, pred_lightgbm3, pred_lightgbm4)]
	ensemble_lightgbm1 = [(i + j) / 2 for i, j, in zip(pred_lightgbm1, pred_lightgbm2)]
	ensemble_lightgbm2 = [(i + j) / 2 for i, j, in zip(pred_lightgbm3, pred_lightgbm4)]
	# print("Ensemble-LightGBM")
	# print("r_square", format(r2_score(ensemble_lightgbm, y_test), ".3f"))
	# print("MAE", format(mean_absolute_error(ensemble_lightgbm, y_test), ".3f"))
	# print("RMSE", format(mean_squared_error(ensemble_lightgbm, y_test, squared=False), ".3f"))

	ensemble_23 = [(i + j) / 2 for i, j, in zip(ensemble_xgb, ensemble_lightgbm)]
	print("Ensemble-XGB-LightGBM")
	print("r_square", format(r2_score(ensemble_23, y_test), ".3f"))
	print("MAE", format(mean_absolute_error(ensemble_23, y_test), ".3f"))
	print("RMSE", format(mean_squared_error(ensemble_23, y_test, squared=False), ".3f"))

	for j in range(len(ensemble_23)):
		predictions1.append(ensemble_xgb[j])
		predictions2.append(ensemble_lightgbm1[j])
		predictions3.append(ensemble_lightgbm2[j])
		# predictions4.append(ensemble_23[j])



print(len(predictions1))
print("r_square", format(r2_score(predictions1, real), ".3f"))
print("MAE", format(mean_absolute_error(predictions1, real), ".3f"))
print("RMSE", format(mean_squared_error(predictions1, real, squared=False), ".3f"))

print(len(predictions2))
print("r_square", format(r2_score(predictions2, real), ".3f"))
print("MAE", format(mean_absolute_error(predictions2, real), ".3f"))
print("RMSE", format(mean_squared_error(predictions2, real, squared=False), ".3f"))

print(len(predictions3))
print("r_square", format(r2_score(predictions3, real), ".3f"))
print("MAE", format(mean_absolute_error(predictions3, real), ".3f"))
print("RMSE", format(mean_squared_error(predictions3, real, squared=False), ".3f"))

print(len(predictions4))
print("r_square", format(r2_score(predictions4, real), ".3f"))
print("MAE", format(mean_absolute_error(predictions4, real), ".3f"))
print("RMSE", format(mean_squared_error(predictions4, real, squared=False), ".3f"))


# print(len(predictions4))
# print("r_square", format(r2_score(predictions4, real), ".3f"))
# print("MAE", format(mean_absolute_error(predictions4, real), ".3f"))
# print("RMSE", format(mean_squared_error(predictions4, real, squared=False), ".3f"))

# plt.plot(ensemble_xgb, label="Ensemble XGB Demand")
# plt.plot(ensemble_lightgbm, label="Ensemble LightGBM Demand")
# plt.plot(predictions, label="Ensemble XGB-LightGBM Demand")
# plt.plot(real, label = "Real Demand")
# plt.title('Tilos Island PV Production Prediction May 2021')
# plt.xlabel('Date and Hour')
# plt.ylabel('Consumption (Watt)')
# plt.legend()
# plt.show()


pd.DataFrame(real).to_excel("output.xlsx", sheet_name='PV_Output') 
pd.DataFrame(predictions1).to_excel("output2.xlsx", sheet_name='PV_Output') 
pd.DataFrame(predictions2).to_excel("output3.xlsx", sheet_name='PV_Output') 
pd.DataFrame(predictions3).to_excel("output4.xlsx", sheet_name='PV_Output') 
pd.DataFrame(predictions4).to_excel("output5.xlsx", sheet_name='PV_Output') 
