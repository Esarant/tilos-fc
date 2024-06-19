#BD4NRG LSP8
#Dataset Tilos

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime, timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product
import lightgbm as lgb




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

# Plots
	# plt.scatter(y_pred, y_test)
	# plt.plot(y_pred, label="Prediction")
	# plt.plot(y_test, label = "Real")
	# plt.legend()
	# plt.show()

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

# Plots
	# xgb.plot_importance(model, max_num_features=10) # top 10 most important features
	# plt.scatter(y_pred, y_test)
	# plt.plot(y_pred, label="Prediction")
	# plt.plot(y_test, label = "Real")
	# plt.legend()
	# plt.show()

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

# Plots
	# xgb.plot_importance(model, max_num_features=10) # top 10 most important features
	# plt.scatter(y_pred, y_test)
	# plt.plot(y_pred, label="Prediction")
	# plt.plot(y_test, label = "Real")
	# plt.legend()
	# plt.show()

	return(y_pred)


# ====================================================================


# Function that generates hourly datetimes between start_date and end_date
def daterange(start_date, end_date):
    delta = timedelta(hours=1)
    while start_date < end_date:
        yield start_date
        start_date += delta

# Function that supports visualization of hourly data
def visualization_hourly(start_date, end_date, number_of_data, data):
	dates = []
	for single_date in daterange(start_date, end_date):
		dates.append(single_date)
	data = data[:number_of_data]
	plt.scatter(dates, data)
	plt.title('Hourly Energy Demand')
	plt.xlabel('Dates')
	plt.ylabel('Demand')
	plt.show()


# Function that supports visualization of daily data
def visualization_daily(start_date, end_date, number_of_data, data):
	dates = []
	dates = pd.date_range(start_date,end_date-timedelta(days=1),freq='d')
	data = data[:number_of_data]
	plt.plot(dates, data)
	plt.title('Daily Energy Demand')
	plt.xlabel('Dates')
	plt.ylabel('Demand')
	plt.show()




number_of_10mins = 158112
number_of_hours = 26352
number_of_days = 1098

# Demand in 10 minute data
demand_10min = pd.read_excel('demand_3year.xlsx', 'Demand')

# Aggregate to demand in hourly data
demand_10min_array = demand_10min["10min Load (kW)"].to_numpy()
demand_hourly_array =  np.zeros(number_of_hours)
for i in range(number_of_hours):
	temp = 0 
	for j in range(6):
		temp += demand_10min_array[i*6 + j]
	demand_hourly_array[i] = temp / 6

# plt.plot(demand_hourly_array)
# plt.show()

# Aggreggate to demand in daily data
demand_daily_array =  np.zeros(number_of_days)
for i in range(number_of_days):
	temp = 0 
	for j in range(24):
		temp += demand_hourly_array[i*24 + j]
	demand_daily_array[i] = temp

# plt.plot(demand_daily_array)
# plt.show()

start_date = datetime(2016, 1, 1, 0, 00)
end_date = datetime(2017, 1, 1, 0, 00)
# visualization_hourly(start_date, end_date, 8784, demand_hourly_array)
# visualization_daily(start_date, end_date, 366, demand_daily_array)

# =====================  Weather Data ======================================================================
#Create the final data which will be used
demand_hourly = demand_10min.iloc[::6, :2]
print(demand_hourly) 

months = [0 for i in range(number_of_hours)]
hours = [0 for i in range(number_of_hours)]
for i in range(number_of_hours):
	months[i] = demand_hourly.iloc[i, 0].to_pydatetime().month
	hours[i] = demand_hourly.iloc[i, 1].hour

days = [0 for i in range(number_of_hours)]
for i in range(number_of_hours):
	days[i] = (i // 24) % 7 

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

demand_hourly['demand'] = demand_hourly_array
demand_hourly['month'] = months
demand_hourly['hour'] = hours
demand_hourly['day'] = days
demand_hourly['yesterday_demand'] = yesterday_demand
demand_hourly['last_week_demand'] = last_week_demand

# 9-months
size = 6546
# 1-year
size = 8784 
# 1-year and 3 months
size = 10946
# 1-year and 6 months
size = 13000

# We will keep only the first year because we have weather data for only one year
demand_hourly = demand_hourly.iloc[:size, :]

weather_hourly = pd.read_excel('demand_3year.xlsx', 'Weather')
print(len(weather_hourly))
weather_hourly = weather_hourly.iloc[:size, :]

demand_hourly.index = weather_hourly.index


# This is the final dataset that will be used
final_dataset = pd.concat([demand_hourly, weather_hourly], axis=1)
final_dataset.drop('ρ(kg/m3)', axis=1, inplace=True)
final_dataset.drop('P(hPa)', axis=1, inplace=True)
final_dataset.drop('V(m/s)', axis=1, inplace=True)
final_dataset.drop('T(oC)', axis=1, inplace=True)


print(final_dataset)

# ======================= Statistics, correlations and ML models =====================
# 
# demand = final_dataset["demand"]
# temperature = final_dataset["T(oC)"]
# v = final_dataset["V(m/s)"]
# hPa = final_dataset["P(hPa)"]
# p = final_dataset["ρ(kg/m3)"]

# correlation = demand.corr(temperature)
# correlation2 = demand.corr(v)
# correlation3 = demand.corr(hPa)
# correlation4 = demand.corr(p)
# print(correlation)
# print(correlation2)
# print(correlation3)
# print(correlation4)


# ML Decision Tree

# Model Construction
X = final_dataset.iloc[:, 3:].values
y = final_dataset.iloc[:, 2].values


# # Encoding categorical data
# # Encoding the Independent Variable
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# # X = ct.fit_transform(X).toarray()
# X = np.array(ct.fit_transform(X))


# Splitting the dataset into the Training set and Test set random
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

X_train, X_test= np.split(X, [size-168])
y_train, y_test= np.split(y, [size-168])


# print(X_train)
# print(X_test)
# print(len(X_train))
# print(len(X_test))






# =======================================  Running DecisionTreeRegressor from sklearn.tree  ================================================
parameters = {
	'criterion': ['mse'], 
	'random_state': [0], 
	'max_depth': [5, 10, 15, 20], 
	'min_samples_split': [2, 5, 10, 20], 
	'min_samples_leaf': [1, 2, 5, 10]
}

# for values in product(*parameters.values()):
#     Decision_Tree(X_train, y_train, X_test, y_test, values[0], values[1], values[2], values[3], values[4])

# # Best so far
# Decision_Tree(X_train, y_train, X_test, y_test, 'mse', 0, 15, 4, 1)
# # Best so far plus day
# Decision_Tree(X_train, y_train, X_test, y_test, 'mse', 0, 10, 5, 1)
# # Best so far plus day and yesterday_demand
# Decision_Tree(X_train, y_train, X_test, y_test, 'mse', 0, 12, 20, 6)

pred_sklearn1 = Decision_Tree(X_train, y_train, X_test, y_test, 'mse', 0, 5, 10, 2)
pred_sklearn2 = Decision_Tree(X_train, y_train, X_test, y_test, 'mse', 0, 10, 10, 2)
pred_sklearn3 = Decision_Tree(X_train, y_train, X_test, y_test, 'mse', 0, 20, 10, 10)

ensemble_sklearn = [(i + j + k) / 3 for i, j, k in zip(pred_sklearn1, pred_sklearn2, pred_sklearn3)]
print("Ensemble-skLearn")
print("r_square", format(r2_score(ensemble_sklearn, y_test), ".3f"))
print("MAE", format(mean_absolute_error(ensemble_sklearn, y_test), ".3f"))
print("RMSE", format(mean_squared_error(ensemble_sklearn, y_test, squared=False), ".3f"))


# ===============================================  Running XGBoost model  =======================================================
# Create D-Arrays
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

parameters = {
	'learning_rate': [ 0.1, 0.3, 0.5], 
	'min_split_loss': [2, 5, 10, 20], 
	'max_depth': [2, 5, 10, 20], 
}

# for values in product(*parameters.values()):
# 	Decision_Tree_XGBoost(dtrain, dtest, values[0], values[1], values[2])

#Best so far all plus day and yesterday_demand and last_week_demand
# Decision_Tree_XGBoost(dtrain, dtest, 0.3, 20, 10)

pred_xgb1 = Decision_Tree_XGBoost(dtrain, dtest, 0.5, 2, 5)
pred_xgb2 = Decision_Tree_XGBoost(dtrain, dtest, 0.3, 2, 5)
pred_xgb3 = Decision_Tree_XGBoost(dtrain, dtest, 0.3, 2, 10)

ensemble_xgb = [(i + j + k) / 3 for i, j, k in zip(pred_xgb1, pred_xgb2, pred_xgb3)]
print("Ensemble-XGB")
print("r_square", format(r2_score(ensemble_xgb, y_test), ".3f"))
print("MAE", format(mean_absolute_error(ensemble_xgb, y_test), ".3f"))
print("RMSE", format(mean_squared_error(ensemble_xgb, y_test, squared=False), ".3f"))

# ==================================================  Running LightGBM model =====================================================
train_data = lgb.Dataset(X_train, label=y_train)
parameters = {
    "objective": ["regression"],
    "metric": ["mse"],
    "num_leaves": [10, 30, 50],
    "learning_rate": [0.1, 0.3, 0.5],
    "bagging_fraction": [0.7],
    "feature_fraction": [0.3, 0.5, 0.7, 0.9],
    "verbosity": [-1]
}

# for values in product(*parameters.values()):
# 	Decision_Tree_LightGBM(train_data, X_test, y_test, values[0], values[1], values[2], values[3], values[4], values[5], values[6])

#Best so far all plus day and yesterday_demand and last_week_demand
# Decision_Tree_LightGBM(train_data, X_test, y_test, "regression", "mse", 50, 0.3, 0.7, 0.9, -1)

pred_lightgbm1 = Decision_Tree_LightGBM(train_data, X_test, y_test, "regression", "mse", 30, 0.1, 0.7, 0.9, -1)
pred_lightgbm2 = Decision_Tree_LightGBM(train_data, X_test, y_test, "regression", "mse", 50, 0.1, 0.7, 0.5, -1)
pred_lightgbm3 = Decision_Tree_LightGBM(train_data, X_test, y_test, "regression", "mse", 30, 0.3, 0.7, 0.7, -1)
pred_lightgbm4 = Decision_Tree_LightGBM(train_data, X_test, y_test, "regression", "mse", 10, 0.1, 0.7, 0.3, -1)

ensemble_lightgbm = [(i + j + k + l) / 4 for i, j, k, l in zip(pred_lightgbm1, pred_lightgbm2, pred_lightgbm3, pred_lightgbm4)]
print("Ensemble-LightGBM")
print("r_square", format(r2_score(ensemble_lightgbm, y_test), ".3f"))
print("MAE", format(mean_absolute_error(ensemble_lightgbm, y_test), ".3f"))
print("RMSE", format(mean_squared_error(ensemble_lightgbm, y_test, squared=False), ".3f"))

ensemble_23 = [(i + j) / 2 for i, j, in zip(ensemble_xgb, ensemble_lightgbm)]
print("Ensemble-XGB-LightGBM")
print("r_square", format(r2_score(ensemble_23, y_test), ".3f"))
print("MAE", format(mean_absolute_error(ensemble_23, y_test), ".3f"))
print("RMSE", format(mean_squared_error(ensemble_23, y_test, squared=False), ".3f"))

ensemble_all = [(i + j + k) / 3 for i, j, k in zip(ensemble_sklearn, ensemble_xgb, ensemble_lightgbm)]
print("Ensemble-all")
print("r_square", format(r2_score(ensemble_all, y_test), ".3f"))
print("MAE", format(mean_absolute_error(ensemble_all, y_test), ".3f"))
print("RMSE", format(mean_squared_error(ensemble_all, y_test, squared=False), ".3f"))


#Plot together
# plt.plot(pred1, label="sklearn.tree")
# plt.plot(pred2, label="XGBoost")
# plt.plot(ensemble_sklearn, label="Ensemble sklearn Demand")
plt.plot(ensemble_xgb, label="Ensemble XGB Demand")
plt.plot(ensemble_lightgbm, label="Ensemble LightGBM Demand")
plt.plot(ensemble_23, label="Ensemble XGB-LightGBM Demand")
# plt.plot(ensemble_all, label="Ensemble-All Demand")
plt.plot(y_test, label = "Real Demand")
plt.title('Tilos Island Consumption Prediction (18-month)')
plt.xlabel('Date and Hour')
plt.ylabel('Consumption')
plt.legend()
plt.show()