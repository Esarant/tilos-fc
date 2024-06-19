import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

initial_dataset = pd.read_excel('island_consumption.xlsx')
initial_dataset['Month'] = initial_dataset.apply(lambda row: int(str(row.DateTime)[4] + str(row.DateTime)[5]), axis=1)
initial_dataset['Hour'] = initial_dataset.apply(lambda row: int(str(row.DateTime)[8] + str(row.DateTime)[9]), axis=1)

days = [0 for i in range(len(initial_dataset))]
for i in range(len(initial_dataset)):
	days[i] = (i // 24) % 7 


demand_hourly_array = initial_dataset["Consumption"].to_numpy()

yesterday_demand = [0 for i in range(len(initial_dataset))]
last_week_demand = [0 for i in range(len(initial_dataset))]
for i in range(len(initial_dataset)):
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


print(X)
print(y)


from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state =  0, max_depth = 10, min_samples_split= 10, min_samples_leaf=2)
regressor.fit(X, y)

filename = 'model1.sav'
pickle.dump(regressor, open(filename, 'wb'))



import lightgbm as lgb

params = {"objective": "regression",
"metric": "mse",
"num_leaves": 50,
"learning_rate": 0.1,
"bagging_fraction": 0.9,
"feature_fraction": 0.9,
"verbosity": -1}

train_data = lgb.Dataset(X, label=y)
model = lgb.train(params, train_data)

filename = 'model2.sav'
pickle.dump(model, open(filename, 'wb'))


import xgboost as xgb

xgbr = xgb.XGBRegressor(verbosity=0, booster='gbtree', learning_rate=0.15, n_estimators=300) 
xgbr.fit(X, y)

filename = 'model3.sav'
pickle.dump(xgbr, open(filename, 'wb'))


