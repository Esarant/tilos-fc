import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


initial_dataset = pd.read_excel('demand_output.xlsx')

print(initial_dataset)

real = initial_dataset["Real"].to_numpy()
sklearn = initial_dataset["SkLearn"].to_numpy()
xgb = initial_dataset["XGBoost"].to_numpy()
lightgbm = initial_dataset["LightGBM"].to_numpy()
ensemble = initial_dataset["OLD ENSEMBLE"].to_numpy()
mlp = initial_dataset["MLP"].to_numpy()
final = initial_dataset["NEW ENSEMBLE"].to_numpy()




print("r_square", format(r2_score(sklearn, real), ".3f"))
print("MAE", format(mean_absolute_error(sklearn, real), ".3f"))
print("RMSE", format(mean_squared_error(sklearn, real, squared=False), ".3f"))

print("r_square", format(r2_score(xgb, real), ".3f"))
print("MAE", format(mean_absolute_error(xgb, real), ".3f"))
print("RMSE", format(mean_squared_error(xgb, real, squared=False), ".3f"))

print("r_square", format(r2_score(lightgbm, real), ".3f"))
print("MAE", format(mean_absolute_error(lightgbm, real), ".3f"))
print("RMSE", format(mean_squared_error(lightgbm, real, squared=False), ".3f"))

print("r_square", format(r2_score(ensemble, real), ".3f"))
print("MAE", format(mean_absolute_error(ensemble, real), ".3f"))
print("RMSE", format(mean_squared_error(ensemble, real, squared=False), ".3f"))

print("r_square", format(r2_score(mlp, real), ".3f"))
print("MAE", format(mean_absolute_error(mlp, real), ".3f"))
print("RMSE", format(mean_squared_error(mlp, real, squared=False), ".3f"))

print("r_square", format(r2_score(final, real), ".3f"))
print("MAE", format(mean_absolute_error(final, real), ".3f"))
print("RMSE", format(mean_squared_error(final, real, squared=False), ".3f"))