import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


number_of_hours = 19008
number_of_10mins = 6 * number_of_hours

demand_10min = pd.read_excel('10_min_demand.xlsx')


# Aggregate the demand in hourly data
demand_10min_array = demand_10min["Consumption"].to_numpy()
demand_hourly_array =  []
for i in range(number_of_hours):
	temp = 0 
	for j in range(6):
		# print(i*6+j)
		temp += demand_10min_array[i*6 + j]
	demand_hourly_array.append(temp / 6)

pd.DataFrame(demand_hourly_array).to_excel("hourly_demand.xlsx")

