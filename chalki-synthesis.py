
# ----------------------------------------------------------------------
# ## Consumption Data

# ======================================================================
import pandas as pd
import numpy as np
np.random.seed(69)
pd.set_option('display.float_format', '{:.2f}'.format)
from datetime import time


# Load data
def load_data():
    data = pd.read_csv('./data/tilos_hourly.csv')
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['value'] = data['value'].astype(np.float32)
    # Rename columns
    data = data.rename(columns={'datetime': 'datetime', 'value': 'consumption'})
    return data

df = load_data()

# Filter data for January, February, and March
df = df[df['datetime'].dt.month.isin([1, 2, 3])]
df.reset_index(drop=True, inplace=True)
df


# ----------------------------------------------------------------------
# ## Synthetic Production Data

# ======================================================================
# Approximate sunrise and sunset times for each month
sunrise_times = {1: 7, 2: 7, 3: 6}   # January, February, March
sunset_times = {1: 17, 2: 18, 3: 19} # January, February, March

# Add a date column
df['date'] = df['datetime'].dt.date

# Calculate daily total consumption
daily_consumption = df.groupby('date')['consumption'].sum().reset_index()

def generate_daily_pv_production(row):
    date = row['date']
    total_consumption = row['consumption']
    month = date.month  # date is a datetime.date object
    sunrise = sunrise_times.get(month, 7)  # Default to 7 AM if month not found
    sunset = sunset_times.get(month, 17)   # Default to 5 PM if month not found

    # Define daylight hours
    daylight_hours = np.arange(sunrise, sunset + 1)
    num_hours = len(daylight_hours)
    
    # Generate a sine wave to simulate solar production
    x = np.linspace(0, np.pi, num_hours)  # Values from 0 to π
    sine_values = np.sin(x)
    
    # Ensure no negative values
    sine_values[sine_values < 0] = 0
    
    # Calculate target PV production (60-70% of daily consumption)
    target_pv_total = total_consumption * np.random.uniform(0.6, 0.7)
    
    # Scale sine wave to match the target PV production
    pv_values = sine_values / sine_values.sum() * target_pv_total
    
    # Create a DataFrame for the PV production
    pv_df = pd.DataFrame({
        'datetime': [pd.Timestamp.combine(date, time(hour=int(h))) for h in daylight_hours],
        'pv_production': pv_values
    })
    return pv_df

# Generate PV production data for all days
pv_dfs = daily_consumption.apply(generate_daily_pv_production, axis=1)
pv_data = pd.concat(pv_dfs.tolist(), ignore_index=True)

# Merge PV production data into the main DataFrame
df = df.merge(pv_data, on='datetime', how='left')

# Fill NaN values (night hours) with 0
df['pv_production'] = df['pv_production'].fillna(0)

df['consumption'] = df['consumption'].round(2)
df['pv_production'] = df['pv_production'].round(2)

# Preview the updated DataFrame
df


# ======================================================================
import matplotlib.pyplot as plt

def plot_consumption_production(df, days=None, start_date=None):
    """
    Plots consumption and production data.
    
    Parameters:
    df (DataFrame): The DataFrame containing 'datetime', 'consumption', and 'pv_production'.
    days (int, optional): Number of days to plot. If None, plot the whole DataFrame.
    start_date (str or Timestamp, optional): The start date for plotting. If None, use the first date in the DataFrame.
    """
    # Convert 'datetime' to datetime if not already
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Set the start date
    if start_date is None:
        start_date = df['datetime'].min().date()
    else:
        start_date = pd.to_datetime(start_date).date()
    
    # If 'days' is provided, calculate the end date
    if days is not None:
        end_date = start_date + pd.Timedelta(days=days)
        mask = (df['datetime'] >= pd.Timestamp(start_date)) & (df['datetime'] < pd.Timestamp(end_date))
        plot_df = df.loc[mask]
    else:
        plot_df = df.copy()
    
    # Plot the data
    plt.figure(figsize=(14, 7))
    plt.plot(plot_df['datetime'], plot_df['consumption'], label='Consumption (kW/h)')
    plt.plot(plot_df['datetime'], plot_df['pv_production'], label='PV Production (kW/h)')
    plt.xlabel('Time')
    plt.ylabel('Energy (kW/h)')
    plt.title(f'Consumption and PV Production from {start_date} for {days if days else "all"} days')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ======================================================================
plot_consumption_production(df, days=3)

# ----------------------------------------------------------------------
# ## EV Data

# ======================================================================
# Step 4: Load and Synthesize EV Charging Data

import pandas as pd
import numpy as np
import re

# Load EV charging data
ev_data = pd.read_csv('./data/reservations.csv')

# Keep only the necessary columns
ev_data = ev_data[['Consumption (in kWh)', 'Start Time', 'End Time', 'Duration(minutes)']]

# Option 1: Remove the Timezone Information
# Remove the ' (UTC+0200)' from 'Start Time' and 'End Time'
ev_data['Start Time'] = ev_data['Start Time'].str.replace(r'\s*\(UTC[^\)]*\)', '', regex=True)
ev_data['End Time'] = ev_data['End Time'].str.replace(r'\s*\(UTC[^\)]*\)', '', regex=True)

# Parse the datetime
ev_data['Start Time'] = pd.to_datetime(ev_data['Start Time'])
ev_data['End Time'] = pd.to_datetime(ev_data['End Time'])

# Alternatively, use Option 2 if you need to handle the timezones properly.

# Calculate the time difference to shift
min_ev_date = ev_data['Start Time'].min()
target_start_date = pd.Timestamp('2021-01-01')
time_shift = target_start_date - min_ev_date

# Shift the dates
ev_data['Start Time'] = ev_data['Start Time'] + time_shift
ev_data['End Time'] = ev_data['End Time'] + time_shift

# Filter the data to the desired period
ev_data = ev_data[(ev_data['Start Time'] >= '2021-01-01') & (ev_data['Start Time'] <= '2021-03-31')]
ev_data.sort_values(by='Start Time', ascending=True, inplace=True)
ev_data.reset_index(drop=True, inplace=True)


# Preview the data
ev_data


# ----------------------------------------------------------------------
# ## Aggregation & Preparation

# ======================================================================
# Step 5: Simulate EV Charging Times

# Define flexibility window in hours
flexibility_window = pd.Timedelta(hours=2)

# Simulate arrival and departure times
ev_data['Arrival Time'] = ev_data['Start Time'] - flexibility_window
ev_data['Departure Time'] = ev_data['End Time'] + flexibility_window

# Ensure times are within the same day
ev_data['Arrival Time'] = ev_data['Arrival Time'].apply(lambda x: max(x, x.normalize()))
ev_data['Departure Time'] = ev_data['Departure Time'].apply(lambda x: min(x, x.normalize() + pd.Timedelta(days=1)))

# Calculate charging duration in hours
ev_data['Charging Duration (hours)'] = ev_data['Duration(minutes)'] / 60

# Preview the data
ev_data[['Start Time', 'End Time', 'Arrival Time', 'Departure Time', 'Charging Duration (hours)']].head()


# ======================================================================
# Function to create an hourly EV load profile for each charging session
def create_ev_load_profile(row):
    start = row['Start Time']
    end = row['End Time']
    consumption = row['Consumption (in kWh)']
    
    # Generate hourly timestamps between start and end
    # Remove the 'closed' parameter
    time_range = pd.date_range(start=start.floor('H'), end=end.ceil('H') - pd.Timedelta(hours=1), freq='H')

    num_hours = len(time_range)
    
    if num_hours == 0:
        # If duration is less than an hour, assign all consumption to the start hour
        time_range = [start.floor('H')]
        num_hours = 1
    
    # Distribute consumption evenly across the hours
    consumption_per_hour = consumption / num_hours
    
    return pd.DataFrame({
        'datetime': time_range,
        'ev_consumption': [consumption_per_hour] * num_hours
    })

# Apply the function to all EV charging sessions
ev_load_profiles = ev_data.apply(create_ev_load_profile, axis=1)

# Concatenate all load profiles
ev_load_data = pd.concat(ev_load_profiles.tolist(), ignore_index=True)

# Aggregate EV consumption by hour
ev_load_data = ev_load_data.groupby('datetime')['ev_consumption'].sum().reset_index()


# ======================================================================
ev_load_data

# ======================================================================
ev_data

# ======================================================================
# Merge the EV load data into the main DataFrame
df = df.merge(ev_load_data, on='datetime', how='left')

# Fill NaN values with 0
df['ev_consumption'] = df['ev_consumption'].fillna(0)

# Round the values
df['ev_consumption'] = df['ev_consumption'].round(2)


# ======================================================================
df

# ----------------------------------------------------------------------
# ## Optimization

# ----------------------------------------------------------------------
# We will formulate the problem as a linear programming (LP) optimization problem:
# 
# Objective:
# 
#     Maximize the use of PV production for EV charging.
# 
# Variables:
# 
#     x_{i,t}: Binary variable indicating whether EV i is charging at time t.
# 
# Constraints:
# 
#     Charging Duration Constraint: For each EV, the sum over time slots t of x_{i,t} equals the required charging duration in hours.
#     ∑t∈Tixi,t=Di
#     t∈Ti​∑​xi,t​=Di​
# 
#     where:
#         TiTi​: Available time slots for EV i between Arrival Time and Departure Time.
#         DiDi​: Charging duration required for EV i.
# 
#     Time Window Constraint: x_{i,t} can only be non-zero during the available time slots.
# 
#     Power Constraint: Total charging power at any time t does not exceed the maximum available PV power.

# ======================================================================
from pulp import LpProblem, LpVariable, LpInteger, LpMinimize, LpMaximize, lpSum, LpStatus


# ======================================================================
# Create a time index for the optimization period
start_datetime = df['datetime'].min()
#end_datetime = df['datetime'].max()
end_datetime = start_datetime + pd.Timedelta(weeks=1)
time_index = pd.date_range(start=start_datetime, end=end_datetime, freq='H')

# Create a dictionary of time slots
time_slots = list(time_index)

# Prepare EV sessions with available time slots
ev_sessions = []
for idx, row in ev_data.iterrows():
    ev_id = idx  # Use the index as the EV identifier
    arrival = row['Arrival Time']
    departure = row['Departure Time']
    duration = row['Charging Duration (hours)']
    
    # Get available time slots for this EV
    available_slots = [t for t in time_slots if arrival <= t < departure]
    
    # Store the session information
    ev_sessions.append({
        'ev_id': ev_id,
        'available_slots': available_slots,
        'duration': duration
    })


# ======================================================================
ev_sessions

# ======================================================================
# Initialize the LP problem
prob = LpProblem("EV_Charging_Scheduling", LpMaximize)


# ======================================================================
# Create decision variables
x = {}

# For each EV session
for session in ev_sessions:
    ev_id = session['ev_id']
    for t in session['available_slots']:
        x[ev_id, t] = LpVariable(f"x_{ev_id}_{t}", cat='Binary')


# ======================================================================
# Create a dictionary of PV production
pv_dict = df.set_index('datetime')['pv_production'].to_dict()

# Since we have PV production per hour, ensure all time slots are covered
pv_power = {t: pv_dict.get(t, 0) for t in time_slots}

# Assume maximum charging power per EV
max_charging_power = 7  # kW


# ======================================================================
pv_dict

# ======================================================================
# Objective: Maximize the use of PV energy for EV charging
prob += lpSum([x[ev_id, t] * pv_power[t] * max_charging_power for (ev_id, t) in x])


# ======================================================================
for t in time_slots:
    # Sum of charging power at time t
    total_charging_power = lpSum([x[ev_id, t] * max_charging_power for ev_id in range(len(ev_sessions)) if (ev_id, t) in x])
    
    # PV production at time t
    pv_available = pv_power[t]
    
    # Constraint: Total charging power <= PV production
    prob += total_charging_power <= pv_available, f"Power_Constraint_{t}"


# ======================================================================
# Solve the problem
status = prob.solve()

# Check the status
print("Status:", LpStatus[prob.status])


# ======================================================================
# Create a DataFrame to store the optimized charging schedule
ev_schedule = []

for (ev_id, t), variable in x.items():
    if variable.varValue > 0:
        ev_schedule.append({
            'ev_id': ev_id,
            'datetime': t,
            'charging': variable.varValue * max_charging_power  # kW
        })

ev_schedule_df = pd.DataFrame(ev_schedule)


# ======================================================================
ev_schedule_df

# ======================================================================
# Aggregate the EV charging load per time slot
ev_optimized_load = ev_schedule_df.groupby('datetime')['charging'].sum().reset_index()

# Merge the optimized EV load into the main DataFrame
df = df.merge(ev_optimized_load, on='datetime', how='left')

# Fill NaN values with 0
df['optimized_ev_consumption'] = df['charging'].fillna(0)

# Drop the 'charging' column
df = df.drop(columns=['charging'])

# Round the values
df['optimized_ev_consumption'] = df['optimized_ev_consumption'].round(2)


# ======================================================================
df

# ======================================================================
def plot_consumption_production_ev(df, days=None, start_date=None):
    # Convert 'datetime' to datetime if not already
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Set the start date
    if start_date is None:
        start_date = df['datetime'].min().date()
    else:
        start_date = pd.to_datetime(start_date).date()
    
    # If 'days' is provided, calculate the end date
    if days is not None:
        end_date = start_date + pd.Timedelta(days=days)
        mask = (df['datetime'] >= pd.Timestamp(start_date)) & (df['datetime'] < pd.Timestamp(end_date))
        plot_df = df.loc[mask]
    else:
        plot_df = df.copy()
    
    # Plot the data
    plt.figure(figsize=(14, 7))
    plt.plot(plot_df['datetime'], plot_df['consumption'], label='Consumption (kW/h)')
    plt.plot(plot_df['datetime'], plot_df['pv_production'], label='PV Production (kW/h)')
    plt.plot(plot_df['datetime'], plot_df['ev_consumption'], label='Original EV Consumption (kW/h)', linestyle='--')
    plt.plot(plot_df['datetime'], plot_df['optimized_ev_consumption'], label='Optimized EV Consumption (kW/h)', linestyle='-.')
    plt.xlabel('Time')
    plt.ylabel('Energy (kW/h)')
    plt.title(f'Consumption, PV Production, and EV Consumption from {start_date} for {days if days else "all"} days')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ======================================================================
# Plot the first week
plot_consumption_production_ev(df, days=7)


# ======================================================================

