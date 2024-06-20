import collections
import tensorflow as tf

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

NUM_EPOCHS = 10
SHUFFLE_BUFFER = 100
BATCH_SIZE = 32
PREFETCH_BUFFER = 2


def train_test_split_and_scale(df, train_size=0.8, sort_by=['DHH']):
    """
    Splits a DataFrame into training and testing sets based on the given train_size percentage.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame to split.
    train_size (float): The percentage of data to allocate for training (default is 0.8).
    sort_by (list): List of columns to sort the DataFrame by (default is ['DHH']).
    
    Returns:
    pandas.DataFrame, pandas.DataFrame: The training and testing sets.
    """
    # Sort the DataFrame by the specified columns to ensure data is in order
    df = df.sort_values(by='DHH')

    # Calculate the index to split the data
    split_index = int(len(df) * train_size)

    # Split the data into training and testing sets
    train_data = df[:split_index]
    test_data = df[split_index:]

    # Reset the index and further sort train_data if needed
    train_data = train_data.reset_index(drop=True)
    #train_data.sort_values(['POD', 'DHH'], inplace=True)

    scaler = MinMaxScaler()
    train_data['kW/h_scaled'] = scaler.fit_transform(train_data[['kW/h']])
    test_data['kW/h_scaled'] = scaler.transform(test_data[['kW/h']])
    train_data.drop(columns=['kW/h'], inplace=True)
    test_data.drop(columns=['kW/h'], inplace=True)
    

    return train_data, test_data, scaler



def create_sliding_window(client_data, window_size, step_size = 1):
    client_windows = []
    client_targets = []
    num_readings = len(client_data)

    # Iterate over the readings using the sliding window
    for i in range(0, num_readings - window_size, step_size):
        window_start = i
        window_end = i + window_size - 1
        prediction_index = window_end + step_size

        # Extract the window and the prediction target
        window = client_data.iloc[window_start:window_end + 1]['kW/h_scaled'].values
        target = client_data.iloc[prediction_index]['kW/h_scaled']

        # Reshape the window to add an extra dimension for features
        window = np.reshape(window, (window.shape[0], 1))

        client_windows.append(window)
        client_targets.append(target)

    return np.array(client_windows), np.array(client_targets)


def align_datetime_with_predictions(test_data, predictions, window_size, step_size):
    """
    Adjust the datetime extraction to match the length of predictions.
    
    Parameters:
    - test_data: DataFrame containing the test dataset with a 'DHH' datetime column.
    - predictions: The predictions array, used to determine the required length.
    - window_size: The size of the window used for each prediction.
    - step_size: The step size used when sliding the window.
    
    Returns:
    - A pandas Series or array of datetime values aligned with the predictions.
    """
    # Calculate the start index for alignment based on window and step size
    start_index = window_size - 1
    # Calculate the end index based on the number of predictions and step size
    end_index = start_index + len(predictions) * step_size
    # Extract and return the aligned datetime values
    aligned_datetimes = test_data['DHH'].iloc[start_index:end_index:step_size].reset_index(drop=True)
    return aligned_datetimes


def plot_predictions(actual_values_uns, predicted_values_uns, client_id, aligned_datetimes, subset_range=(6000, 8000)):
    """
    Plots the full range and a subset of predictions against actual values side by side.

    Parameters:
    - datetime_values: Array of datetime values corresponding to each prediction/actual value
    - actual_values_uns: Unscaled actual values
    - predicted_values_uns: Unscaled predicted values
    - client_id: Identifier for the client (used in plot titles)
    - subset_range: Tuple indicating the start and end indices for the subset plot
    """
    start_point, end_point = subset_range

    # Ensure datetime_values is a numpy array or similar for easy indexing
    datetime_values = np.array(aligned_datetimes)
    
    # Setup for side-by-side plotting
    fig, axs = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
    fig.suptitle(f'Model Predictions vs Actual Values for {client_id}')

    # Full Range Plot
    
    axs[0].plot(datetime_values, actual_values_uns, label='Actual Values', color='deepskyblue')
    axs[0].plot(datetime_values, predicted_values_uns, label='Predictions', color='deeppink')
    axs[0].set_title('Full Range')
    axs[0].set_xlabel('Datetime')
    axs[0].set_ylabel('kW/h')
    axs[0].legend()

    # Subset 
    # axs[1].plot(datetime_values[start_point:end_point], actual_values_uns[start_point:end_point], label='Actual Values', color='deepskyblue')
    # axs[1].plot(datetime_values[start_point:end_point], predicted_values_uns[start_point:end_point], label='Predictions', color='deeppink')    
    # axs[1].set_title(f'Subset (Samples: {start_point} to {end_point})')
    # axs[1].set_xlabel('Datetime')
    # axs[1].legend()

    # Convert numpy.datetime64 to pandas.Timestamp for formatting, if necessary
    start_date_formatted = pd.to_datetime(datetime_values[start_point]).strftime('%Y-%m-%d')
    end_date_formatted = pd.to_datetime(datetime_values[end_point]).strftime('%Y-%m-%d')

    axs[1].plot(datetime_values[start_point:end_point], actual_values_uns[start_point:end_point], label='Actual Values', color='deepskyblue')
    axs[1].plot(datetime_values[start_point:end_point], predicted_values_uns[start_point:end_point], label='Predictions', color='deeppink')    
    axs[1].set_title(f'Subset samples from: {start_date_formatted} to: {end_date_formatted}')
    axs[1].set_xlabel('Datetime')
    axs[1].legend()
    

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect to make room for the suptitle
    plt.show()



def compute_basic_metrics_and_save(metrics_df, notebook_name):
    """
    Computes basic statistics (min, max, mean) for the metrics in the DataFrame and saves the results to a CSV file.
    
    Parameters:
    - metrics_df: DataFrame containing metrics for each client.
    - notebook_name: String representing the notebook name, used in the output file name.
    - save_location: Directory path where the CSV file will be saved.
    """
    stats = pd.DataFrame(index=['Min', 'Max', 'Mean'], columns=metrics_df.columns[1:])  # Exclude 'Client' column for stats
    
    for col in stats.columns:  # Iterate over metrics columns, excluding 'Client'
        stats.at['Min', col] = metrics_df[col].min()
        stats.at['Max', col] = metrics_df[col].max()
        stats.at['Mean', col] = round(metrics_df[col].mean(), 5)

    parent_folder = '/home/verxus/diplo/publication/results/LL'
    
    # Ensure the save location exists
    os.makedirs(parent_folder, exist_ok=True)
    
    # Construct file path
    metrics_file_name = f"{notebook_name}_metrics.csv"
    metrics_full_file_path = os.path.join(parent_folder, metrics_file_name)
    
    stats_file_name = f"{notebook_name}_stats.csv"
    stats_full_file_path = os.path.join(parent_folder, stats_file_name)

    # Save the statistics & metrics DataFrame to CSV
    metrics_df.to_csv(metrics_full_file_path, index=False)
    stats.to_csv(stats_full_file_path, index=True)
    
    
    print(f"Metrics summary saved to: {metrics_full_file_path}")
    print(stats)


def compute_metrics_statistics_with_client(metrics_df):
    # Initialize a DataFrame to hold the stats and corresponding client IDs
    stats = pd.DataFrame(columns=['Metric', 'Min Value', 'Min Client', 'Max Value', 'Max Client', 'Mean Value'])
    
    # Iterate over each metric column to calculate stats
    for col in ['Train Loss', 'Train MAE', 'Train RMSE', 'Test Loss', 'Test MAE', 'Test RMSE']:
        min_value = metrics_df[col].min()
        max_value = metrics_df[col].max()
        mean_value = metrics_df[col].mean()
        
        # Identify the clients with min and max values
        min_client = metrics_df.loc[metrics_df[col] == min_value, 'Client'].iloc[0]
        max_client = metrics_df.loc[metrics_df[col] == max_value, 'Client'].iloc[0]
        
        # Append the stats for this metric to the DataFrame
        stats = stats.append({'Metric': col, 
                              'Min Value': min_value, 'Min Client': min_client,
                              'Max Value': max_value, 'Max Client': max_client,
                              'Mean Value': mean_value}, ignore_index=True)
    
    return stats

