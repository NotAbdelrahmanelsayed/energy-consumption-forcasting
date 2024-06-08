import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os
import argparse
from train import read_data

def create_sequences(data, sequence_length):
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
    return np.array(X)

def main(data_path):
    # Load the model architecture from JSON
    with open('models/final_model.json', 'r') as json_file:
        model_json = json_file.read()

    # Recreate the model from the JSON file
    loaded_model = model_from_json(model_json)

    # Load the model weights
    loaded_model.load_weights('models/final_model.h5')

    # Compile the loaded model
    loaded_model.compile(optimizer='adam', loss='mean_squared_error')

    # Read and preprocess the data
    data = read_data(data_path)

    # Assuming sequence_length is the same as used during training
    sequence_length = 24

    # Create sequences from the data
    X = create_sequences(data, sequence_length)

    # Make predictions
    predictions = loaded_model.predict(X)

    # Prepare the results dataframe
    # Read the original dataframe to get the dates
    original_df = pd.read_csv(data_path)
    
    # Rename features for consistency
    renamed_features = {
        'Date ': 'date',
        'Ambient Temperature': 'ambient_temperature',
        'wind_speed_10m (km/h)': 'wind_speed_10m_kmh',
        'wind_speed_100m (km/h)': 'wind_speed_100m_kmh',
        'Solar Radiation': 'solar_radiation'
    }
    original_df.rename(columns=renamed_features, inplace=True)
    original_df['date'] = pd.to_datetime(original_df['date'])
    original_df.set_index('date', inplace=True)

    # Generate future dates based on the last date in the original data
    last_date = original_df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=len(predictions) + 1)

    # Create the results dataframe
    results_df = pd.DataFrame(predictions, index=future_dates, columns=['ambient_temperature', 'wind_speed_10m_kmh', 'wind_speed_100m_kmh', 'solar_radiation'])

    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save the results to a CSV file
    results_df.to_csv('results/predictions.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions using the trained LSTM model.')
    parser.add_argument('data_path', type=str, help='Path to the CSV file containing the dataset.')
    args = parser.parse_args()
    main(args.data_path)
