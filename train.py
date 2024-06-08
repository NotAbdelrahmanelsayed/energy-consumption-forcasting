import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import argparse

def read_data(data_path):
    # Read the dataset
    df = pd.read_csv(data_path)

    # Rename features for consistency
    renamed_features = {
        'Date ': 'date',
        'Ambient Temperature': 'ambient_temperature',
        'wind_speed_10m (km/h)': 'wind_speed_10m_kmh',
        'wind_speed_100m (km/h)': 'wind_speed_100m_kmh',
        'Solar Radiation': 'solar_radiation'
    }
    df.rename(columns=renamed_features, inplace=True)

    df.dropna(axis=0, inplace=True)
    # Convert date to datetime and set it as the index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Convert the dataset into numpy array
    data = np.array(df)

    # Scale the dataset
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def main(data_path):
    data = read_data(data_path)

    sequence_length = 24

    # Prepare the entire dataset
    X, y = create_sequences(data, sequence_length)

    # Define the model
    final_model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(sequence_length, 4)),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(4)
    ])

    # Compile the model
    final_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    final_history = final_model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2)

    # Create directory to save models
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the model
    final_model.save('models/final_model.h5')

    # Save the model architecture to JSON
    model_json = final_model.to_json()
    with open("models/final_model.json", "w") as json_file:
        json_file.write(model_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an LSTM model on weather data.')
    parser.add_argument('data_path', type=str, help='Path to the CSV file containing the dataset.')
    args = parser.parse_args()
    main(args.data_path)
