import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

    # Debugging: Print info about the dataframe
    print("Initial data info:")
    print(df.info())
    print("Initial data description:")
    print(df.describe())

    # Drop rows with missing values
    df.dropna(axis=0, inplace=True)

    # Convert date to datetime and set it as the index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Debugging: Check for NaN values and first few rows
    print("Data after dropping NaN values:")
    print(df.isna().sum())
    print(df.head())

    # Convert the dataset into numpy array
    data = np.array(df)

    # Scale the dataset
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Debugging: Check the scaled data
    print("Scaled data:")
    print(data[:5])
    
    return data, scaler, df.index

def create_sequences(data, sequence_length):
    X = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
    
    # Debugging: Print the shape of the sequences
    print(f"Created sequences with shape: {np.array(X).shape}")
    
    return np.array(X)

def generate_future_predictions(model, initial_sequence, num_predictions):
    predictions = []
    current_sequence = initial_sequence

    for _ in range(num_predictions):
        prediction = model.predict(current_sequence[np.newaxis, :, :])
        predictions.append(prediction.flatten())
        current_sequence = np.append(current_sequence[1:], prediction, axis=0)

    # Debugging: Print the shape and first few predictions
    print(f"Generated predictions with shape: {np.array(predictions).shape}")
    print(f"First few predictions: {predictions[:5]}")
    
    return np.array(predictions)

def generate_future_dates(last_date, num_days):
    return pd.date_range(start=last_date, periods=num_days + 1, inclusive='right')
