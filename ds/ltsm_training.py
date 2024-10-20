# lstm_training.py

import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def load_csv_data():
    # Load historical CSV data
    df_csv = pd.read_csv('SOL_USD_daily_data.csv')
    df_csv['Date'] = pd.to_datetime(df_csv['Date'])
    return df_csv

def fetch_api_data():
    # Fetch the last 60 days of data from CoinGecko API
    url = "https://api.coingecko.com/api/v3/coins/solana/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '60',
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    data = response.json()
    df_prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df_volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    df_api = df_prices.merge(df_volumes, on='timestamp')
    df_api['Date'] = pd.to_datetime(df_api['timestamp'], unit='ms')
    df_api.drop('timestamp', axis=1, inplace=True)
    return df_api

def load_data():
    df_csv = load_csv_data()
    df_api = fetch_api_data()
    # Ensure no overlapping dates
    max_csv_date = df_csv['Date'].max()
    min_api_date = df_api['Date'].min()
    df_csv = df_csv[df_csv['Date'] < min_api_date]
    # Rename columns to match
    df_csv.rename(columns={'Close': 'price', 'Volume': 'volume'}, inplace=True)
    df_csv = df_csv[['Date', 'price', 'volume']]
    df_api = df_api[['Date', 'price', 'volume']]
    # Combine the dataframes
    df_combined = pd.concat([df_csv, df_api], ignore_index=True)
    # Sort by date
    df_combined.sort_values('Date', inplace=True)
    df_combined.reset_index(drop=True, inplace=True)
    return df_combined

def compute_technical_indicators(df):
    # Exponential Moving Average (EMA)
    df['ema_10'] = df['price'].ewm(span=10, adjust=False).mean()
    # Relative Strength Index (RSI)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    return df

def preprocess_data(df):
    # Compute technical indicators
    df = compute_technical_indicators(df)
    # Drop rows with NaN values resulting from calculations
    df.dropna(inplace=True)
    # Set 'Date' as index
    df.set_index('Date', inplace=True)
    # Select features and target
    features = ['price', 'volume', 'ema_10', 'rsi_14']
    data = df[features].values
    # Scale the data to (0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # Predicting 'price'
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_test_split(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=64)))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))
    return model

def compile_model(model, learning_rate=0.001):
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

def train_model(model, X_train, y_train, epochs=100, batch_size=32, validation_data=None):
    # Implement Early Stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=[early_stopping]
    )
    return history

def predict(model, X):
    return model.predict(X)

def inverse_transform(scaler, data, feature_index=0):
    # Create an array of zeros with the same number of features
    expanded_data = np.zeros((len(data), scaler.n_features_in_))
    # Set the target feature to the data
    expanded_data[:, feature_index] = data
    # Inverse transform
    inv_data = scaler.inverse_transform(expanded_data)
    return inv_data[:, feature_index]

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")

def forecast_future_prices(model, last_sequence, seq_length, future_days, scaler):
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(future_days):
        input_seq = current_sequence.reshape((1, seq_length, scaler.n_features_in_))
        pred = model.predict(input_seq)
        predictions.append(pred[0, 0])
        # Update the current sequence by removing the first entry and adding the prediction
        next_sequence = np.zeros((seq_length, scaler.n_features_in_))
        next_sequence[:-1] = current_sequence[1:]
        # Assume other features remain the same (e.g., use the last known values)
        next_sequence[-1] = current_sequence[-1]
        next_sequence[-1, 0] = pred[0, 0]  # Update the 'price' with the prediction
        current_sequence = next_sequence
    predictions = np.array(predictions)
    # Inverse transform the predictions
    predictions = inverse_transform(scaler, predictions)
    return predictions

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    # Hyperparameters
    seq_length = 60  # Adjust sequence length
    test_size = 0.2
    epochs = 100
    batch_size = 32
    learning_rate = 0.001
    future_days = 60  # Number of days to predict into the future

    df = load_data()
    scaled_data, scaler = preprocess_data(df)
    X, y = create_sequences(scaled_data, seq_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    compile_model(model, learning_rate=learning_rate)
    history = train_model(
        model, X_train, y_train,
        epochs=epochs, batch_size=batch_size,
        validation_data=(X_test, y_test)
    )
    # Plot training history
    plot_training_history(history)
    # Evaluate the model
    y_pred_scaled = predict(model, X_test)
    y_pred = inverse_transform(scaler, y_pred_scaled.flatten())
    y_test_actual = inverse_transform(scaler, y_test)
    evaluate(y_test_actual, y_pred)
    # Forecast future prices
    last_sequence = scaled_data[-seq_length:]
    future_predictions = forecast_future_prices(
        model, last_sequence, seq_length, future_days, scaler
    )
    # Prepare future dates
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
    # Create DataFrame for future predictions
    df_future = pd.DataFrame({
        'Date': future_dates,
        'price': future_predictions.flatten()
    })
    df_future.to_csv('lstm_forecast.csv', index=False)
    print("Future predictions saved to lstm_forecast.csv")

if __name__ == '__main__':
    main()
