# train_model_xgb.py

import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

def load_csv_data():
    # Load CSV data
    df_csv = pd.read_csv('SOL_USD_daily_data.csv')
    df_csv['Date'] = pd.to_datetime(df_csv['Date'])
    return df_csv

def fetch_api_data():
    # Fetch last 60 days of data from CoinGecko API
    url = "https://api.coingecko.com/api/v3/coins/solana/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '60',
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Create DataFrame from the fetched data
    df_prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df_volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])

    # Merge DataFrames on timestamp
    df_api = df_prices.merge(df_volumes, on='timestamp')

    # Convert timestamp to datetime
    df_api['Date'] = pd.to_datetime(df_api['timestamp'], unit='ms')

    # Drop timestamp column
    df_api.drop('timestamp', axis=1, inplace=True)

    return df_api

def load_data():
    # Load CSV data
    df_csv = load_csv_data()

    # Fetch API data
    df_api = fetch_api_data()

    # Ensure no overlapping dates
    max_csv_date = df_csv['Date'].max()
    min_api_date = df_api['Date'].min()

    # Remove overlapping dates from df_csv if any
    df_csv = df_csv[df_csv['Date'] < min_api_date]

    # Rename columns to match
    df_csv.rename(columns={
        'Close': 'price',
        'Volume': 'volume'
    }, inplace=True)

    # Select necessary columns
    df_csv = df_csv[['Date', 'price', 'volume']]
    df_api = df_api[['Date', 'price', 'volume']]

    # Combine the dataframes
    df_combined = pd.concat([df_csv, df_api], ignore_index=True)

    # Sort by date
    df_combined.sort_values('Date', inplace=True)

    # Reset index
    df_combined.reset_index(drop=True, inplace=True)

    return df_combined

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def preprocess_data(df):
    # Feature Engineering
    df['price_change'] = df['price'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['rolling_mean'] = df['price'].rolling(window=5).mean()
    df['rolling_std'] = df['price'].rolling(window=5).std()
    df['ema_10'] = df['price'].ewm(span=10, adjust=False).mean()
    df['rsi_14'] = compute_rsi(df['price'], window=14)

    # Drop rows with NaN values resulting from calculations
    df.dropna(inplace=True)

    # Define features and target
    features = [
        'price_change', 'volume_change', 'rolling_mean', 'rolling_std',
        'ema_10', 'rsi_14'
    ]
    X = df[features]
    y = df['price']

    return X, y, df

def create_multistep_dataset(X, y, n_out=60):
    Xs, ys = [], []
    for i in range(len(X) - n_out + 1):
        Xs.append(X.iloc[i].values)
        ys.append(y.iloc[i:i + n_out].values)
    return np.array(Xs), np.array(ys)

def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Evaluation on Test Set:")
    print(f" - Mean Squared Error: {mse:.2f}")
    print(f" - Mean Absolute Error: {mae:.2f}")
    print(f" - R-squared Score: {r2:.2f}")

def forecast_future(model, X_last, periods=60):
    # Forecast the next 'periods' steps
    future_predictions = model.predict(X_last)
    # Assuming last known date
    last_date = df_with_features['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)

    df_future = pd.DataFrame({
        'Date': future_dates,
        'price': future_predictions.flatten()
    })
    return df_future

def main():
    df = load_data()
    X, y, df_with_features = preprocess_data(df)

    # Create multi-step dataset
    n_out = 60  # Number of future steps to predict
    X_ms, y_ms = create_multistep_dataset(X, y, n_out=n_out)

    # Split data into training and test sets
    split_index = int(len(X_ms) * 0.8)
    X_train, X_test = X_ms[:split_index], X_ms[split_index:]
    y_train, y_test = y_ms[:split_index], y_ms[split_index:]

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    joblib.dump(model, 'xgb_model.pkl')
    print("Model saved to xgb_model.pkl")

    # Forecast future
    X_last = X.tail(1).values  # Use the last row of features
    df_future = forecast_future(model, X_last, periods=n_out)
    df_future.to_csv('xgb_forecast.csv', index=False)
    print("Future predictions saved to xgb_forecast.csv")

if __name__ == '__main__':
    main()
