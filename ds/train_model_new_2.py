# train_model.py

import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

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
        'days': '60',  # Get data for the last 60 days
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Create DataFrame from the fetched data
    df_prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df_volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    df_market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])

    # Merge DataFrames on timestamp
    df_api = df_prices.merge(df_volumes, on='timestamp').merge(df_market_caps, on='timestamp')

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

    # Rename columns in df_csv to match df_api
    df_csv.rename(columns={
        'Close': 'price',
        'Volume': 'volume'
    }, inplace=True)

    # For consistency, add missing columns to df_csv
    df_csv['market_cap'] = np.nan  # We don't have market cap data in the CSV

    # Select only necessary columns
    df_csv = df_csv[['Date', 'price', 'volume', 'market_cap']]
    df_api = df_api[['Date', 'price', 'volume', 'market_cap']]

    # Combine the dataframes
    df_combined = pd.concat([df_csv, df_api], ignore_index=True)

    # Sort by date
    df_combined.sort_values('Date', inplace=True)

    # Reset index
    df_combined.reset_index(drop=True, inplace=True)

    return df_combined

def preprocess_data(df):
    # Feature Engineering
    df['price_change'] = df['price'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['rolling_mean'] = df['price'].rolling(window=5).mean()
    df['rolling_std'] = df['price'].rolling(window=5).std()
    df['change_24h'] = df['price_change'] * 100  # Convert to percentage

    # Drop rows with NaN values resulting from calculations
    df.dropna(inplace=True)

    # Define features and target
    features = ['price_change', 'volume_change', 'rolling_mean', 'rolling_std', 'volume', 'change_24h']
    X = df[features]
    y = df['price']

    return X, y

def train_model_with_grid_search(X, y):
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    params = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='neg_mean_squared_error', cv=tscv)
    grid_search.fit(X, y)
    print(f"Best parameters found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    return best_model

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    print(f"Model Evaluation:")
    print(f" - Mean Squared Error: {mse:.2f}")
    print(f" - Mean Absolute Error: {mae:.2f}")
    print(f" - R-squared Score: {r2:.2f}")

def main():
    df = load_data()
    X, y = preprocess_data(df)
    model = train_model_with_grid_search(X, y)
    evaluate_model(model, X, y)
    joblib.dump(model, 'xgb_model.pkl')
    print("Model saved to xgb_model.pkl")

if __name__ == '__main__':
    main()
