# prophet_training.py

import pandas as pd
import numpy as np
import requests
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
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

    # Rename columns to match Prophet's expected format
    df_csv.rename(columns={'Date': 'ds', 'Close': 'y', 'Volume': 'volume'}, inplace=True)
    df_api.rename(columns={'Date': 'ds', 'price': 'y', 'volume': 'volume'}, inplace=True)

    # Select necessary columns
    df_csv = df_csv[['ds', 'y', 'volume']]
    df_api = df_api[['ds', 'y', 'volume']]

    # Combine the dataframes
    df = pd.concat([df_csv, df_api], ignore_index=True)

    # Sort by date
    df.sort_values('ds', inplace=True)

    return df

def remove_outliers(df):
    # Remove outliers based on z-score
    df['z_score'] = (df['y'] - df['y'].mean()) / df['y'].std()
    df = df[df['z_score'].abs() <= 3]
    df.drop('z_score', axis=1, inplace=True)
    return df

def train_prophet_model(df):
    # Apply log transformation to stabilize variance
    df['y'] = np.log(df['y'])

    # Initialize Prophet model with additional parameters
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    # Add monthly seasonality
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    # Add volume as an external regressor
    model.add_regressor('volume')

    # Fit the model
    model.fit(df)
    return model

def evaluate_prophet_model(model):
    # Use cross-validation to evaluate the model
    print("Evaluating Prophet model performance...")
    df_cv = cross_validation(
        model, initial='730 days', period='180 days', horizon='60 days'
    )
    df_p = performance_metrics(df_cv)
    print(df_p[['horizon', 'mse', 'mae', 'rmse', 'mape', 'coverage']])
    # Save performance metrics
    df_p.to_csv('prophet_performance_metrics.csv', index=False)
    print("Performance metrics saved to prophet_performance_metrics.csv")

def save_model(model, filename='prophet_model_unstable.pkl'):
    with open(filename, 'wb') as f:
        joblib.dump(model, f)
    print(f"Model saved to {filename}")

def main():
    df = load_data()
    df = remove_outliers(df)
    model = train_prophet_model(df)
    evaluate_prophet_model(model)
    save_model(model)

if __name__ == '__main__':
    main()
