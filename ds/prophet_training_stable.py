import pandas as pd
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
        'days': '60',  # Get data for the last 60 days
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Create DataFrame from the fetched data
    df_prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])

    # Convert timestamp to datetime
    df_prices['Date'] = pd.to_datetime(df_prices['timestamp'], unit='ms')

    # Drop timestamp column
    df_prices.drop('timestamp', axis=1, inplace=True)

    return df_prices

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
    df_csv.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    df_api.rename(columns={'Date': 'ds', 'price': 'y'}, inplace=True)

    # Select necessary columns
    df_csv = df_csv[['ds', 'y']]
    df_api = df_api[['ds', 'y']]

    # Combine the dataframes
    df = pd.concat([df_csv, df_api], ignore_index=True)

    # Sort by date
    df.sort_values('ds', inplace=True)

    return df

def train_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model

def evaluate_prophet_model(model):
    # Use cross-validation to evaluate the model
    print("Evaluating Prophet model performance...")
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='60 days')
    df_p = performance_metrics(df_cv)
    print(df_p[['horizon', 'mse', 'mae', 'rmse', 'mape', 'coverage']])
    # Save performance metrics
    df_p.to_csv('prophet_performance_metrics.csv', index=False)
    print("Performance metrics saved to prophet_performance_metrics.csv")
    
def save_model(model, filename='prophet_model.pkl'):
    with open(filename, 'wb') as f:
        joblib.dump(model, f)
    print(f"Model saved to {filename}")

def main():
    df = load_data()
    model = train_prophet_model(df)
    evaluate_prophet_model(model)
    save_model(model)

if __name__ == '__main__':
    main()
