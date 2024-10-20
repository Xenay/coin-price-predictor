
import pandas as pd
import joblib
import sqlite3
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import requests
import uvicorn
from datetime import datetime, timedelta

# Load the trained model
# def load_model(filename='xgb_model.pkl'):
#     model = joblib.load(filename)
#     return model
def load_prophet_model(filename='prophet_model.pkl'):
    model = joblib.load(filename)
    return model

def load_xgb_model(filename='xgb_model.pkl'):
    model = joblib.load(filename)
    return model

# Create a FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model when the app starts
prophet_model = load_prophet_model()
xgb_model = load_xgb_model()

df_features = pd.read_csv('xgb_forecast.csv')
df_features['Date'] = pd.to_datetime(df_features['Date'])
# Database setup
DB_FILENAME = 'features_cache.db'

def init_db():
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS features_cache (
            date TEXT PRIMARY KEY,
            price_change REAL,
            volume_change REAL,
            rolling_mean REAL,
            rolling_std REAL,
            volume REAL,
            change_24h REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()  # Initialize the database

def get_latest_data():
    url = "https://api.coingecko.com/api/v3/coins/solana/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': '2',  # Get data for the last 2 days
        'interval': 'daily'
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data
def get_last_known_volume():
    # Connect to your database
    conn = sqlite3.connect('features_cache.db')  # Replace with your DB filename
    cursor = conn.cursor()
    
    # Write a query to fetch the latest volume
    cursor.execute('SELECT vol_24h FROM features_cache ORDER BY date DESC LIMIT 1')
    result = cursor.fetchone()
    conn.close()
    
    if result:
        last_volume = result[0]
        return last_volume
    else:
        # Handle the case where no data is found
        raise ValueError("No volume data found in the database.")
def compute_features():
    today_str = datetime.utcnow().strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()

    # Check if data for today exists
    cursor.execute('SELECT * FROM features_cache WHERE date = ?', (today_str,))
    row = cursor.fetchone()

    if row:
        # Data exists, use it
        features = {
            'price_change': row[1],
            'volume_change': row[2],
            'rolling_mean': row[3],
            'rolling_std': row[4],
            'volume': row[5],
            'change_24h': row[6]
        }
        print("Using cached data for", today_str)
    else:
        # Fetch new data
        data = get_latest_data()
        prices = data['prices']
        volumes = data['total_volumes']

        # Extract the latest and previous data points
        latest_price = prices[-1][1]
        prev_price = prices[-2][1]
        latest_volume = volumes[-1][1]
        prev_volume = volumes[-2][1]

        # Compute features
        price_change = (latest_price - prev_price) / prev_price
        volume_change = (latest_volume - prev_volume) / prev_volume
        rolling_mean = pd.Series([prices[i][1] for i in range(-5, 0)]).mean()
        rolling_std = pd.Series([prices[i][1] for i in range(-5, 0)]).std()
        change_24h = price_change * 100  # Convert to percentage
        volume = latest_volume

        features = {
            'price_change': price_change,
            'volume_change': volume_change,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'volume': volume,
            'change_24h': change_24h
        }

        # Save to database
        cursor.execute('''
            INSERT INTO features_cache (
                date, price_change, volume_change, rolling_mean, rolling_std, volume, change_24h
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            today_str,
            features['price_change'],
            features['volume_change'],
            features['rolling_mean'],
            features['rolling_std'],
            features['volume'],
            features['change_24h']
        ))
        conn.commit()
        print("Fetched and cached new data for", today_str)

    conn.close()
    return features

@app.post('/predict_prophet')
async def predict_prophet():
    # Generate future dates for prediction
    future = prophet_model.make_future_dataframe(periods=60)
    forecast = prophet_model.predict(future)

    # Extract the predictions for the next 60 days
    today = datetime.utcnow().date()
    future_dates = [today + timedelta(days=i) for i in range(1, 61)]
    forecast = forecast[forecast['ds'].dt.date.isin(future_dates)]

    # Prepare the data for the frontend
    predictions = forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'predicted_price'})
    predictions['date'] = predictions['date'].astype(str)  # Convert dates to string for JSON serialization

    # Convert to list of dictionaries
    prediction_list = predictions.to_dict(orient='records')

    return {'predictions': prediction_list}

@app.post('/predict_xgb')
async def predict_xgb():
    # Use the precomputed predictions from 'xgb_forecast.csv'
    df_future = pd.read_csv('xgb_forecast.csv')
    df_future.rename(columns={'Date': 'date', 'price': 'predicted_price'}, inplace=True)
    df_future['date'] = df_future['date'].astype(str)  # Ensure date is a string for JSON serialization

    prediction_list = df_future.to_dict(orient='records')

    return {'predictions': prediction_list}

@app.post('/predict_all')
async def predict_all():
    # Prophet predictions
    future = prophet_model.make_future_dataframe(periods=60)

    last_known_volume = get_last_known_volume()
    print(last_known_volume)
    future['volume'] = last_known_volume

    forecast = prophet_model.predict(future)
    today = datetime.utcnow().date()
    future_dates = [today + timedelta(days=i) for i in range(1, 61)]
    forecast = forecast[forecast['ds'].dt.date.isin(future_dates)]
    predictions_prophet = forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'predicted_price'})
    predictions_prophet['date'] = predictions_prophet['date'].astype(str)

    # XGBoost predictions
    df_future = pd.read_csv('xgb_forecast.csv')
    df_future.rename(columns={'Date': 'date', 'price': 'predicted_price'}, inplace=True)
    df_future['date'] = df_future['date'].astype(str)
    #ltsm predictions
    df_lstm_future = pd.read_csv('lstm_forecast.csv')
    df_lstm_future.rename(columns={'Date': 'date', 'price': 'predicted_price'}, inplace=True)
    df_lstm_future['date'] = pd.to_datetime(df_lstm_future['date']).dt.strftime('%Y-%m-%d')
    lstm_predictions = df_lstm_future.to_dict(orient='records')

    return {
        'prophet_predictions': predictions_prophet.to_dict(orient='records'),
        'xgb_predictions': df_future.to_dict(orient='records'),
        'lstm_predictions': lstm_predictions

    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)