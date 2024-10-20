import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data():
    data = pd.read_csv('SOL_USD_daily_data.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def preprocess_data(data):
    data['price_change'] = data['Close'].pct_change()
    data['volume_change'] = data['Volume'].pct_change()
    data['rolling_mean'] = data['Close'].rolling(window=5).mean()
    data['rolling_std'] = data['Close'].rolling(window=5).std()
    data.fillna(0, inplace=True)
    X = data[['price_change', 'volume_change', 'rolling_mean', 'rolling_std']]
    y = data['Close']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=2000,
        learning_rate=0.11,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Model Evaluation:")
    print(f" - Mean Squared Error: {mse}")
    print(f" - Mean Absolute Error: {mae}")
    print(f" - R-squared Score: {r2}")

def save_model(model, filename='xgb_model.pkl'):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def main():
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)

if __name__ == '__main__':
    main()
