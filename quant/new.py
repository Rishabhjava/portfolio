import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import yfinance as yf

def load_and_preprocess_data(file_path, use_yahoo=False):
    if use_yahoo:
        df = yf.download('V', start='2001-01-01', end='2022-12-31')
        df.index = df.index.tz_localize(None)
        df = df.rename(columns={'Adj Close': 'Adj_Close'})
    else:
        df = pd.read_csv(file_path, skiprows=2)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    
    # Sort index and remove duplicates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    # Initial check for NaN values
    print("Initial NaN count:", df.isna().sum())
    
    # Handle missing or infinite values in price data
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close']
    df[price_columns] = df[price_columns].replace([np.inf, -np.inf], np.nan)
    df[price_columns] = df[price_columns].ffill().bfill()
    
    # Handle Volume separately
    median_volume = df['Volume'].median()
    df['Volume'] = df['Volume'].replace([np.inf, -np.inf], np.nan)
    df['Volume'] = df['Volume'].replace(0, np.nan)
    df['Volume'] = df['Volume'].fillna(median_volume)
    
    # Calculate technical indicators
    df['ma_10'] = df['Adj_Close'].rolling(window=10, min_periods=1).mean()
    df['ma_50'] = df['Adj_Close'].rolling(window=50, min_periods=1).mean()
    
    # RSI with error handling
    delta = df['Adj_Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    
    # Avoid division by zero in RSI calculation
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    rs = rs.fillna(0)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_14'] = df['rsi_14'].fillna(50)
    
    # Bollinger Bands
    df['bb_middle'] = df['Adj_Close'].rolling(window=20, min_periods=1).mean()
    df['roll_std_20'] = df['Adj_Close'].rolling(window=20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['roll_std_20']
    df['bb_lower'] = df['bb_middle'] - 2 * df['roll_std_20']
    
    # MACD
    exp1 = df['Adj_Close'].ewm(span=12, adjust=False, min_periods=1).mean()
    exp2 = df['Adj_Close'].ewm(span=26, adjust=False, min_periods=1).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
    
    # Calculate returns
    df['return'] = df['Adj_Close'].pct_change().shift(-1)
    
    # Handle any remaining NaN values
    for column in df.columns:
        if df[column].dtype in [np.float64, np.float32]:
            median_val = df[column].median()
            df[column] = df[column].fillna(median_val)
    
    # Remove outliers from returns
    returns_mean = df['return'].mean()
    returns_std = df['return'].std()
    df = df[abs(df['return'] - returns_mean) <= 3 * returns_std]
    
    # Final verification
    if df.isnull().any().any():
        print("Columns with NaN:", df.columns[df.isnull().any()].tolist())
        print("NaN counts:", df.isnull().sum())
        raise ValueError("NaN values remain after preprocessing")
    
    if np.isinf(df.values).any():
        raise ValueError("Infinite values remain after preprocessing")
    
    print(f"Processed data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {df.columns.tolist()}")
    print("Final NaN check:", df.isna().sum())
    
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def strategy_a_ridge(df):
    # Prepare features
    features = ['ma_10', 'ma_50', 'rsi_14', 'macd', 'macd_signal', 'roll_std_20']
    X = df[features]
    y = df['return']
    
    # Additional data cleaning for model input
    # Check for NaN values before splitting
    print("\nFeature NaN counts before cleaning:")
    print(X.isna().sum())
    
    # Handle any remaining NaN values in features
    for feature in features:
        if X[feature].isna().any():
            median_val = X[feature].median()
            X[feature] = X[feature].fillna(median_val)
    
    # Handle NaN in target variable
    y = y.fillna(0)  # Fill NaN returns with 0
    
    # Verify no NaN values remain
    assert not X.isna().any().any(), "NaN values remain in features after cleaning"
    assert not y.isna().any(), "NaN values remain in target variable after cleaning"
    
    # Convert all dates to tz-naive
    df.index = df.index.tz_localize(None)
    
    # Split dates
    train_end = pd.Timestamp('2022-06-30')
    val_end = pd.Timestamp('2022-12-31')
    
    # Split into train, validation, and test
    X_train = X[df.index <= train_end]
    X_val = X[(df.index > train_end) & (df.index <= val_end)]
    X_test = X[df.index > val_end]
    
    y_train = y[df.index <= train_end]
    y_val = y[(df.index > train_end) & (df.index <= val_end)]
    y_test = y[df.index > val_end]
    
    print(f"\nTraining data: {X_train.index[0]} to {X_train.index[-1]}")
    print(f"Validation data: {X_val.index[0]} to {X_val.index[-1]}")
    print(f"Testing data: {X_test.index[0]} to {X_test.index[-1]}")
    
    # Try different alpha values using validation set
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = None
    best_val_mse = float('inf')
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_pred)
        
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_alpha = alpha
    
    print(f"Best alpha: {best_alpha}")
    
    # Train final model with best alpha using all training data
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    
    # Make predictions
    y_pred = final_model.predict(X_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': abs(final_model.coef_)
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mse, mae, y_test, y_pred

def strategy_b_lstm(df):
    # Prepare data
    features = ['Close', 'ma_10', 'ma_50', 'rsi_14', 'macd', 'macd_signal', 'roll_std_20']
    data = df[features].values
    
    # Convert all dates to tz-naive
    df.index = df.index.tz_localize(None)
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split dates
    train_end = pd.Timestamp('2022-06-30')
    val_end = pd.Timestamp('2022-12-31')
    
    # Calculate split indices
    train_idx = len(df[df.index <= train_end]) - seq_length
    val_idx = len(df[df.index <= val_end]) - seq_length
    
    # Split into train, validation, and test
    X_train = X[:train_idx]
    X_val = X[train_idx:val_idx]
    X_test = X[val_idx:]
    
    y_train = y[:train_idx]
    y_val = y[train_idx:val_idx]
    y_test = y[val_idx:]
    
    # Build LSTM model with early stopping
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, len(features))),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(len(features))
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model with validation data
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Calculate metrics for Close price only
    mse = mean_squared_error(y_test_inv[:, 0], y_pred_inv[:, 0])
    mae = mean_absolute_error(y_test_inv[:, 0], y_pred_inv[:, 0])
    
    return mse, mae, y_test_inv[:, 0], y_pred_inv[:, 0]

def plot_predictions(actual, predicted, title):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load training data from Yahoo Finance
    df_train = load_and_preprocess_data(None, use_yahoo=True)
    print(f"Training data loaded: {df_train.index[0]} to {df_train.index[-1]}")
    
    # Load testing data from CSV
    df_test = load_and_preprocess_data('dataset.csv', use_yahoo=False)
    print(f"Testing data loaded: {df_test.index[0]} to {df_test.index[-1]}")
    
    # Combine datasets
    df = pd.concat([df_train, df_test])
    
    # Strategy A: Ridge Regression
    ridge_mse, ridge_mae, ridge_actual, ridge_pred = strategy_a_ridge(df)
    print("\nRidge Regression Results:")
    print(f"MSE: {ridge_mse:.6f}")
    print(f"MAE: {ridge_mae:.6f}")
    
    # Strategy B: LSTM
    lstm_mse, lstm_mae, lstm_actual, lstm_pred = strategy_b_lstm(df)
    print("\nLSTM Results:")
    print(f"MSE: {lstm_mse:.6f}")
    print(f"MAE: {lstm_mae:.6f}")
    
    # Plot results
    plot_predictions(ridge_actual, ridge_pred, 'Ridge Regression: Actual vs Predicted Returns')
    plot_predictions(lstm_actual, lstm_pred, 'LSTM: Actual vs Predicted Prices')
