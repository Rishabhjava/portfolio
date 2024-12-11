# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

initial_capital = 5000  # Initial investment capital

# Load and preprocess the dataset
file_path = 'full_data.csv'
data = pd.read_csv(file_path, skiprows=2)

# Rename columns and preprocess
data.columns = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace=True)  # Set Date as the index for easier time-based indexing

# Feature Engineering for the Predictive Model (Strategy A)
features = ['ma_10', 'ma_50', 'rsi_14', 'ma_20', 'bb_upper', 'bb_lower', 'macd', 'macd_signal', 'roll_std_20']

# Calculating Features
# 10-day and 50-day moving averages to capture short- and long-term trends
data['ma_10'] = data['Close'].rolling(window=10).mean()
data['ma_50'] = data['Close'].rolling(window=50).mean()

# Calculate RSI (Relative Strength Index) for momentum
# Calculate daily price changes
delta = data['Close'].diff(1)
gain = delta.where(delta > 0, 0)  # Positive gains
loss = -delta.where(delta < 0, 0)  # Negative losses

# Calculate average gain and loss over 14 periods
avg_gain = gain.rolling(window=14, min_periods=1).mean()
avg_loss = loss.rolling(window=14, min_periods=1).mean()

# Calculate RSI using the average gain and average loss
rs = avg_gain / avg_loss
data['rsi_14'] = 100 - (100 / (1 + rs))

# Bollinger Bands to capture volatility
data['ma_20'] = data['Close'].rolling(window=20).mean()  # 20-day moving average
data['bb_upper'] = data['ma_20'] + 2 * data['Close'].rolling(window=20).std()  # Upper Bollinger Band
data['bb_lower'] = data['ma_20'] - 2 * data['Close'].rolling(window=20).std()  # Lower Bollinger Band

# MACD (Moving Average Convergence Divergence) to identify trend changes
ema_12 = data['Close'].ewm(span=12, adjust=False).mean()  # 12-day EMA
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()  # 26-day EMA
data['macd'] = ema_12 - ema_26  # MACD line
data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()  # Signal line

# Rolling 20-day standard deviation of returns as a volatility measure
data['roll_std_20'] = data['Close'].pct_change().rolling(window=20).std()

# Drop NaN values caused by rolling calculations
data.dropna(inplace=True)

# Calculate target variable: next-day return
data['return'] = data['Close'].pct_change().shift(-1)  # Predict next-day return

# Drop NaN values again to remove any remaining missing data
data.dropna(inplace=True)

# Plot a correlation matrix for the features
plt.figure(figsize=(12, 8))
corr_matrix = data[features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()

# Split the dataset into training and testing based on the given dates
# Training data: All data before 2023-01-01
# Testing data: Data from 2023-01-01 to 2024-10-01
train_data = data[(data.index < pd.to_datetime('2023-01-01')) & (data.index > pd.to_datetime('2014-01-01'))]
test_data = data[(data.index >= pd.to_datetime('2023-01-01')) & (data.index <= pd.to_datetime('2024-10-01'))]

# Drop NaN values caused by rolling calculations
train_data = train_data.dropna()

# Extract feature and target arrays for training
X_train = train_data[features]
y_train = train_data['return']

# Standardize the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Split the training data using TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)
model = Ridge(alpha=1.0)  # Ridge regression model to prevent overfitting with L2 regularization

# Train and cross-validate the model
for train_index, val_index in tscv.split(X_train_scaled):
    X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    # Train the model on the current fold
    model.fit(X_train_fold, y_train_fold)
    
    # Validate the model and calculate MSE
    y_pred = model.predict(X_val_fold)
    mse = np.mean((y_pred - y_val_fold) ** 2)
    print(f"Validation MSE for fold: {mse}")

# Train the final model on the entire training dataset
model.fit(X_train_scaled, y_train)

# Prepare the testing dataset for evaluation
X_test = test_data[features]
X_test_scaled = scaler.transform(X_test)  # Scale the test set using the same scaler as the training set

# Create an explicit copy of the test data
test_data = data[(data.index >= pd.to_datetime('2023-01-01')) & (data.index <= pd.to_datetime('2024-10-01'))].copy()

# First model: Ridge with all features
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
test_data.loc[:, 'Predicted_Return'] = model.predict(X_test_scaled)

# Second model: Ridge with reduced features
reduced_features = ['ma_10', 'rsi_14', 'macd', 'roll_std_20']
X_train_reduced = train_data[reduced_features]
X_train_reduced_scaled = scaler.fit_transform(X_train_reduced)
model.fit(X_train_reduced_scaled, y_train)
X_test_reduced = test_data[reduced_features]
X_test_reduced_scaled = scaler.transform(X_test_reduced)
test_data.loc[:, 'Reduced_Feature_Predicted_Return'] = model.predict(X_test_reduced_scaled)

# Third model: Random Forest with all features
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train_scaled, y_train)
test_data.loc[:, 'RF_Predicted_Return'] = rf_model.predict(X_test_scaled)

# Calculate Buy & Hold returns
test_data.loc[:, 'Buy_Hold_Returns'] = test_data['Close'].pct_change()
test_data.loc[:, 'Buy_Hold_Equity'] = initial_capital * (1 + test_data['Buy_Hold_Returns']).cumprod()

# Define threshold values for buy/sell decisions
BUY_THRESHOLD = 0.001  # Buy when predicted return > 0.1%
SELL_THRESHOLD = -0.001  # Sell when predicted return < -0.1%

# Function to calculate position-based returns
def calculate_position_returns(predictions, close_prices):
    position = 0  # -1 for short, 0 for cash, 1 for long
    returns = []
    
    for i in range(len(predictions)):
        # Trading logic
        if predictions.iloc[i] > BUY_THRESHOLD and position <= 0:
            position = 1  # Buy
        elif predictions.iloc[i] < SELL_THRESHOLD and position >= 0:
            position = 0  # Sell to cash (or could be -1 for short)
            
        # Calculate returns based on position
        actual_return = close_prices.pct_change().iloc[i] if i > 0 else 0
        strategy_return = actual_return * position if i > 0 else 0
        returns.append(strategy_return)
    
    return pd.Series(returns, index=close_prices.index)

# Apply the threshold strategy to each model's predictions
test_data.loc[:, 'Ridge_Strategy_Returns'] = calculate_position_returns(
    test_data['Predicted_Return'], 
    test_data['Close']
)

test_data.loc[:, 'Reduced_Feature_Strategy_Returns'] = calculate_position_returns(
    test_data['Reduced_Feature_Predicted_Return'], 
    test_data['Close']
)

test_data.loc[:, 'RF_Strategy_Returns'] = calculate_position_returns(
    test_data['RF_Predicted_Return'], 
    test_data['Close']
)

# Calculate equity curves with the new position-based returns
test_data.loc[:, 'Strategy_Equity'] = initial_capital * (1 + test_data['Ridge_Strategy_Returns']).cumprod()
test_data.loc[:, 'Reduced_Feature_Strategy_Equity'] = initial_capital * (1 + test_data['Reduced_Feature_Strategy_Returns']).cumprod()
test_data.loc[:, 'RF_Strategy_Equity'] = initial_capital * (1 + test_data['RF_Strategy_Returns']).cumprod()

# Calculate performance metrics for all strategies
def calculate_metrics(equity_curve, initial_capital):
    final_equity = equity_curve.iloc[-1]
    cumulative_return = (final_equity / initial_capital) - 1
    annualized_return = ((1 + cumulative_return) ** (1 / (len(equity_curve) / 252))) - 1
    annualized_volatility = equity_curve.pct_change().std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility
    max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()
    return {
        'Final Equity': final_equity,
        'Cumulative Return': cumulative_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

# Metrics for each strategy
metrics_buy_hold = calculate_metrics(test_data['Buy_Hold_Equity'], initial_capital)
metrics_ridge = calculate_metrics(test_data['Strategy_Equity'], initial_capital)
metrics_reduced = calculate_metrics(test_data['Reduced_Feature_Strategy_Equity'], initial_capital)
metrics_rf = calculate_metrics(test_data['RF_Strategy_Equity'], initial_capital)

# Create metrics DataFrame
metrics_df = pd.DataFrame([metrics_buy_hold, metrics_ridge, metrics_reduced, metrics_rf],
                         index=['Buy & Hold', 'Ridge Regression', 'Reduced Feature Ridge', 'Random Forest'])

# Display the metrics table
print(metrics_df)

# Plot all strategies
plt.figure(figsize=(14, 7))
plt.plot(test_data.index, test_data['Buy_Hold_Equity'], label='Buy & Hold Strategy', color='blue')
plt.plot(test_data.index, test_data['Strategy_Equity'], label='Ridge Regression Strategy', color='red', linestyle='--')
plt.plot(test_data.index, test_data['Reduced_Feature_Strategy_Equity'], label='Reduced Feature Strategy', color='green', linestyle='-.')
plt.plot(test_data.index, test_data['RF_Strategy_Equity'], label='Random Forest Strategy', color='purple', linestyle=':')
plt.title('Equity Curve: Comparison of Threshold-Based Strategies')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.show()

# Plot Moving Averages (10-day and 50-day) with Close Price
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Close Price', color='blue')
plt.plot(data.index, data['ma_10'], label='10-Day Moving Average', color='orange', linestyle='--')
plt.plot(data.index, data['ma_50'], label='50-Day Moving Average', color='green', linestyle='--')
plt.title('Visa Stock Price with 10-Day and 50-Day Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()


