# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and preprocess the dataset
file_path = '/Users/rishabhjava/Documents/portfolio/quant/dataset'
data = pd.read_csv(file_path, skiprows=2)

# Clean and prepare the data
cleaned_data = data.copy()
# Assuming the date column is the first column but might have a different name
date_column = data.columns[0]
cleaned_data.rename(columns={date_column: 'Date'}, inplace=True)
cleaned_data["Date"] = pd.to_datetime(cleaned_data["Date"])
cleaned_data.set_index("Date", inplace=True)

# Feature engineering
data['ma_10'] = data['Close'].rolling(window=10).mean()
data['ma_50'] = data['Close'].rolling(window=50).mean()

delta = data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14, min_periods=1).mean()
avg_loss = loss.rolling(window=14, min_periods=1).mean()

rs = avg_gain / avg_loss
data['rsi_14'] = 100 - (100 / (1 + rs))

data['ma_20'] = data['Close'].rolling(window=20).mean()
data['bb_upper'] = data['ma_20'] + 2 * data['Close'].rolling(window=20).std()
data['bb_lower'] = data['ma_20'] - 2 * data['Close'].rolling(window=20).std()

ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['macd'] = ema_12 - ema_26
data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

data['roll_std_20'] = data['Close'].pct_change().rolling(window=20).std()
data.dropna(inplace=True)

# Strategy A: Predictive Modeling with Ridge Regression
features = ['ma_10', 'ma_50', 'rsi_14', 'macd', 'macd_signal', 'roll_std_20']
data['return'] = data['Close'].pct_change().shift(-1)  # Predict next-day return
data.dropna(inplace=True)

X = data[features]
y = data['return']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Ridge Regression model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Use the trained model to make predictions on the test set
y_pred = ridge_model.predict(X_test)

# Evaluate model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error

ridge_mse = mean_squared_error(y_test, y_pred)
ridge_mae = mean_absolute_error(y_test, y_pred)

# Display Ridge model performance metrics
print(f"Ridge Regression - Mean Squared Error: {ridge_mse}")
print(f"Ridge Regression - Mean Absolute Error: {ridge_mae}")

# Plot predictions vs actual returns
plt.figure(figsize=(14, 7))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Returns')
plt.ylabel('Predicted Returns')
plt.title('Ridge Regression: Actual vs Predicted Returns')
plt.show()
