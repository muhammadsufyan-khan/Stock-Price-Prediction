
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

sns.set(style="whitegrid")

# 1. Select stock and load historical data
stock_symbol = "AAPL"
start_date = "2023-01-01"
end_date = "2025-11-01"

df = yf.download(stock_symbol, start=start_date, end=end_date)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# 2. Create next-day Close as target
df['Next_Close'] = df['Close'].shift(-1)
df = df.dropna()

X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Next_Close']

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 4. Train model (Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 7. Visualization
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual Close", color='blue')
plt.plot(y_pred, label="Predicted Close", color='red')
plt.xlabel("Days")
plt.ylabel("Price")
plt.title(f"{stock_symbol} Actual vs Predicted Close Price")
plt.legend()
plt.show()
