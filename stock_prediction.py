import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Download stock data
data = yf.download('AAPL', start='2022-01-01', end='2024-12-31')
data = data[['Close']].dropna()

# Prepare data
data['Days'] = np.arange(len(data))  # Turn dates into numbers
X = data[['Days']]
y = data['Close']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
data['Predicted'] = model.predict(X)

# Plot
plt.plot(data['Close'], label='Actual Price')
plt.plot(data['Predicted'], label='Predicted Price', linestyle='--')
plt.title('Simple Stock Price Prediction (AAPL)')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
