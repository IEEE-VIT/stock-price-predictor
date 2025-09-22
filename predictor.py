import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("AAPL.csv") 
data['Date'] = pd.to_datetime(data['Date']) 
data = data.dropna(subset=['Date', 'Close'])
data = data.sort_values('Date')

data['Day'] = range(1, len(data) + 1) 

X = data[['Day']]
y = data['Close']

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(data) + 1, len(data) + 6).reshape(-1, 1)
preds = model.predict(future_days)

print("Predictions for next 5 days:", preds)

plt.scatter(X, y, color="blue", label="Actual")
plt.plot(future_days, preds, color="red", marker='o', label="Predicted")
plt.xlabel('Day')
plt.ylabel('Close Price')
plt.title('AAPL Stock Price Prediction for Next 5 Days')
plt.legend()
plt.show()
