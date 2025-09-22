import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

data = pd.read_csv("AAPL.csv") 
data['Date'] = pd.to_datetime(data['Date']) 
data = data.dropna(subset=['Date', 'Close'])
data = data.sort_values('Date')

data['Day'] = range(1, len(data) + 1) 

X = data['Day'].values.reshape(-1, 1)
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
split_idx = int(len(data) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = LinearRegression()
model.fit(X_train, y_train)

test_preds = model.predict(X_test)

mae = mean_absolute_error(y_test, test_preds)
print(f"Model Mean Absolute Error on Test Data: ${mae:.2f}")

y_pred_test = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_test)
print(f"Mean Squared Error on test set: {mse}")

future_days = np.arange(len(data) + 1, len(data) + 6)
future_days = future_days.reshape(-1, 1)
future_preds = model.predict(future_days)

print("Predictions for next 5 days:", future_preds)

plt.scatter(X, y, color="blue")
plt.plot(future_days, preds, color="red")

plt.scatter(X_train, y_train, color="blue", label="Train Data")
plt.scatter(X_test, y_test, color="green", label="Test Data")
plt.plot(future_days, future_preds, color="red", marker='o', label="Future Predictions")
plt.xlabel('Day')
plt.ylabel('Close Price')
plt.title('AAPL Stock Price Prediction with Train/Test Split')
plt.legend()
plt.show()
