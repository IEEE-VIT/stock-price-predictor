import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("AAPL.csv") 

data['Date'] = pd.to_datetime(data['Date']) 
data = data.dropna(subset=['Date', 'Close'])
data = data.sort_values('Date')

data['Day'] = range(1, len(data) + 1) 

X = data['Day'].values.reshape(-1, 1)
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

test_preds = model.predict(X_test)

mae = mean_absolute_error(y_test, test_preds)
print(f"Model Mean Absolute Error on Test Data: ${mae:.2f}")

future_days = np.arange(len(data)+1, len(data)+6).reshape(-1,1)
preds = model.predict(future_days)

print("Predictions for next 5 days:", preds)

plt.scatter(X, y, color="blue")
plt.plot(future_days, preds, color="red")
plt.show()
