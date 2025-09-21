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

future_days = np.arange(len(data)+1, len(data)+6).reshape(-1,1)
preds = model.predict(future_days)

print("Predictions for next 5 days:", preds)

plt.scatter(X, y, color="blue")
plt.plot(future_days["day"], preds, color="red")
plt.show()
