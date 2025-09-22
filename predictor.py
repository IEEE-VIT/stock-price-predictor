import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys

# --- 1. Robust Data Loading and Validation (No changes from before) ---
try:
    data = pd.read_csv("AAPL.csv")
    REQUIRED_COLUMNS = {'Date', 'Close'}
    if not REQUIRED_COLUMNS.issubset(data.columns):
        print(f"Error: CSV file must contain the following columns: {REQUIRED_COLUMNS}")
        sys.exit()
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['Date', 'Close'])
    if data.empty:
        print("Error: No valid data remaining after cleaning. Cannot create a model.")
        sys.exit()
    data = data.sort_values('Date')
except FileNotFoundError:
    print("Error: 'AAPL.csv' not found. Please ensure the file is in the correct directory.")
    sys.exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit()

print("Data loaded and validated successfully.")

# --- 2. Prepare Data and Evaluate Model Performance ---
data['Day'] = range(1, len(data) + 1)
X = data[['Day']]
y = data['Close']

# WHY: Split data into training (80%) and testing (20%) sets.
# random_state ensures the split is the same every time you run the script.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# WHY: Train the model ONLY on the training data.
eval_model = LinearRegression()
eval_model.fit(X_train, y_train)

# WHY: Make predictions on the unseen test data.
predictions = eval_model.predict(X_test)

# WHY: Calculate the error. We use Root Mean Squared Error (RMSE).
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("\n--- Model Performance Evaluation ---")
print(f"Root Mean Squared Error (RMSE) on Test Data: ${rmse:.2f}")
print("This means the model's predictions are, on average, off by this amount.")


# --- 3. Train Final Model on ALL Data and Predict Future ---
# WHY: For the best possible forecast, we retrain the model on the entire dataset
# now that we know its approximate performance.
final_model = LinearRegression()
final_model.fit(X, y) # Fit on all data: X, y

future_days = np.arange(len(data)+1, len(data)+6).reshape(-1,1)
future_preds = final_model.predict(future_days)

print("\n--- Future Predictions ---")
print("Predictions for next 5 days:", future_preds)


# --- 4. Visualization ---
plt.figure(figsize=(12, 7))
# Plot historical data
plt.scatter(X_test, y_test, color='blue', label='Actual Test Data', alpha=0.6)
# Plot the model's predictions on the test set
plt.scatter(X_test, predictions, color='orange', label='Model Predictions on Test Data', alpha=0.7)
plt.title("AAPL Stock Price: Model Performance Evaluation")
plt.xlabel("Day Sequence")
plt.ylabel("Close Price (USD)")
plt.legend()
plt.grid(True)
plt.show()