import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sys # Import sys to allow for a clean exit

# --- 1. Robust Data Loading and Validation ---
try:
    data = pd.read_csv("AAPL.csv")

    # WHY: First, check if the columns you need actually exist.
    # This prevents a KeyError if 'Date' or 'Close' is missing.
    REQUIRED_COLUMNS = {'Date', 'Close'}
    if not REQUIRED_COLUMNS.issubset(data.columns):
        print(f"Error: CSV file must contain the following columns: {REQUIRED_COLUMNS}")
        sys.exit() # Stop the script if columns are missing.

    # WHY: Use errors='coerce' to turn any unreadable dates into NaT (Not a Time)
    # instead of crashing the script.
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

    # WHY: Now, dropna will remove rows with missing 'Close' values (NaN) AND
    # rows with the bad dates we just converted to NaT.
    data = data.dropna(subset=['Date', 'Close'])
    
    # WHY: After cleaning, the DataFrame might be empty. If so, we can't proceed.
    # This check prevents errors in the modeling steps below.
    if data.empty:
        print("Error: No valid data remaining after cleaning. Cannot create a model.")
        sys.exit()

    data = data.sort_values('Date')

# WHY: This 'except' block catches the specific error if "AAPL.csv" is not found.
except FileNotFoundError:
    print("Error: 'AAPL.csv' not found. Please ensure the file is in the correct directory.")
    sys.exit()
# WHY: This is a general catch-all for any other unexpected pandas/numpy errors.
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit()


# --- 2. Model Training and Prediction (No changes needed here) ---
# This part of the code will now only run if the data is successfully loaded and cleaned.

print("Data loaded and validated successfully. Proceeding with model training...")

data['Day'] = range(1, len(data) + 1)

X = data[['Day']]
y = data['Close']

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(len(data)+1, len(data)+6).reshape(-1,1)
preds = model.predict(future_days)


# --- 3. Output and Visualization (No changes needed here) ---

print("\nPredictions for next 5 days:", preds)

plt.figure(figsize=(10, 6)) # Make the plot a bit bigger
plt.scatter(data['Day'], y, color="blue", label="Historical Data")
plt.plot(future_days[:,0], preds, color="red", marker='o', linestyle='--', label="Forecast")
plt.title("AAPL Stock Price Forecast")
plt.xlabel("Day Sequence")
plt.ylabel("Close Price (USD)")
plt.legend()
plt.grid(True)
plt.show()