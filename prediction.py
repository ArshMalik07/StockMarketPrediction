import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tkinter as tk
from tkinter import messagebox


# Set start and end dates dynamically
end = datetime.now()
start = end - timedelta(days=20 * 365)


def predict_stock_prices():
    stock = stock_entry.get().strip()
    if not stock:
        messagebox.showerror("Error", "Please enter a stock symbol!")
        return

    try:
        stock_data = yf.download(stock, start=start, end=end)
        if stock_data.empty:
            raise ValueError("No data found. Please check the stock symbol or date range.")
    except Exception as e:
        messagebox.showerror("Error", f"Error downloading data: {e}")
        return

    # Handle missing values
    stock_data.fillna(method='ffill', inplace=True)
    stock_data.dropna(inplace=True)

    # Use appropriate column for adjusted prices
    if 'Adj Close' in stock_data.columns:
        Adj_close_price = stock_data[['Adj Close']]
    elif 'Close' in stock_data.columns:
        Adj_close_price = stock_data[['Close']]
    else:
        messagebox.showerror("Error", "Neither 'Adj Close' nor 'Close' column found in the dataset.")
        return

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(Adj_close_price)

    # Prepare sequences for LSTM (Train-Test Split)
    sequence_length = 360
    x_data, y_data = [], []

    for i in range(sequence_length, len(scaled_data)):
        x_data.append(scaled_data[i - sequence_length:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Split data into train and test sets
    train_size = int(len(x_data) * 0.7)
    x_train, x_test = x_data[:train_size], x_data[train_size:]
    y_train, y_test = y_data[:train_size], y_data[train_size:]

    # Build LSTM model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=10)

    # Evaluate model performance
    predictions = model.predict(x_test)
    inv_predictions = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_test)

    rmse = np.sqrt(np.mean((inv_predictions - inv_y_test) ** 2))
    mae = np.mean(np.abs(inv_predictions - inv_y_test))
    mape = np.mean(np.abs((inv_predictions - inv_y_test) / inv_y_test)) * 100

    result_text = f"Model Evaluation Metrics:\n"
    result_text += f"RMSE: {rmse:.4f}\n"
    result_text += f"MAE: {mae:.4f}\n"
    result_text += f"MAPE: {mape:.2f}%"

    result_label.config(text=result_text)

    # Plot Test vs Predicted Values
    plt.figure(figsize=(15, 6))
    plt.plot(inv_y_test, label="Actual Prices", color="blue")
    plt.plot(inv_predictions, label="Predicted Prices", color="orange")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"Test vs Predicted Prices for {stock}")
    plt.legend()
    plt.grid()
    plt.show()

    # User input for number of days for future prediction
    try:
        future_days = int(future_days_entry.get())
        if future_days <= 0:
            messagebox.showerror("Error", "Please enter a valid number of days!")
            return
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid integer for future days!")
        return

    last_sequence = scaled_data[-sequence_length:]
    future_predictions = []

    for _ in range(future_days):
        next_prediction = model.predict(last_sequence.reshape(1, -1, 1))[0][0]
        future_predictions.append(next_prediction)
        last_sequence = np.append(last_sequence[1:], [[next_prediction]], axis=0)

    # Scale back future predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_data = pd.DataFrame({"Predicted Price": future_predictions.flatten()})

    future_predictions_label.config(text=future_data)

    # Plot future predictions
    plt.figure(figsize=(15, 6))
    plt.plot(future_data, label="Future Predicted Prices", color="orange")
    plt.xlabel("Time (Days)")
    plt.ylabel("Predicted Price")
    plt.title(f"Future Stock Price Predictions for {stock}")
    plt.legend()
    plt.grid()
    plt.show()


# Create GUI window
root = tk.Tk()
root.title("Stock Price Prediction using LSTM")

# Set window size
root.geometry("600x500")

# Stock symbol input
stock_label = tk.Label(root, text="Enter stock symbol (e.g., GOOG, AAPL, INFY.NS, RELIANCE.BO):")
stock_label.pack(pady=10)
stock_entry = tk.Entry(root, width=50)
stock_entry.pack(pady=5)

# Number of future days input
future_days_label = tk.Label(root, text="Enter number of days for future prediction:")
future_days_label.pack(pady=10)
future_days_entry = tk.Entry(root, width=50)
future_days_entry.pack(pady=5)

# Submit button to trigger prediction
predict_button = tk.Button(root, text="Predict", command=predict_stock_prices, height=2, width=20)
predict_button.pack(pady=20)

# Label to display evaluation metrics
result_label = tk.Label(root, text="", font=("Helvetica", 12))
result_label.pack(pady=20)

# Label to display future predictions
future_predictions_label = tk.Label(root, text="", font=("Helvetica", 12))
future_predictions_label.pack(pady=10)

# Start GUI event loop
root.mainloop()
