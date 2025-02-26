# Stock Price Prediction using LSTM

## Project Description
This project predicts stock prices using Long Short-Term Memory (LSTM) neural networks. It provides a graphical user interface (GUI) for users to input stock symbols and receive predictions.

## Prerequisites
- Python 3.x
- Required libraries:
  - `yfinance`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `keras`
  - `tkinter`

To install the required libraries, run:
```bash
pip install yfinance pandas scikit-learn matplotlib keras
```

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd StockMarketPrediction
   ```

## Running the Application
To run the application, execute:
```bash
python prediction.py
```
A GUI will open for stock price predictions.

## Using the Application
1. Enter a stock symbol (e.g., GOOG, AAPL) in the input field.
2. Enter the number of days for future predictions.
3. Click the "Predict" button to see the results.

## Understanding the Output
The application displays evaluation metrics:
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

It also generates plots for actual vs. predicted prices and future predictions.

## Contributing
Feel free to submit issues or pull requests for improvements.

