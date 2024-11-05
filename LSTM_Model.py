import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

# Function to fetch historical stock data using Yahoo Finance

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to create LSTM model with increased complexity and regularization
def create_lstm_model(input_shape, num_lstm_layers=3, num_lstm_units=64, dropout_rate=0.3):
    model = Sequential()
    for _ in range(num_lstm_layers - 1):
        model.add(LSTM(units=num_lstm_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))  # Dropout layer for regularization
    model.add(LSTM(units=num_lstm_units))
    model.add(Dropout(dropout_rate))  # Dropout layer for regularization
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Streamlit UI
st.title('Stock Price Prediction')
st.sidebar.header('User Input')

# User input for stock ticker and date range
ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL)', 'AAPL')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2022-01-01'))

# Fetch historical stock data
stock_data = get_stock_data(ticker, start_date, end_date)

# Data preprocessing
if not stock_data.empty:
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Open', 'High', 'Low', 'Close']])

    # Prepare training data
    prediction_days = 60
    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i - prediction_days:i, :])  # Input sequences with shape (60, 4)
        y_train.append(scaled_data[i, 3])  # Closing price

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape input data to fit the LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

    # Create and train LSTM model with increased complexity and regularization
    lstm_model = create_lstm_model(input_shape=(x_train.shape[1], x_train.shape[2]), num_lstm_layers=3, num_lstm_units=64, dropout_rate=0.3)

    # Implement Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Monitor training and validation loss
    history = lstm_model.fit(x_train, y_train, epochs=150, batch_size=16, validation_split=0.1, callbacks=[early_stopping])

    # Plot training and validation loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Display the Matplotlib figure using st.pyplot() with explicit passing of the figure
    st.pyplot(fig)

    # Test the model
    test_data = scaled_data[-prediction_days:]
    x_test = np.array([test_data])  # Input shape: (1, 60, 4)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    predicted_prices = []
    for _ in range(prediction_days):
        prediction = lstm_model.predict(x_test)
        predicted_prices.append(prediction[0, 0])
        x_test = np.roll(x_test, -1, axis=1)  # Shift the array to remove the first element along the second axis
        x_test[0, -1] = prediction  # Add the new prediction at the end along the second axis

    # Inverse transform the predictions
    predicted_prices = np.array(predicted_prices).reshape(-1, 1)  # Reshape to (60, 1)
    predicted_prices = scaler.inverse_transform(np.hstack((test_data[:, 0:3], predicted_prices)))[:, 3].reshape(-1, 1)  # Concatenate and inverse transform

    # Generate future dates for the LSTM predictions
    last_date_lstm = stock_data.index[-1]
    future_dates_lstm = pd.date_range(start=last_date_lstm, periods=prediction_days, freq='B')

    # Convert LSTM predicted prices to DataFrame
    predicted_prices_df_lstm = pd.DataFrame(predicted_prices, index=future_dates_lstm, columns=['LSTM Predictions'])

    # Convert stock_data to DataFrame and concatenate with predicted_prices_df_lstm
    stock_data_df_lstm = pd.DataFrame(stock_data)
    merged_data_lstm = pd.concat([stock_data_df_lstm, predicted_prices_df_lstm])

    # Model evaluation for LSTM
    true_values_lstm = stock_data['Close'][-prediction_days:].values
    mae_lstm = mean_absolute_error(true_values_lstm, predicted_prices.flatten())
    mse_lstm = mean_squared_error(true_values_lstm, predicted_prices.flatten())

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape_lstm = np.mean(np.abs((true_values_lstm - predicted_prices.flatten()) / true_values_lstm)) * 100

    # Visualize LSTM Model Predictions
    st.subheader('LSTM Model Predictions')
    st.write(f'LSTM Mean Absolute Error: {mae_lstm:.2f}')
    st.write(f'LSTM Mean Squared Error: {mse_lstm:.2f}')
    st.write(f'LSTM Mean Absolute Percentage Error (MAPE): {mape_lstm:.2f}%')
    fig_lstm = px.line(merged_data_lstm, x=merged_data_lstm.index, y=['Close', 'LSTM Predictions'])
    st.plotly_chart(fig_lstm)
else:
    st.warning('No data available for the selected stock and date range.')
