import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

START = "2015-01-01"

st.title('Stock Prediction App')

# Add a search bar for stock symbols
selected_stock = st.text_input('Enter stock symbol (e.g., AAPL)', 'AAPL')

period = 60  # Predict for the next 60 days

@st.cache_data(persist=True)
def load_data(ticker):
    end_date = date.today() - timedelta(days=2)  # Fetch data up to 2 days before today
    data = yf.download(ticker, START, end_date.strftime("%Y-%m-%d"))
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Predict forecast with Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Evaluate the model
true_values = data['Close'].values[-period:]
predicted_values = forecast['yhat'].values[-period:]

mae = mean_absolute_error(true_values, predicted_values)
mse = mean_squared_error(true_values, predicted_values)
accuracy = 100 - (mae / true_values.mean() * 100)  # Calculate accuracy as a percentage

# Display forecast data
st.subheader('Forecast data')
st.write(forecast.tail())

# Display evaluation metrics
st.subheader('Evaluation Metrics')
st.write(f'Mean Absolute Error: {mae:.2f}')
st.write(f'Mean Squared Error: {mse:.2f}')
st.write(f'Accuracy: {accuracy:.2f}%')

# Plot forecast data
st.write(f'Forecast plot for the next {period} days')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Display forecast components
st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
