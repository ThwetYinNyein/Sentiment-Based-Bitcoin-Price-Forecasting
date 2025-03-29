import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Binance API setup
api_key = "YOUR_BINANCE_API_KEY"
api_secret = "YOUR_BINANCE_API_SECRET"
client = Client(api_key, api_secret)

# Load pre-trained LSTM model
model = load_model("saved_model.h5")  # Replace with your model path

# Function to fetch real-time data
def fetch_real_time_data(symbol, interval="1m", lookback="1 hour ago UTC"):
    klines = client.get_historical_klines(symbol, interval, lookback)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data['close'] = data['close'].astype(float)
    # Convert to Myanmar Time (UTC+6:30)
    data['timestamp'] = data['timestamp'] + timedelta(hours=6, minutes=30)
    # Sort by timestamp
    data = data.sort_values(by='timestamp')
    return data

# Function to preprocess data for LSTM
def preprocess_data(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['close']])
    sequence_length = 60
    X = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, scaler

# Function to generate forecasts
def generate_forecasts(model, X, scaler):
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# Streamlit app
st.title("Sentiment-based Crypto Price Forecasting Dashboard")
st.write("Real-time crypto price visualization and forecasting using LSTM")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About LSTM"])

# Home Page
if page == "Home":
    # Coin selection
    coin = st.sidebar.selectbox("Select Coin", ["BTC", "ETH", "DOGE"])  # ETH and DOGE are placeholders
    st.sidebar.write(f"Selected Coin: {coin}")

    # Fetch real-time data
    data = fetch_real_time_data(f"{coin}USDT")

    # Display raw data
    st.subheader("Raw Data")
    st.write(data.tail())

    # Display real-time candlestick chart
    st.subheader(f"Real-Time {coin} Price (Candlestick Chart)")
    candlestick = go.Candlestick(
        x=data['timestamp'],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close']
    )
    fig = go.Figure(data=[candlestick])
    st.plotly_chart(fig)

    # Preprocess data for LSTM
    X, scaler = preprocess_data(data)

    # Generate forecasts
    if st.button("Generate Forecasts"):
        predictions = generate_forecasts(model, X, scaler)
        st.subheader("Forecasted Prices")
        st.write(predictions)

        # Plot forecasts
        forecast_dates = pd.date_range(start=data['timestamp'].iloc[-1], periods=len(predictions), freq='T')
        forecast_df = pd.DataFrame({'timestamp': forecast_dates, 'forecast': predictions.flatten()})
        st.subheader(f"Forecasted {coin} Prices (Next 60 Minutes)")
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=forecast_df['timestamp'], y=forecast_df['forecast'], mode='lines', name='Forecast'))
        st.plotly_chart(fig_forecast)

# About LSTM Page
elif page == "About LSTM":
    st.title("About LSTM Model")
    st.write("""
    ### What is LSTM?
    Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN) 
    designed to handle sequential data, such as time series, text, or audio. 


#### **Layers & Parameters:**
1. **Input Layer:** Accepts sequential input data.  
2. **LSTM Layer 1:**
   - **Kernel Shape:** (1×200)  
   - **Recurrent Kernel Shape:** (50×200)  
   - **Bias Shape:** (200)  
   - Includes an **activation function** for feature extraction.  
3. **Dropout Layer:** Prevents overfitting by randomly dropping connections.  
4. **LSTM Layer 2:**  
   - **Kernel Shape:** (50×200)  
   - **Recurrent Kernel Shape:** (50×200)  
   - **Bias Shape:** (200)  
   - Includes an **activation function** for better learning.  
5. **Dropout Layer:** Improves generalization.  
6. **Dense (Fully Connected) Layer:**  
   - **Kernel Shape:** (50×1)  
   - **Bias Shape:** (1)  
   - Outputs the final prediction.  

This model is optimized for handling time-series or sequential data efficiently.

### Training Details
Optimizer: Adam with a learning rate of 0.001.
Loss Function: Mean Squared Error (MSE).
Training Data: Historical BTC price data from 2024 to February 2025.
Sequence Length: 60 minutes of historical data used to predict the next minute.

### Applications
This LSTM model is used to forecast cryptocurrency prices (e.g., BTC, ETH, DOGE) in real-time. It can predict prices for the next minute, hour, day, or even month, depending on the training data and model configuration.
""")

    # Optional: Add a diagram of the LSTM model
    st.image("modelinfo1.png", caption="LSTM Model Architecture", use_container_width=600)