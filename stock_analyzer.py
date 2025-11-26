import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
from datetime import datetime, timedelta

# Function to fetch stock data
def fetch_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Function to visualize stock data
def visualize_data(data, ticker):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.set_title(f'{ticker} Stock Price Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    return fig

# Function for trend analysis
def trend_analysis(data):
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Simple trend detection
    if data['MA20'].iloc[-1] > data['MA50'].iloc[-1]:
        trend = "Uptrend"
    elif data['MA20'].iloc[-1] < data['MA50'].iloc[-1]:
        trend = "Downtrend"
    else:
        trend = "Sideways"
    
    return data, trend

# Function for ML prediction
def predict_price(data):
    # Prepare data for ML
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    
    X = data[['Days']]
    y = data['Close']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict next day
    next_day = data['Days'].max() + 1
    predicted_price = model.predict([[next_day]])[0]
    
    return predicted_price, model.score(X_test, y_test)

# Streamlit Dashboard
st.title("Stock Market Trend Analyzer")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

if st.button("Analyze"):
    data = fetch_stock_data(ticker, period)
    
    if data.empty:
        st.error("No data found for this ticker.")
    else:
        # Visualization
        st.subheader("Stock Price Visualization")
        fig = visualize_data(data, ticker)
        st.pyplot(fig)
        
        # Trend Analysis
        data_with_ma, trend = trend_analysis(data)
        st.subheader("Trend Analysis")
        st.write(f"Current Trend: {trend}")
        st.line_chart(data_with_ma[['Close', 'MA20', 'MA50']])
        
        # ML Prediction
        predicted_price, accuracy = predict_price(data)
        st.subheader("ML Prediction (Next Day Closing Price)")
        st.write(f"Predicted Price: ${predicted_price:.2f}")
        st.write(f"Model Accuracy (R²): {accuracy:.2f}")
        
        # Additional Insights
        st.subheader("Key Insights")
        latest_price = data['Close'].iloc[-1]
        change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
        st.write(f"Latest Closing Price: ${latest_price:.2f}")
        st.write(f"Daily Change: ${change:.2f} ({(change/latest_price*100):.2f}%)")