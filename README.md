# Sentiment-Based-Bitcoin-Price-Forecasting
This project aims to predict Bitcoin price movements using sentiment analysis from social media platforms (e.g., X, Reddit, Google) and news articles. Combining sentiment analysis models and time-series forecasting techniques provides a data-driven approach to understanding how market sentiment influences cryptocurrency prices.
### Introduction
Bitcoin and other cryptocurrencies are highly volatile assets, often influenced by market sentiment. This project explores the relationship between public sentiment (derived from social media and news) and Bitcoin price movements. By analyzing textual data and applying machine learning models, we aim to forecast short-term price trends.
### Features
✨Sentiment Analysis: Extract and analyze sentiment from Twitter, Reddit, and news headlines.

✨Price Forecasting: Predict Bitcoin prices using the time-series model LSTM.

✨Data Visualization: Interactive visualizations of sentiment trends and price movements.

✨Alter System: Streamlit push notification for hitting the price threshold.
### Methodology
Data Collection: Gather textual data from X and Kaggle and BTC price data from BINANCE.

Preprocessing: Clean and preprocess text data (tokenization, stemming, etc.).

Sentiment Analysis: Use VADER to classify sentiment.

Feature Engineering: Combine sentiment scores with historical price data.

Model Training: Train LSTM for price forecasting.

Evaluation: Evaluate model performance using metrics like RMSE and MAE.
