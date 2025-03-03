# Sentiment-Based-Bitcoin-Price-Forecasting
This repository contains a Python-based project that leverages sentiment analysis from social media and news sources to forecast Bitcoin price movements. The project integrates VADER, LSTM, and time-series analysis to predict Bitcoin prices based on market sentiment. Ideal for cryptocurrency enthusiasts, data scientists, and financial analysts.
This project aims to predict Bitcoin price movements using sentiment analysis from social media platforms (e.g., Twitter, Reddit) and news articles. By combining Natural Language Processing (NLP) and time-series forecasting techniques, we provide a data-driven approach to understanding how market sentiment influences cryptocurrency prices.
### Introduction
Bitcoin and other cryptocurrencies are highly volatile assets, often influenced by market sentiment. This project explores the relationship between public sentiment (derived from social media and news) and Bitcoin price movements. By analyzing textual data and applying machine learning models, we aim to forecast short-term price trends.
### Features
Sentiment Analysis: Extract and analyze sentiment from Twitter, Reddit, and news headlines.

Price Forecasting: Predict Bitcoin prices using time-series models (e.g., ARIMA, LSTM).

Data Visualization: Interactive visualizations of sentiment trends and price movements.

Modular Codebase: Easy-to-extend code for adding new data sources or models.
### Methodology
Data Collection: Gather textual data from X and Kaggle and BTC price data from BINANCE.

Preprocessing: Clean and preprocess text data (tokenization, stemming, etc.).

Sentiment Analysis: Use VADER to classify sentiment.

Feature Engineering: Combine sentiment scores with historical price data.

Model Training: Train LSTM for price forecasting.

Evaluation: Evaluate model performance using metrics like RMSE and MAE.
