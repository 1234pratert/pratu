import yfinance as yf
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import talib
import streamlit as st
import matplotlib.pyplot as plt

# स्टॉक डेटा लाने के लिए एक फंक्शन
def get_stock_data(ticker, interval, period):
    return yf.download(ticker, interval=interval, period=period)

# सेंटिमेंट विश्लेषण फंक्शन
def get_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)['compound']

# तकनीकी संकेतक को जोड़ने के लिए फंक्शन
def add_technical_indicators(data):
    data['SMA_5'] = talib.SMA(data['Close'], timeperiod=5)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    return data

# Streamlit UI के लिए कोड
def app():
    st.title('Stock Sentiment Analysis Dashboard')

    # यूज़र से स्टॉक ट ticker का चयन कराना
    ticker = st.text_input("Enter the Stock Ticker (e.g. AAPL):", "AAPL")
    
    if ticker:
        # 1 घंटे, 15 मिनट, और 5 मिनट के डेटा लाना
        data_1h = get_stock_data(ticker, '1h', '1d')
        data_15m = get_stock_data(ticker, '15m', '1d')
        data_5m = get_stock_data(ticker, '5m', '1d')

        # तकनीकी संकेतक जोड़ना
        data_1h = add_technical_indicators(data_1h)
        data_15m = add_technical_indicators(data_15m)
        data_5m = add_technical_indicators(data_5m)

        # कंपनी और सेक्टर के सेंटिमेंट (यह उदाहरण के लिए हैं, आप इसे अपने हिसाब से बदल सकते हैं)
        company_sentiment = get_sentiment("Company has launched a new product, stock price is up!")
        sector_sentiment = get_sentiment("Tech sector is growing, stocks are performing well.")

        # इंडिकेटर और सेंटिमेंट को जोड़कर परिणाम निकालना
        sentiment_score = company_sentiment + sector_sentiment
        technical_score_1h = data_1h['RSI'][-1] / 100
        technical_score_15m = data_15m['RSI'][-1] / 100
        technical_score_5m = data_5m['RSI'][-1] / 100

        # सेंटिमेंट और तकनीकी संकेतक को जोड़ना
        overall_sentiment_1h = sentiment_score + technical_score_1h
        overall_sentiment_15m = sentiment_score + technical_score_15m
        overall_sentiment_5m = sentiment_score + technical_score_5m

        # सिग्नल (बुलिश/बेयरिश)
        def get_signal(sentiment):
            if sentiment > 0.6:
                return "Bullish"
            elif sentiment < -0.6:
                return "Bearish"
            else:
                return "Neutral"

        # Streamlit पर परिणाम दिखाना
        st.header(f"Sentiment Analysis for {ticker}")
        st.write(f"Company Sentiment: {company_sentiment:.2f}")
        st.write(f"Sector Sentiment: {sector_sentiment:.2f}")

        st.subheader("1-Hour Sentiment")
        st.write(f"RSI: {data_1h['RSI'][-1]:.2f}")
        st.write(f"Sentiment: {get_signal(overall_sentiment_1h)}")

        st.subheader("15-Minute Sentiment")
        st.write(f"RSI: {data_15m['RSI'][-1]:.2f}")
        st.write(f"Sentiment: {get_signal(overall_sentiment_15m)}")

        st.subheader("5-Minute Sentiment")
        st.write(f"RSI: {data_5m['RSI'][-1]:.2f}")
        st.write(f"Sentiment: {get_signal(overall_sentiment_5m)}")

        # डेटा का चार्ट
        st.subheader("Stock Price Chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data_1h['Close'], label="1-Hour Close Price")
        ax.set_title(f'{ticker} Stock Price (1-Hour)')
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

# ऐप को रन करना
if __name__ == '__main__':
    app()
