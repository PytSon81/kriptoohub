import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import requests
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(layout="wide")


# === Binance API'den veri çek ===
@st.cache_data
def load_data(symbol, interval, limit):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df


# === Hareketli Ortalamalar ===
def moving_averages(df, short_window, long_window):
    df['MA_short'] = df['close'].rolling(window=short_window).mean()
    df['MA_long'] = df['close'].rolling(window=long_window).mean()
    return df


# === Al/Sat Sinyalleri ===
def generate_signals(df):
    df['buy_signal'] = np.where((df['MA_short'] > df['MA_long']) & (df['MA_short'].shift(1) <= df['MA_long'].shift(1)),
                                df['close'], np.nan)
    df['sell_signal'] = np.where((df['MA_short'] < df['MA_long']) & (df['MA_short'].shift(1) >= df['MA_long'].shift(1)),
                                 df['close'], np.nan)
    return df


# === RSI Hesapla ===
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


# === MACD Hesapla ===
def calculate_macd(df):
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df


# === Bollinger Bandı Hesapla ===
def calculate_bollinger_bands(df, window=20, num_std=2):
    df['BB_Middle'] = df['close'].rolling(window=window).mean()
    df['BB_Std'] = df['close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + num_std * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - num_std * df['BB_Std']
    return df


# === Grafik Çiz (Bollinger dahil) ===
def plot_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                                 name='Candlestick', yaxis='y'))

    # Bollinger Bandı
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_Upper'], line=dict(color='purple', width=1.2), name='BB Üst',
                             yaxis='y'))
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['BB_Middle'], line=dict(color='gray', width=1.2, dash='dot'), name='BB Orta',
                   yaxis='y'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_Lower'], line=dict(color='purple', width=1.2), name='BB Alt',
                             yaxis='y'))

    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['MA_short'], line=dict(color='blue', width=1.2), name='MA Short', yaxis='y'))
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['MA_long'], line=dict(color='orange', width=1.2), name='MA Long', yaxis='y'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['buy_signal'], mode='markers',
                             marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal', yaxis='y'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['sell_signal'], mode='markers',
                             marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal', yaxis='y'))
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Hacim', marker=dict(color='gray'), yaxis='y2'))
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple', width=2), name='RSI', yaxis='y3'))
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['MACD'], line=dict(color='blue', width=2), name='MACD', yaxis='y4'))
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['Signal_Line'], line=dict(color='red', width=2), name='Signal Line',
                   yaxis='y4'))
    fig.update_layout(
        title='Kripto Para Grafiği',
        xaxis=dict(title='Tarih'),
        yaxis=dict(title='Fiyat', domain=[0.55, 1]),
        yaxis2=dict(title='Hacim', domain=[0.40, 0.54], showgrid=False),
        yaxis3=dict(title='RSI', domain=[0.25, 0.39], showgrid=True, range=[0, 100]),
        yaxis4=dict(title='MACD', domain=[0, 0.24], showgrid=True),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_rangeslider_visible=False,
        height=900
    )
    return fig


# === Uygulama Başlat ===
st.title("📈 Kripto Para Teknik Analiz Paneli")

col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.text_input("Coin Sembolü (örn: BTCUSDT)", "BTCUSDT")
with col2:
    interval = st.selectbox("Zaman Aralığı", ['1m', '5m', '15m', '1h', '4h', '1d','1w',"1M"], index=5)
with col3:
    limit = st.slider("Veri Sayısı", min_value=100, max_value=1000, value=500)

df = load_data(symbol.upper(), interval, limit)

col4, col5 = st.columns(2)
with col4:
    short_window = st.slider("Kısa MA", min_value=2, max_value=50, value=10)
with col5:
    long_window = st.slider("Uzun MA", min_value=5, max_value=200, value=30)

df = moving_averages(df, short_window, long_window)
df = generate_signals(df)
df = calculate_rsi(df)
df = calculate_macd(df)
df = calculate_bollinger_bands(df)  # Bollinger Bandı hesapla

fig = plot_chart(df)
st.plotly_chart(fig, use_container_width=True)

# === Sinyal Performans Analizi ===
df['position'] = 0
df.loc[df['buy_signal'].notna(), 'position'] = 1
df.loc[df['sell_signal'].notna(), 'position'] = 0
df['position'] = df['position'].ffill().fillna(0)

df['return'] = df['close'].pct_change()
df['strategy_return'] = df['return'] * df['position']
cumulative_strategy_return = (1 + df['strategy_return']).cumprod()
if not cumulative_strategy_return.empty:
    total_return = cumulative_strategy_return.iloc[-1] - 1
else:
    total_return = 0 

buy_count = df['buy_signal'].notna().sum()
sell_count = df['sell_signal'].notna().sum()

trades = df[(df['buy_signal'].notna()) | (df['sell_signal'].notna())].copy()
trades['future_close'] = df['close'].shift(-3)
trades['result'] = np.where(
    trades['buy_signal'].notna() & (trades['future_close'] > trades['close']), 'correct',
    np.where(trades['sell_signal'].notna() & (trades['future_close'] < trades['close']), 'correct', 'wrong')
)
accuracy = (trades['result'] == 'correct').mean() * 100

st.subheader("📊 Sinyal Performansı")
st.write(f"✅ Toplam strateji getirisi: **%{total_return:.2f}**")
st.write(f"🔁 Toplam AL sinyali: **{buy_count}**, SAT sinyali: **{sell_count}**")
st.write(f"🎯 Tahmini doğruluk oranı: **%{accuracy:.2f}**")

#streamlit run "C:\Users\melih\PycharmProjects\PythonProject\2\borsa proje\deneme17.py"


