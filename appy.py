# Gerekli kütüphaneleri yükle
!pip install ccxt tensorflow ta plotly -q

import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytz
import ta

# === AYARLAR ===
coin_symbol = 'SOL/USDT'
time_frame = '1h'
data_length = 1500
sequence_length = 50
local_timezone = 'Europe/Istanbul'

# Veriyi çek
print("Veriler çekiliyor...")
exchange = ccxt.gateio()
exchange.rateLimit = 1000

since = exchange.milliseconds() - data_length * 3600 * 1000
ohlcv = exchange.fetch_ohlcv(coin_symbol, time_frame, since)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
df['timestamp'] = df['timestamp'].dt.tz_convert(local_timezone)

# Teknik göstergeler
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df = df.bfill()  # Eski uyumlu şekilde bfill

# Sentiment simülasyonu
np.random.seed(42)
df['sentiment'] = np.random.uniform(-1, 1, size=len(df))

# Veri ön işleme
data = df[['close', 'volume', 'rsi', 'macd', 'macd_signal', 'sentiment']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Sekans oluşturma
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)

# Eğitim/test ayrımı
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Model
print("Model eğitiliyor...")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Tahmin
print("Tahmin yapılıyor...")
predicted_prices = model.predict(X_test, verbose=0)
predicted_prices = scaler.inverse_transform(
    np.concatenate((predicted_prices, np.zeros((len(predicted_prices), data.shape[1]-1))), axis=1)
)[:, 0]
real_prices = scaler.inverse_transform(
    np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), data.shape[1]-1))), axis=1)
)[:, 0]

# Alım ve satış sinyalleri (Basit bir strateji: RSI < 30 Alım, RSI > 70 Satış)
buy_signals = df['rsi'] < 30
sell_signals = df['rsi'] > 70

# Alım (BUY) ve satış (SELL) noktalarını grafikte göster
buy_signals = pd.Series(buy_signals, index=df['timestamp']).dropna()
sell_signals = pd.Series(sell_signals, index=df['timestamp']).dropna()

# TradingView tarzı grafik (Plotly)
print("Grafik oluşturuluyor...")
last_test_index = df.index[-len(real_prices):]
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df['timestamp'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name="Gerçek Fiyat"
))
fig.add_trace(go.Scatter(
    x=df['timestamp'].iloc[-len(predicted_prices):],
    y=predicted_prices,
    mode='lines',
    name='Tahmin Edilen Fiyat',
    line=dict(color='red')
))

# Alım (BUY) ve satış (SELL) noktalarını ekleyelim
fig.add_trace(go.Scatter(
    x=buy_signals.index,
    y=buy_signals.values,
    mode='markers',
    marker=dict(symbol='triangle-up', color='green', size=10),
    name='Alım (BUY)'
))

fig.add_trace(go.Scatter(
    x=sell_signals.index,
    y=sell_signals.values,
    mode='markers',
    marker=dict(symbol='triangle-down', color='red', size=10),
    name='Satış (SELL)'
))

fig.update_layout(
    title=f'{coin_symbol} Fiyat Tahmini ({time_frame})',
    xaxis_title='Zaman',
    yaxis_title='Fiyat',
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)
fig.show()

# Metirkler
mae = mean_absolute_error(real_prices, predicted_prices)
mse = mean_squared_error(real_prices, predicted_prices)
rmse = np.sqrt(mse)
r2 = r2_score(real_prices, predicted_prices)

print("\n--- MODEL BAŞARI METRİKLERİ ---")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2 Skoru: {r2:.4f}")

# Bir sonraki fiyat tahmini
last_sequence = scaled_data[-sequence_length:]
last_sequence = np.reshape(last_sequence, (1, sequence_length, last_sequence.shape[1]))
next_price_scaled = model.predict(last_sequence, verbose=0)
next_price = scaler.inverse_transform(
    np.concatenate((next_price_scaled, np.zeros((1, data.shape[1]-1))), axis=1)
)[:, 0]
next_timestamp = df['timestamp'].iloc[-1] + pd.Timedelta(hours=1)
print(f"\nTahmin edilen bir sonraki fiyat: {next_price[0]:.4f} ({next_timestamp})")
