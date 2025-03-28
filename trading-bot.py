import MetaTrader5 as mt5
import time
import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from datetime import datetime

# Connect to MT5
def connect_mt5():
    if not mt5.initialize():
        print("MT5 connection failed!", mt5.last_error())
        quit()
    print("Connected to MT5")

# Get Market Data
def get_market_data(symbol, timeframe=mt5.TIMEFRAME_M1, bars=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# Load ML Model and Scaler
def load_ml_model():
    model = tf.keras.models.load_model("model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

# Predict Market Movement
def predict_movement(model, scaler, df):
    df_scaled = scaler.transform(df[['close', 'volatility', 'rsi']])
    X = np.array([df_scaled[-50:]])
    prediction = model.predict(X)[0][0]
    return prediction

# Execute Trades
def place_trade(symbol, lot_size, trade_type, sl_points=50, tp_points=100):
    price = mt5.symbol_info_tick(symbol).ask if trade_type == 'buy' else mt5.symbol_info_tick(symbol).bid
    deviation = 10
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if trade_type == 'buy' else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": price - sl_points * mt5.symbol_info(symbol).point if trade_type == 'buy' else price + sl_points * mt5.symbol_info(symbol).point,
        "tp": price + tp_points * mt5.symbol_info(symbol).point if trade_type == 'buy' else price - tp_points * mt5.symbol_info(symbol).point,
        "deviation": deviation,
        "magic": 123456,
        "comment": "AI Trading Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }
    order_result = mt5.order_send(request)
    return order_result

# Close All Trades at Session End
def close_all_trades():
    positions = mt5.positions_get()
    for pos in positions:
        request = {
            "action": mt5.TRADE_ACTION_CLOSE_BY,
            "position": pos.ticket
        }
        mt5.order_send(request)

# Trading Loop
def trading_loop(symbol, session_end):
    model, scaler = load_ml_model()
    while datetime.now().time() < session_end:
        df = get_market_data(symbol)
        prediction = predict_movement(model, scaler, df)
        if prediction > df['close'].iloc[-1]:
            place_trade(symbol, lot_size=0.1, trade_type='buy')
        else:
            place_trade(symbol, lot_size=0.1, trade_type='sell')
        time.sleep(60)  # Check market every minute
    close_all_trades()
    print("Session ended. All trades closed.")

if __name__ == "__main__":
    connect_mt5()
    trading_loop('EURUSD', session_end=datetime.strptime("23:00:00", "%H:%M:%S").time())
    mt5.shutdown()
