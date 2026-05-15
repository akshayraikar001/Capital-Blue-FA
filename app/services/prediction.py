import json
import threading
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.config import MODEL_DIR
from app.services.market_data import (
    fetch_company_metrics,
    fetch_company_news,
    fetch_stock_data,
)

_model_cache_lock = threading.Lock()
_model_cache: dict[str, tuple[object, object]] = {}


def model_path(symbol: str):
    return MODEL_DIR / f"{symbol}_model.h5"


def scaler_path(symbol: str):
    return MODEL_DIR / f"{symbol}_scaler.gz"


def prepare_data(df: pd.DataFrame, window: int = 60):
    closes = df["close"].astype("float").values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(closes)
    x_values, y_values = [], []
    for index in range(window, len(scaled)):
        x_values.append(scaled[index - window : index, 0])
        y_values.append(scaled[index, 0])
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    x_values = np.reshape(x_values, (x_values.shape[0], x_values.shape[1], 1))
    return x_values, y_values, scaler


def build_model(input_shape):
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_and_save(df: pd.DataFrame, symbol: str, window: int = 60, epochs: int = 5):
    x_values, y_values, scaler = prepare_data(df, window)
    model = build_model((x_values.shape[1], 1))
    model.fit(x_values, y_values, epochs=epochs, batch_size=32, verbose=0)
    model.save(model_path(symbol))
    joblib.dump(scaler, scaler_path(symbol))
    with _model_cache_lock:
        _model_cache[symbol.upper()] = (model, scaler)
    return model, scaler


def load_model_and_scaler(symbol: str):
    symbol = symbol.upper()
    with _model_cache_lock:
        cached = _model_cache.get(symbol)
        if cached is not None:
            return cached

    if not model_path(symbol).exists() or not scaler_path(symbol).exists():
        raise FileNotFoundError("Model or scaler not found")
    from tensorflow.keras.models import load_model

    model = load_model(model_path(symbol))
    scaler = joblib.load(scaler_path(symbol))
    with _model_cache_lock:
        _model_cache[symbol] = (model, scaler)
    return model, scaler


def get_or_train_model(df: pd.DataFrame, symbol: str, window: int = 60, epochs: int = 5, retrain: bool = False):
    try:
        if retrain:
            raise FileNotFoundError
        return load_model_and_scaler(symbol)
    except Exception:
        return train_and_save(df, symbol, window=window, epochs=epochs)


def predict_future_prices(model, scaler, df: pd.DataFrame, future_days: int = 5, window: int = 60):
    closes = df["close"].astype("float").values.reshape(-1, 1)
    scaled = scaler.transform(closes)
    last_window = scaled[-window:].reshape(1, window, 1)

    predictions = []
    for _ in range(future_days):
        scaled_prediction = model.predict(last_window, verbose=0)
        predicted_price = scaler.inverse_transform(scaled_prediction)[0][0]
        predictions.append(float(predicted_price))
        scaled_prediction = scaled_prediction.reshape(1, 1, 1)
        last_window = np.concatenate([last_window[:, 1:, :], scaled_prediction], axis=1)
    return predictions


def _normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    if "date" not in frame.columns:
        frame = frame.reset_index()
    if "close" not in frame.columns:
        for column in frame.columns:
            if "close" in column:
                frame = frame.rename(columns={column: "close"})
                break
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["close"] = frame["close"].astype(float)
    return frame


def build_prediction_context(symbol: str, realtime_flag: bool = False) -> dict:
    context = {
        "symbol": symbol.upper(),
        "error": None,
        "metrics": {},
        "dates_json": "[]",
        "combined_dates_json": "[]",
        "prices_json": "[]",
        "pred_dates_json": "[]",
        "pred_prices_json": "[]",
        "recommendation": None,
        "pred_next_price": None,
        "last_close": None,
        "gauge_values_json": json.dumps([60, 25, 15]),
        "news": [],
        "realtime_flag": realtime_flag,
    }

    if not symbol:
        context["error"] = "Please enter a stock symbol (e.g., AAPL)."
        return context

    daily_df = _normalize_price_frame(fetch_stock_data(symbol, days=400, intraday=False))
    if daily_df.empty:
        raise ValueError("No historical daily data returned for this symbol.")

    window = 60
    if len(daily_df) < window + 5:
        raise ValueError(f"Not enough historical rows: {len(daily_df)} (need at least {window + 5}).")

    if realtime_flag:
        try:
            display_df = fetch_stock_data(
                symbol,
                days=1,
                intraday=True,
                intraday_minutes=1440,
                intraday_interval="1m",
            )
            display_df = _normalize_price_frame(display_df)
        except Exception:
            display_df = daily_df.copy()
    else:
        display_df = daily_df.copy()

    if realtime_flag:
        display_dates = display_df["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    else:
        display_dates = display_df["date"].dt.strftime("%Y-%m-%d").tolist()
    closes = display_df["close"].tolist()

    model, scaler = get_or_train_model(daily_df, symbol, window=window, epochs=6, retrain=False)
    pred_prices = [float(value) for value in predict_future_prices(model, scaler, daily_df, future_days=5, window=window)]

    last_daily_date = pd.to_datetime(daily_df["date"].iloc[-1])
    pred_dates = [(last_daily_date + timedelta(days=index + 1)).strftime("%Y-%m-%d") for index in range(5)]
    combined_dates = display_dates + pred_dates

    metrics = fetch_company_metrics(symbol) or {}
    news_items = fetch_company_news(symbol, days=7) or []

    last_close = float(daily_df["close"].iloc[-1])
    next_day_prediction = float(pred_prices[0])
    change_pct = ((next_day_prediction - last_close) / last_close) * 100.0

    if change_pct >= 2.0:
        buy, hold, sell = 80, 15, 5
    elif change_pct >= 0.5:
        buy, hold, sell = 60, 30, 10
    elif change_pct > -0.5:
        buy, hold, sell = 30, 50, 20
    elif change_pct > -2.0:
        buy, hold, sell = 10, 40, 50
    else:
        buy, hold, sell = 5, 20, 75

    if buy > sell and buy >= 50:
        recommendation = "Buy"
    elif sell > buy and sell >= 50:
        recommendation = "Sell"
    else:
        recommendation = "Neutral"

    context.update(
        {
            "metrics": metrics,
            "dates_json": json.dumps(display_dates),
            "combined_dates_json": json.dumps(combined_dates),
            "prices_json": json.dumps(closes),
            "pred_dates_json": json.dumps(pred_dates),
            "pred_prices_json": json.dumps(pred_prices),
            "recommendation": recommendation,
            "pred_next_price": round(next_day_prediction, 2),
            "last_close": round(last_close, 2),
            "gauge_values_json": json.dumps([buy, hold, sell]),
            "news": news_items,
        }
    )
    return context


def build_stock_chart_api_payload(symbol: str) -> dict:
    context = build_prediction_context(symbol, realtime_flag=True)
    return {
        "dates": json.loads(context["dates_json"]),
        "pred_dates": json.loads(context["pred_dates_json"]),
        "prices": json.loads(context["prices_json"]),
        "pred_prices": json.loads(context["pred_prices_json"]),
        "recommendation": context["recommendation"],
        "last_close": context["last_close"],
        "pred_next_price": context["pred_next_price"],
        "gauge_values": json.loads(context["gauge_values_json"]),
        "error": context["error"],
    }
