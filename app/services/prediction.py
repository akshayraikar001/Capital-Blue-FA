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
POSITIVE_WORDS = {
    "beat",
    "bullish",
    "growth",
    "gain",
    "gains",
    "surge",
    "strong",
    "upside",
    "record",
    "profit",
    "profits",
    "outperform",
    "upgrade",
    "momentum",
    "rally",
}
NEGATIVE_WORDS = {
    "miss",
    "bearish",
    "drop",
    "drops",
    "fall",
    "falls",
    "weak",
    "downside",
    "loss",
    "losses",
    "downgrade",
    "risk",
    "lawsuit",
    "decline",
    "selloff",
}


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


def format_price(value) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def format_compact_number(value, prefix: str = "") -> str:
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"

    abs_number = abs(number)
    suffix = ""
    if abs_number >= 1_000_000_000_000:
        number /= 1_000_000_000_000
        suffix = "T"
    elif abs_number >= 1_000_000_000:
        number /= 1_000_000_000
        suffix = "B"
    elif abs_number >= 1_000_000:
        number /= 1_000_000
        suffix = "M"
    elif abs_number >= 1_000:
        number /= 1_000
        suffix = "K"
    if suffix:
        return f"{prefix}{number:,.1f}{suffix}"
    if float(value).is_integer():
        return f"{prefix}{int(float(value)):,}"
    return f"{prefix}{float(value):,.2f}"


def analyze_news_relevance(news_items: list[dict], recommendation: str | None) -> tuple[list[dict], dict]:
    enriched_items: list[dict] = []
    support_count = 0
    conflict_count = 0
    mixed_count = 0

    for item in news_items:
        article = dict(item)
        text = f"{article.get('headline', '')} {article.get('summary', '')}".lower()
        positive_hits = sum(word in text for word in POSITIVE_WORDS)
        negative_hits = sum(word in text for word in NEGATIVE_WORDS)

        if positive_hits > negative_hits:
            tone = "Positive"
        elif negative_hits > positive_hits:
            tone = "Negative"
        else:
            tone = "Neutral"

        if recommendation == "Buy":
            alignment = "Supports" if tone == "Positive" else "Conflicts" if tone == "Negative" else "Mixed"
        elif recommendation == "Sell":
            alignment = "Supports" if tone == "Negative" else "Conflicts" if tone == "Positive" else "Mixed"
        else:
            alignment = "Mixed"

        if alignment == "Supports":
            support_count += 1
        elif alignment == "Conflicts":
            conflict_count += 1
        else:
            mixed_count += 1

        article["tone"] = tone
        article["alignment"] = alignment
        enriched_items.append(article)

    if support_count > conflict_count and support_count >= 2:
        summary_label = "News supports the forecast"
        summary_tone = "support"
    elif conflict_count > support_count and conflict_count >= 2:
        summary_label = "News conflicts with the forecast"
        summary_tone = "conflict"
    else:
        summary_label = "News is mixed versus the forecast"
        summary_tone = "mixed"

    return enriched_items, {
        "label": summary_label,
        "tone": summary_tone,
        "support_count": support_count,
        "conflict_count": conflict_count,
        "mixed_count": mixed_count,
    }


def build_prediction_context(symbol: str, realtime_flag: bool = False, future_days: int = 5) -> dict:
    context = {
        "symbol": symbol.upper(),
        "error": None,
        "metrics": {},
        "metrics_display": {},
        "dates_json": "[]",
        "combined_dates_json": "[]",
        "prices_json": "[]",
        "pred_dates_json": "[]",
        "pred_prices_json": "[]",
        "recommendation": None,
        "pred_next_price": None,
        "last_close": None,
        "last_close_formatted": "-",
        "pred_next_price_formatted": "-",
        "price_change_pct": None,
        "price_change_pct_formatted": "-",
        "forecast_cards": [],
        "forecast_horizon": future_days,
        "gauge_values_json": json.dumps([0, 0, 0]),
        "news": [],
        "news_relevance": {
            "label": "Run a prediction to compare news tone with the forecast.",
            "tone": "mixed",
            "support_count": 0,
            "conflict_count": 0,
            "mixed_count": 0,
        },
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
    pred_prices = [
        float(value) for value in predict_future_prices(model, scaler, daily_df, future_days=future_days, window=window)
    ]

    last_daily_date = pd.to_datetime(daily_df["date"].iloc[-1])
    pred_dates = [(last_daily_date + timedelta(days=index + 1)).strftime("%Y-%m-%d") for index in range(future_days)]
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

    metrics_display = {
        "open": format_price(metrics.get("open")),
        "fiftyTwoWeekHigh": format_price(metrics.get("fiftyTwoWeekHigh")),
        "fiftyTwoWeekLow": format_price(metrics.get("fiftyTwoWeekLow")),
        "volume": format_compact_number(metrics.get("volume")),
        "market_cap": format_compact_number(metrics.get("marketCap"), prefix="$"),
        "trailingPE": "-" if metrics.get("trailingPE") in (None, "") else f"{float(metrics.get('trailingPE')):.2f}",
    }
    forecast_cards = [
        {
            "day_label": f"Day {index + 1}",
            "date": pred_dates[index],
            "price": round(pred_prices[index], 2),
            "price_formatted": format_price(pred_prices[index]),
        }
        for index in range(len(pred_prices))
    ]
    enriched_news, news_relevance = analyze_news_relevance(news_items, recommendation)

    context.update(
        {
            "metrics": metrics,
            "metrics_display": metrics_display,
            "dates_json": json.dumps(display_dates),
            "combined_dates_json": json.dumps(combined_dates),
            "prices_json": json.dumps(closes),
            "pred_dates_json": json.dumps(pred_dates),
            "pred_prices_json": json.dumps(pred_prices),
            "recommendation": recommendation,
            "pred_next_price": round(next_day_prediction, 2),
            "last_close": round(last_close, 2),
            "pred_next_price_formatted": format_price(next_day_prediction),
            "last_close_formatted": format_price(last_close),
            "price_change_pct": round(change_pct, 2),
            "price_change_pct_formatted": f"{change_pct:+.2f}%",
            "forecast_cards": forecast_cards,
            "forecast_horizon": future_days,
            "gauge_values_json": json.dumps([buy, hold, sell]),
            "news": enriched_news,
            "news_relevance": news_relevance,
        }
    )
    return context


def build_stock_chart_api_payload(symbol: str, future_days: int = 5) -> dict:
    context = build_prediction_context(symbol, realtime_flag=True, future_days=future_days)
    return {
        "dates": json.loads(context["dates_json"]),
        "pred_dates": json.loads(context["pred_dates_json"]),
        "prices": json.loads(context["prices_json"]),
        "pred_prices": json.loads(context["pred_prices_json"]),
        "recommendation": context["recommendation"],
        "last_close": context["last_close"],
        "pred_next_price": context["pred_next_price"],
        "last_close_formatted": context["last_close_formatted"],
        "pred_next_price_formatted": context["pred_next_price_formatted"],
        "price_change_pct_formatted": context["price_change_pct_formatted"],
        "gauge_values": json.loads(context["gauge_values_json"]),
        "forecast_cards": context["forecast_cards"],
        "error": context["error"],
    }
