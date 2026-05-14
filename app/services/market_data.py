import datetime as dt

import finnhub
import pandas as pd
import requests
import yfinance as yf

from app.config import FINNHUB_API_KEY

finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    normalized_columns = []
    for column in df.columns:
        if isinstance(column, tuple):
            column = "_".join(str(part) for part in column)
        normalized_columns.append(str(column).lower())
    df.columns = normalized_columns

    if "date" not in df.columns:
        df = df.reset_index()
        if "date" not in df.columns:
            for column in df.columns:
                if column in {"index", "datetime"} or "date" in column or "time" in column:
                    df = df.rename(columns={column: "date"})
                    break

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def fetch_quote(symbol: str) -> dict:
    response = requests.get(
        "https://finnhub.io/api/v1/quote",
        params={"symbol": symbol.upper(), "token": FINNHUB_API_KEY},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def fetch_general_market_news(limit: int = 30) -> list[dict]:
    response = requests.get(
        "https://finnhub.io/api/v1/news",
        params={"category": "general", "token": FINNHUB_API_KEY},
        timeout=20,
    )
    response.raise_for_status()
    articles = response.json() or []
    articles = sorted(articles, key=lambda item: item.get("datetime", 0), reverse=True)
    return [_serialize_article(item) for item in articles[:limit]]


def _serialize_article(item: dict) -> dict:
    timestamp = item.get("datetime")
    if timestamp:
        formatted_datetime = dt.datetime.fromtimestamp(timestamp).strftime("%b %d, %Y %H:%M")
    else:
        formatted_datetime = ""

    summary = (item.get("summary") or "").strip()
    if len(summary) > 100:
        summary = f"{summary[:97]}..."

    return {
        "headline": item.get("headline", ""),
        "url": item.get("url", ""),
        "image": item.get("image", ""),
        "summary": summary,
        "source": item.get("source", ""),
        "datetime": timestamp or 0,
        "datetime_formatted": formatted_datetime,
    }


def fetch_candles_finnhub(symbol: str, days: int = 365) -> pd.DataFrame:
    end_ts = int(dt.datetime.now().timestamp())
    start_ts = int((dt.datetime.now() - dt.timedelta(days=days)).timestamp())
    candles = finnhub_client.stock_candles(symbol.upper(), "D", start_ts, end_ts)
    if candles.get("s") != "ok":
        raise RuntimeError(f"Finnhub error: {candles}")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(candles["t"], unit="s"),
            "open": candles["o"],
            "high": candles["h"],
            "low": candles["l"],
            "close": candles["c"],
            "volume": candles["v"],
        }
    )
    return _normalize_df(frame)


def fetch_intraday_finnhub(symbol: str, minutes: int = 1440, resolution: str = "1") -> pd.DataFrame:
    end_ts = int(dt.datetime.now().timestamp())
    start_ts = int((dt.datetime.now() - dt.timedelta(minutes=minutes)).timestamp())
    candles = finnhub_client.stock_candles(symbol.upper(), resolution, start_ts, end_ts)
    if candles.get("s") != "ok" or not candles.get("t"):
        raise RuntimeError(f"Finnhub intraday error or empty data: {candles}")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(candles["t"], unit="s", origin="unix"),
            "open": candles["o"],
            "high": candles["h"],
            "low": candles["l"],
            "close": candles["c"],
            "volume": candles["v"],
        }
    )
    return _normalize_df(frame)


def fetch_candles_yfinance(symbol: str, days: int = 365) -> pd.DataFrame:
    frame = yf.download(symbol.upper(), period=f"{days}d", interval="1d", progress=False)
    if frame is None or frame.empty:
        raise RuntimeError("yfinance returned empty data")
    frame = frame.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return _normalize_df(frame)


def fetch_intraday_yfinance(symbol: str, days: int = 1, interval: str = "1m") -> pd.DataFrame:
    frame = yf.download(symbol.upper(), period=f"{days}d", interval=interval, progress=False)
    if frame is None or frame.empty:
        raise RuntimeError("yfinance intraday returned empty data")
    frame = frame.reset_index().rename(
        columns={
            "Datetime": "date",
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return _normalize_df(frame)


def fetch_stock_data(
    symbol: str,
    days: int = 365,
    intraday: bool = False,
    intraday_minutes: int = 1440,
    intraday_interval: str = "1m",
) -> pd.DataFrame:
    symbol = symbol.upper()
    if intraday:
        try:
            resolution = str(int(intraday_interval.replace("m", "")))
            return fetch_intraday_finnhub(symbol, minutes=intraday_minutes, resolution=resolution)
        except Exception:
            try:
                return fetch_intraday_yfinance(symbol, days=max(1, int(days)), interval=intraday_interval)
            except Exception:
                return fetch_candles_yfinance(symbol, days=days)
    try:
        return fetch_candles_finnhub(symbol, days=days)
    except Exception:
        return fetch_candles_yfinance(symbol, days=days)


def fetch_company_metrics(symbol: str) -> dict:
    try:
        info = yf.Ticker(symbol.upper()).info or {}
        return {
            "market_cap": info.get("marketCap"),
            "trailingPE": info.get("trailingPE"),
            "previousClose": info.get("previousClose"),
            "open": info.get("open"),
            "volume": info.get("volume"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return {}


def fetch_company_news(symbol: str, days: int = 7) -> list[dict]:
    symbol = symbol.upper()
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=days)
        response = finnhub_client.company_news(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        return [_serialize_article(item) for item in response[:6]]
    except Exception:
        try:
            items = getattr(yf.Ticker(symbol), "news", None) or []
            serialized = []
            for item in items[:6]:
                serialized.append(
                    {
                        "headline": item.get("title") or item.get("headline") or "",
                        "url": item.get("link") or item.get("url") or "",
                        "image": item.get("img") or "",
                        "summary": "",
                        "source": item.get("publisher") or "",
                        "datetime": 0,
                        "datetime_formatted": "",
                    }
                )
            return serialized
        except Exception:
            return []


def fetch_popular_stock_quotes(symbols: list[str]) -> list[dict]:
    quotes = []
    for symbol in symbols:
        try:
            data = fetch_quote(symbol)
            quotes.append(
                {
                    "symbol": symbol.upper(),
                    "current_price": data.get("c"),
                    "previous_close": data.get("pc"),
                    "change": round(data.get("d", 0) or 0, 2),
                }
            )
        except Exception:
            quotes.append(
                {
                    "symbol": symbol.upper(),
                    "current_price": None,
                    "previous_close": None,
                    "change": 0,
                }
            )
    return quotes

