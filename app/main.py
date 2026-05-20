import json
from urllib.parse import urlencode
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.config import SESSION_SECRET, STATIC_DIR, TEMPLATES_DIR
from app.database import (
    add_watchlist_symbol,
    create_user,
    fetch_prediction_history,
    fetch_user_by_email,
    fetch_user_by_id,
    fetch_user_by_username,
    fetch_watchlist_symbols,
    init_db,
    remove_watchlist_symbol,
    save_prediction_history,
)
from app.security import hash_password, verify_password
from app.services.market_data import (
    fetch_company_metrics,
    fetch_general_market_news,
    fetch_popular_stock_quotes,
    fetch_quote,
    fetch_stock_data,
)
from app.services.prediction import (
    build_prediction_context,
    build_stock_chart_api_payload,
    format_compact_number,
    format_price,
)

app = FastAPI(title="Capital Blue FastAPI")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.on_event("startup")
def startup_event():
    init_db()


def add_message(request: Request, text: str, category: str = "success") -> None:
    messages = list(request.session.get("messages", []))
    messages.append({"text": text, "category": category})
    request.session["messages"] = messages


def pop_messages(request: Request) -> list[dict]:
    messages = list(request.session.get("messages", []))
    request.session["messages"] = []
    return messages


def get_current_user(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return fetch_user_by_id(int(user_id))


def login_required(request: Request):
    user = get_current_user(request)
    if user:
        return user
    next_url = request.url.path
    if request.url.query:
        next_url = f"{next_url}?{request.url.query}"
    query = urlencode({"next": next_url})
    return RedirectResponse(url=f"/login?{query}", status_code=303)


def render(request: Request, template_name: str, context: dict | None = None, status_code: int = 200):
    base_context = {
        "request": request,
        "messages": pop_messages(request),
        "current_user": get_current_user(request),
    }
    if context:
        base_context.update(context)
    return templates.TemplateResponse(request, template_name, base_context, status_code=status_code)


def empty_prediction_context(
    symbol: str | None = None,
    error: str | None = None,
    realtime_flag: bool = False,
    forecast_horizon: int = 5,
) -> dict:
    return {
        "symbol": symbol,
        "error": error,
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
        "pred_next_price_formatted": "-",
        "last_close_formatted": "-",
        "price_change_pct": None,
        "price_change_pct_formatted": "-",
        "forecast_cards": [],
        "forecast_horizon": forecast_horizon,
        "gauge_values_json": json.dumps([0, 0, 0]),
        "news": [],
        "news_relevance": {
            "label": "Run a prediction to compare news tone with the forecast.",
            "tone": "mixed",
            "support_count": 0,
            "conflict_count": 0,
            "mixed_count": 0,
        },
        "quick_symbols": ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN"],
        "prediction_history": [],
        "watchlist_symbols": [],
        "is_in_watchlist": False,
        "realtime_flag": realtime_flag,
    }


def normalize_forecast_horizon(value) -> int:
    try:
        horizon = int(value)
    except (TypeError, ValueError):
        return 5
    return horizon if horizon in {1, 3, 5} else 5


def serialize_prediction_history(rows) -> list[dict]:
    items = []
    for row in rows:
        last_close = row["last_close"]
        predicted_price = row["predicted_price"]
        delta_pct = None
        if last_close not in (None, 0) and predicted_price is not None:
            delta_pct = ((predicted_price - last_close) / last_close) * 100.0
        items.append(
            {
                "symbol": row["symbol"],
                "forecast_horizon": row["forecast_horizon"],
                "recommendation": row["recommendation"] or "No Signal",
                "last_close_formatted": format_price(last_close),
                "predicted_price_formatted": format_price(predicted_price),
                "delta_pct_formatted": f"{delta_pct:+.2f}%" if delta_pct is not None else "-",
                "created_at": row["created_at"],
            }
        )
    return items


def prediction_page_extras(request: Request, symbol: str | None = None) -> dict:
    extras = {
        "quick_symbols": ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN"],
        "prediction_history": [],
        "watchlist_symbols": [],
        "is_in_watchlist": False,
    }
    user = get_current_user(request)
    if not user:
        return extras

    watchlist_rows = fetch_watchlist_symbols(int(user["id"]))
    watchlist_symbols = [row["symbol"] for row in watchlist_rows]
    extras["watchlist_symbols"] = watchlist_symbols
    extras["is_in_watchlist"] = bool(symbol and symbol.upper() in watchlist_symbols)
    extras["prediction_history"] = serialize_prediction_history(fetch_prediction_history(int(user["id"])))
    return extras


def build_stock_page_insights(
    symbol: str,
    current_price: float,
    previous_close: float | None,
    history_closes: list[float],
    metrics: dict,
) -> dict:
    weekly_high = max(history_closes) if history_closes else None
    weekly_low = min(history_closes) if history_closes else None
    avg_price = sum(history_closes) / len(history_closes) if history_closes else None
    range_spread = (weekly_high - weekly_low) if weekly_high is not None and weekly_low is not None else None
    change = round((current_price - previous_close), 2) if previous_close not in (None, 0) else 0.0
    change_pct = ((current_price - previous_close) / previous_close * 100.0) if previous_close not in (None, 0) else 0.0

    if change_pct >= 2:
        direction_label = "Bullish"
        direction_tone = "up"
        insight = f"{symbol} is trading firmly above the previous close with strong short-term momentum."
    elif change_pct > 0:
        direction_label = "Positive"
        direction_tone = "up"
        insight = f"{symbol} is holding a positive intraday bias and staying above the previous close."
    elif change_pct <= -2:
        direction_label = "Weak"
        direction_tone = "down"
        insight = f"{symbol} is under pressure today and trading notably below the previous close."
    elif change_pct < 0:
        direction_label = "Soft"
        direction_tone = "down"
        insight = f"{symbol} is slightly below the previous close, signaling cautious short-term sentiment."
    else:
        direction_label = "Neutral"
        direction_tone = "flat"
        insight = f"{symbol} is trading near the previous close with balanced short-term momentum."

    if weekly_high not in (None, 0):
        distance_from_high = ((current_price - weekly_high) / weekly_high) * 100.0
    else:
        distance_from_high = None
    if weekly_low not in (None, 0):
        distance_from_low = ((current_price - weekly_low) / weekly_low) * 100.0
    else:
        distance_from_low = None

    trend_badges = []
    if history_closes:
        if current_price >= weekly_high:
            trend_badges.append("7D High")
        elif current_price <= weekly_low:
            trend_badges.append("7D Low")
        elif avg_price is not None and current_price > avg_price:
            trend_badges.append("Above Avg")
        else:
            trend_badges.append("Near Avg")
    if abs(change_pct) < 0.5:
        trend_badges.append("Stable")
    elif change_pct > 0:
        trend_badges.append("Recovered")
    else:
        trend_badges.append("Pullback")

    return {
        "change": change,
        "change_pct": round(change_pct, 2),
        "direction_label": direction_label,
        "direction_tone": direction_tone,
        "insight": insight,
        "weekly_high": weekly_high,
        "weekly_low": weekly_low,
        "avg_price": avg_price,
        "range_spread": range_spread,
        "distance_from_high": distance_from_high,
        "distance_from_low": distance_from_low,
        "trend_badges": trend_badges[:3],
        "open_gap": round((current_price - metrics.get("open", current_price)), 2) if metrics.get("open") not in (None, "") else None,
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render(request, "home.html")


@app.get("/signup", response_class=HTMLResponse)
def signup_get(request: Request):
    return render(request, "signup.html", {"errors": [], "form_data": {}})


@app.post("/signup", response_class=HTMLResponse)
def signup_post(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password1: str = Form(...),
    password2: str = Form(...),
):
    errors = []
    form_data = {"username": username, "email": email}

    if password1 != password2:
        errors.append("Passwords do not match.")
    if fetch_user_by_username(username):
        errors.append("Username already exists.")
    if fetch_user_by_email(email):
        errors.append("Email already exists.")
    if len(password1) < 8:
        errors.append("Password must be at least 8 characters long.")

    if errors:
        return render(request, "signup.html", {"errors": errors, "form_data": form_data}, status_code=400)

    create_user(username=username, email=email, password_hash=hash_password(password1))
    add_message(request, "Account created successfully! Please login.")
    return RedirectResponse(url="/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return render(
        request,
        "login.html",
        {
            "errors": [],
            "next_url": request.query_params.get("next", ""),
        },
    )


@app.post("/login", response_class=HTMLResponse)
def login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form(default=""),
):
    user = fetch_user_by_username(username)
    if not user or not verify_password(password, user["password_hash"]):
        return render(
            request,
            "login.html",
            {
                "errors": ["Please enter a correct username and password."],
                "next_url": next,
            },
            status_code=400,
        )

    request.session["user_id"] = user["id"]
    add_message(request, f"Welcome back, {user['username']}!")
    return RedirectResponse(url=next or "/home_user/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/home_user/", response_class=HTMLResponse)
def home_user(request: Request):
    protected = login_required(request)
    if isinstance(protected, RedirectResponse):
        return protected

    stock_data = fetch_popular_stock_quotes(["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META"])
    watchlist_symbols = prediction_page_extras(request)["watchlist_symbols"]
    watchlist_quotes = fetch_popular_stock_quotes(watchlist_symbols[:6]) if watchlist_symbols else []
    try:
        articles = fetch_general_market_news(limit=6)
    except Exception:
        articles = []

    return render(
        request,
        "home_user.html",
        {
            "stock_data": stock_data,
            "articles": articles,
            "watchlist_symbols": watchlist_symbols,
            "watchlist_quotes": watchlist_quotes,
        },
    )


@app.get("/api/stock/")
def get_stock_price(symbol: str = "TSLA"):
    try:
        data = fetch_quote(symbol)
        return {
            "symbol": symbol.upper(),
            "price": data.get("c"),
            "high": data.get("h"),
            "low": data.get("l"),
        }
    except Exception as exc:
        return JSONResponse({"error": f"Failed to fetch data: {exc}"}, status_code=500)


@app.get("/load_trending/", response_class=HTMLResponse)
@app.get("/trending_predictions/", response_class=HTMLResponse)
def trending_predictions(request: Request):
    return render(request, "trending_predictions.html")


@app.get("/search_stock/", response_class=HTMLResponse)
def search_stock(request: Request, symbol: str | None = None):
    symbol = (symbol or "").strip()
    context = {
        "symbol": symbol.upper() if symbol else "",
        "price": None,
        "price_formatted": "-",
        "error": None,
        "dates_json": "[]",
        "prices_json": "[]",
        "change": None,
        "change_pct": None,
        "change_class": "text-slate-500",
        "metrics_display": {
            "open": "-",
            "previous_close": "-",
            "volume": "-",
            "market_cap": "-",
            "fifty_two_high": "-",
            "fifty_two_low": "-",
        },
    }

    if symbol:
        data = {}
        metrics = {}
        history_prices: list[float] = []

        try:
            data = fetch_quote(symbol) or {}
        except Exception:
            data = {}

        try:
            metrics = fetch_company_metrics(symbol) or {}
        except Exception:
            metrics = {}

        try:
            history = fetch_stock_data(symbol, days=7, intraday=False)
            history.columns = [str(column).lower() for column in history.columns]
            if "date" not in history.columns:
                history = history.reset_index()
            if "close" not in history.columns:
                for column in history.columns:
                    if "close" in column:
                        history = history.rename(columns={column: "close"})
                        break
            history["date"] = history["date"].astype(str)
            history_prices = history["close"].astype(float).tail(7).tolist()
            context["dates_json"] = json.dumps([f"Point {index + 1}" for index in range(len(history_prices))])
            context["prices_json"] = json.dumps(history_prices)
        except Exception:
            history_prices = []

        current_price = data.get("c")
        if current_price in (None, 0) and history_prices:
            current_price = history_prices[-1]

        if current_price not in (None, 0):
            change = data.get("d")
            change_pct = data.get("dp")
            previous_close = data.get("pc") or metrics.get("previousClose")
            if (change is None or change_pct is None) and previous_close not in (None, 0):
                change = round(float(current_price) - float(previous_close), 2)
                change_pct = round(((float(current_price) - float(previous_close)) / float(previous_close)) * 100.0, 2)

            change = round(change or 0, 2)
            change_pct = round(change_pct or 0, 2)
            change_class = "text-emerald-600" if change > 0 else "text-red-500" if change < 0 else "text-slate-500"

            context.update(
                {
                    "price": float(current_price),
                    "price_formatted": format_price(current_price),
                    "change": change,
                    "change_pct": change_pct,
                    "change_class": change_class,
                    "metrics_display": {
                        "open": format_price(metrics.get("open")),
                        "previous_close": format_price(metrics.get("previousClose") or data.get("pc")),
                        "volume": format_compact_number(metrics.get("volume")),
                        "market_cap": format_compact_number(metrics.get("market_cap"), prefix="$"),
                        "fifty_two_high": format_price(metrics.get("fiftyTwoWeekHigh")),
                        "fifty_two_low": format_price(metrics.get("fiftyTwoWeekLow")),
                    },
                }
            )
            if not history_prices:
                context["error"] = "Live price loaded, but recent chart data is unavailable right now."
        else:
            context["error"] = "Unable to load stock data for this symbol right now."

    return render(request, "stock_price.html", context)


@app.get("/prediction", response_class=HTMLResponse)
def dashboard_get(request: Request, symbol: str | None = None):
    normalized_symbol = (symbol or "").strip().upper() or None
    context = empty_prediction_context(symbol=normalized_symbol)
    context.update(prediction_page_extras(request, normalized_symbol))
    return render(request, "prediction_dashboard.html", context)


@app.post("/prediction", response_class=HTMLResponse)
def dashboard_post(
    request: Request,
    symbol: str = Form(default=""),
    realtime: str | None = Form(default=None),
    forecast_horizon: int = Form(default=5),
):
    realtime_flag = realtime == "on"
    forecast_horizon = normalize_forecast_horizon(forecast_horizon)
    normalized_symbol = symbol.strip().upper()

    if not normalized_symbol:
        context = empty_prediction_context(
            error="Please enter a stock symbol (e.g., AAPL).",
            realtime_flag=realtime_flag,
            forecast_horizon=forecast_horizon,
        )
        context.update(prediction_page_extras(request))
        return render(
            request,
            "prediction_dashboard.html",
            context,
            status_code=400,
        )

    try:
        context = build_prediction_context(
            normalized_symbol,
            realtime_flag=realtime_flag,
            future_days=forecast_horizon,
        )
        user = get_current_user(request)
        if user and not context.get("error"):
            save_prediction_history(
                int(user["id"]),
                normalized_symbol,
                forecast_horizon,
                context.get("recommendation"),
                context.get("last_close"),
                context.get("pred_next_price"),
            )
    except Exception as exc:
        context = empty_prediction_context(
            symbol=normalized_symbol,
            error=f"Error: {exc}",
            realtime_flag=realtime_flag,
            forecast_horizon=forecast_horizon,
        )
    context.update(prediction_page_extras(request, normalized_symbol))
    return render(request, "prediction_dashboard.html", context)


@app.post("/watchlist/add")
def add_to_watchlist(request: Request, symbol: str = Form(default="")):
    user = get_current_user(request)
    is_ajax = request.headers.get("x-requested-with") == "XMLHttpRequest"
    if not user:
        if is_ajax:
            return JSONResponse({"ok": False, "error": "login_required", "redirect_url": "/login?next=/prediction"}, status_code=401)
        query = urlencode({"next": "/prediction"})
        return RedirectResponse(url=f"/login?{query}", status_code=303)

    normalized_symbol = symbol.strip().upper()
    if normalized_symbol:
        add_watchlist_symbol(int(user["id"]), normalized_symbol)
        add_message(request, f"{normalized_symbol} added to your watchlist.")
        if is_ajax:
            return JSONResponse({"ok": True, "symbol": normalized_symbol, "saved": True})

    if is_ajax:
        return JSONResponse({"ok": False, "error": "missing_symbol"}, status_code=400)

    return RedirectResponse(url=f"/prediction?symbol={normalized_symbol}" if normalized_symbol else "/prediction", status_code=303)


@app.post("/watchlist/remove")
def remove_from_watchlist(request: Request, symbol: str = Form(default="")):
    user = get_current_user(request)
    if not user:
        query = urlencode({"next": "/home_user/"})
        return RedirectResponse(url=f"/login?{query}", status_code=303)

    normalized_symbol = symbol.strip().upper()
    if normalized_symbol:
        remove_watchlist_symbol(int(user["id"]), normalized_symbol)
        add_message(request, f"{normalized_symbol} removed from your watchlist.")

    return RedirectResponse(url="/home_user/", status_code=303)


@app.get("/api/stock-data/{symbol}/")
def stock_data_api(symbol: str, forecast_horizon: int = 5):
    try:
        return build_stock_chart_api_payload(symbol.upper(), future_days=normalize_forecast_horizon(forecast_horizon))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/market_news/", response_class=HTMLResponse)
def market_news(request: Request):
    try:
        articles = fetch_general_market_news(limit=30)
    except Exception:
        articles = []

    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JSONResponse({"articles": articles})

    return render(request, "market_news.html", {"articles": articles})
