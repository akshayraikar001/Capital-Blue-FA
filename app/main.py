import json
from urllib.parse import urlencode

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.config import SESSION_SECRET, STATIC_DIR, TEMPLATES_DIR
from app.database import (
    create_user,
    fetch_user_by_email,
    fetch_user_by_id,
    fetch_user_by_username,
    init_db,
)
from app.security import hash_password, verify_password
from app.services.market_data import (
    fetch_general_market_news,
    fetch_popular_stock_quotes,
    fetch_quote,
    fetch_stock_data,
)
from app.services.prediction import build_prediction_context, build_stock_chart_api_payload

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
    context = {
        "symbol": symbol.upper() if symbol else "",
        "price": None,
        "error": None,
        "dates_json": "[]",
        "prices_json": "[]",
    }

    if symbol:
        try:
            data = fetch_quote(symbol)
            current_price = data.get("c")
            if current_price and current_price != 0:
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
                context.update(
                    {
                        "price": current_price,
                        "dates_json": json.dumps(history["date"].tail(7).tolist()),
                        "prices_json": json.dumps(history["close"].astype(float).tail(7).tolist()),
                    }
                )
            else:
                context["error"] = "No valid price data found for this stock."
        except Exception:
            context["error"] = "API request failed. Please check your API key or internet connection."

    return render(request, "stock_price.html", context)


@app.get("/prediction", response_class=HTMLResponse)
def dashboard_get(request: Request):
    context = {
        "symbol": None,
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
        "realtime_flag": False,
    }
    return render(request, "prediction_dashboard.html", context)


@app.post("/prediction", response_class=HTMLResponse)
def dashboard_post(
    request: Request,
    symbol: str = Form(...),
    realtime: str | None = Form(default=None),
):
    realtime_flag = realtime == "on"
    try:
        context = build_prediction_context(symbol.strip().upper(), realtime_flag=realtime_flag)
    except Exception as exc:
        context = {
            "symbol": symbol.strip().upper(),
            "error": f"Error: {exc}",
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
    return render(request, "prediction_dashboard.html", context)


@app.get("/api/stock-data/{symbol}/")
def stock_data_api(symbol: str):
    try:
        return build_stock_chart_api_payload(symbol.upper())
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
