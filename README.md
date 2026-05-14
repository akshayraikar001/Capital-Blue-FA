# Capital Blue FastAPI

FastAPI rebuild of the original Django Capital Blue project, keeping the same UI, routes, stock/news integrations, and reusable LSTM model assets where possible.

## Current setup note

This folder reuses the original Django project's installed ML/data packages through:

`.\.venv\Lib\site-packages\capitalblue_source_sitepackages.pth`

That keeps the existing TensorFlow, pandas, yfinance, and Finnhub stack usable here without retraining or redownloading the heavy model dependencies. If the original `Capital-Blue` folder moves, update that `.pth` file.

## Run

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`

## Routes

- `/`
- `/signup`
- `/login`
- `/logout`
- `/home_user/`
- `/search_stock/`
- `/prediction`
- `/market_news/`
- `/api/stock/`
- `/api/stock-data/{symbol}/`
