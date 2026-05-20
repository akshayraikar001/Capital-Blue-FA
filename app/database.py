import sqlite3
from contextlib import contextmanager

from .config import DATABASE_PATH


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


@contextmanager
def get_connection():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, symbol),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                forecast_horizon INTEGER NOT NULL,
                recommendation TEXT,
                last_close REAL,
                predicted_price REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )


def fetch_user_by_username(username: str):
    init_db()
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        ).fetchone()


def fetch_user_by_email(email: str):
    init_db()
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,),
        ).fetchone()


def fetch_user_by_id(user_id: int):
    init_db()
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()


def create_user(username: str, email: str, password_hash: str) -> None:
    init_db()
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )


def add_watchlist_symbol(user_id: int, symbol: str) -> None:
    init_db()
    with get_connection() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO watchlist (user_id, symbol) VALUES (?, ?)",
            (user_id, symbol.upper()),
        )


def fetch_watchlist_symbols(user_id: int):
    init_db()
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT symbol, created_at
            FROM watchlist
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()


def remove_watchlist_symbol(user_id: int, symbol: str) -> None:
    init_db()
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM watchlist WHERE user_id = ? AND symbol = ?",
            (user_id, symbol.upper()),
        )


def save_prediction_history(
    user_id: int,
    symbol: str,
    forecast_horizon: int,
    recommendation: str | None,
    last_close: float | None,
    predicted_price: float | None,
) -> None:
    init_db()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO prediction_history (
                user_id, symbol, forecast_horizon, recommendation, last_close, predicted_price
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, symbol.upper(), forecast_horizon, recommendation, last_close, predicted_price),
        )


def fetch_prediction_history(user_id: int, limit: int = 6):
    init_db()
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT symbol, forecast_horizon, recommendation, last_close, predicted_price, created_at
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
