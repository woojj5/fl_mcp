"""Small SQLite store shared by the dashboard and deployment server."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def connect(path: str | Path) -> Iterator[sqlite3.Connection]:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize(path: str | Path) -> None:
    with connect(path) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                task_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dataset TEXT NOT NULL,
                strategy TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'running',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS round_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                round INTEGER NOT NULL,
                loss REAL,
                accuracy REAL,
                clients INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(task_id, round)
            );
            CREATE TABLE IF NOT EXISTS client_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                client_id TEXT NOT NULL,
                round INTEGER NOT NULL,
                samples INTEGER NOT NULL DEFAULT 0,
                loss REAL,
                accuracy REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def record_run(path: str | Path, task_id: str, model: str, dataset: str, strategy: str) -> None:
    initialize(path)
    with connect(path) as db:
        db.execute(
            "INSERT INTO runs(task_id, model, dataset, strategy, status) VALUES (?, ?, ?, ?, 'running') "
            "ON CONFLICT(task_id) DO UPDATE SET model=excluded.model, dataset=excluded.dataset, "
            "strategy=excluded.strategy, status='running'",
            (task_id, model, dataset, strategy),
        )


def record_round(path: str | Path, task_id: str, round_number: int, clients: int,
                 loss: float | None = None, accuracy: float | None = None) -> None:
    initialize(path)
    with connect(path) as db:
        db.execute(
            "INSERT INTO round_metrics(task_id, round, loss, accuracy, clients) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(task_id, round) DO UPDATE SET loss=excluded.loss, "
            "accuracy=excluded.accuracy, clients=excluded.clients",
            (task_id, round_number, loss, accuracy, clients),
        )


def record_client(path: str | Path, task_id: str, client_id: str, round_number: int,
                  samples: int, loss: float | None, accuracy: float | None) -> None:
    initialize(path)
    with connect(path) as db:
        db.execute(
            "INSERT INTO client_metrics(task_id, client_id, round, samples, loss, accuracy) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, client_id, round_number, samples, loss, accuracy),
        )


def get_runs(path: str | Path) -> list[dict[str, Any]]:
    initialize(path)
    with connect(path) as db:
        return [dict(row) for row in db.execute("SELECT * FROM runs ORDER BY created_at DESC")]


def get_rounds(path: str | Path, task_id: str | None = None) -> list[dict[str, Any]]:
    initialize(path)
    query = "SELECT * FROM round_metrics"
    params: tuple[Any, ...] = ()
    if task_id:
        query += " WHERE task_id=?"
        params = (task_id,)
    query += " ORDER BY round DESC LIMIT 100"
    with connect(path) as db:
        return [dict(row) for row in db.execute(query, params)]


def get_clients(path: str | Path, task_id: str | None = None) -> list[dict[str, Any]]:
    initialize(path)
    query = "SELECT * FROM client_metrics"
    params: tuple[Any, ...] = ()
    if task_id:
        query += " WHERE task_id=?"
        params = (task_id,)
    query += " ORDER BY id DESC LIMIT 200"
    with connect(path) as db:
        return [dict(row) for row in db.execute(query, params)]
