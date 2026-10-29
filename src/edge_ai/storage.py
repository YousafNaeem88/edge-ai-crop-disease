from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class HistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_name TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def add(self, label: str, confidence: float, model_name: str) -> None:
        if not label or not 0 <= confidence <= 1:
            raise ValueError("label must be non-empty and confidence must be between 0 and 1")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO predictions(created_at, predicted_class, confidence, model_name) VALUES (?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(), label, confidence, model_name),
            )

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT created_at, predicted_class, confidence, model_name "
                "FROM predictions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM predictions")

