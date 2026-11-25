from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .descriptor import validate_bins
from .exceptions import IndexNotFoundError
from .models import ImageRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    rel_path  TEXT NOT NULL,
    source    TEXT NOT NULL,
    sha256    TEXT NOT NULL UNIQUE,
    width     INTEGER NOT NULL,
    height    INTEGER NOT NULL,
    added_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS features (
    image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    bins        INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    config_json TEXT NOT NULL,
    PRIMARY KEY (image_id, bins)
);
CREATE TABLE IF NOT EXISTS eval_queries (
    query_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id        INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    relevance_group TEXT NOT NULL,
    notes           TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS search_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query_path      TEXT NOT NULL,
    bins            INTEGER NOT NULL,
    result_ids_json TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @classmethod
    def open_existing(cls, db_path: str | Path) -> "Storage":
        if not Path(db_path).is_file():
            raise IndexNotFoundError(f"Indexul nu există: {db_path}")
        return cls(db_path)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ImageRecord:
        return ImageRecord(
            id=row["id"],
            file_name=row["file_name"],
            rel_path=row["rel_path"],
            source=row["source"],
            sha256=row["sha256"],
            width=row["width"],
            height=row["height"],
        )

    def get_image_by_hash(self, sha256: str) -> ImageRecord | None:
        row = self._conn.execute("SELECT * FROM images WHERE sha256 = ?", (sha256,)).fetchone()
        return self._to_record(row) if row else None

    def get_image(self, image_id: int) -> ImageRecord | None:
        row = self._conn.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return self._to_record(row) if row else None

    def get_image_by_id(self, image_id: int) -> ImageRecord | None:
        return self.get_image(image_id)

    def find_image_by_name(self, file_name: str) -> ImageRecord | None:
        row = self._conn.execute(
            "SELECT * FROM images WHERE file_name = ? OR rel_path = ? ORDER BY id LIMIT 1",
            (file_name, file_name),
        ).fetchone()
        return self._to_record(row) if row else None

    def add_image(
        self,
        file_name: str,
        rel_path: str,
        source: str,
        sha256: str,
        width: int,
        height: int,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO images (file_name, rel_path, source, sha256, width, height, added_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (file_name, rel_path, source, sha256, width, height, _now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_images(self) -> list[ImageRecord]:
        rows = self._conn.execute("SELECT * FROM images ORDER BY id").fetchall()
        return [self._to_record(r) for r in rows]

    def count_images(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    # ---------------------------------------------------------- features
    def has_features(self, image_id: int, bins: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM features WHERE image_id = ? AND bins = ?", (image_id, bins)
        ).fetchone()
        return row is not None

    def save_features(self, image_id: int, bins: int, vector: np.ndarray, config: dict | None = None) -> None:
        bins = validate_bins(bins)
        vec = np.asarray(vector, dtype=np.float32)
        if vec.shape != (3 * bins,):
            raise ValueError(f"Vector de lungime {vec.shape} pentru bins={bins}")
        cfg = json.dumps(config or {"descriptor": "rgb_histogram", "bins": bins, "distance": "L1"})
        self._conn.execute(
            "INSERT OR REPLACE INTO features (image_id, bins, vector, config_json) VALUES (?, ?, ?, ?)",
            (image_id, bins, vec.tobytes(), cfg),
        )
        self._conn.commit()

    def add_features(self, image_id: int, bins: int, vector: np.ndarray, config: dict | None = None) -> None:
        self.save_features(image_id, bins, vector, config)

    def load_features(self, bins: int) -> tuple[np.ndarray, np.ndarray]:
        bins = validate_bins(bins)
        rows = self._conn.execute(
            "SELECT image_id, vector FROM features WHERE bins = ? ORDER BY image_id", (bins,)
        ).fetchall()
        if not rows:
            raise IndexNotFoundError(f"Nu există caracteristici salvate pentru bins={bins}.")
        ids = np.array([r["image_id"] for r in rows], dtype=np.int64)
        matrix = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        return ids, matrix

    def indexed_bins(self) -> list[int]:
        rows = self._conn.execute("SELECT DISTINCT bins FROM features ORDER BY bins").fetchall()
        return [int(r[0]) for r in rows]

    def replace_eval_queries(self, queries: list[tuple[int, str, str]]) -> None:
        self._conn.execute("DELETE FROM eval_queries")
        self._conn.executemany(
            "INSERT INTO eval_queries (image_id, relevance_group, notes) VALUES (?, ?, ?)", queries
        )
        self._conn.commit()

    def load_eval_queries(self) -> list[tuple[int, int, str, str]]:
        rows = self._conn.execute(
            "SELECT query_id, image_id, relevance_group, notes FROM eval_queries ORDER BY query_id"
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def log_search(self, query_path: str, bins: int, result_ids: list[int]) -> None:
        self._conn.execute(
            "INSERT INTO search_log (query_path, bins, result_ids_json, created_at) VALUES (?, ?, ?, ?)",
            (query_path, bins, json.dumps(result_ids), _now()),
        )
        self._conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None
