from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

BINS: int = 8
TOP_K: int = 5

DEFAULT_IMAGES_DIR: Path = PROJECT_ROOT / "data" / "images"
DEFAULT_DB_PATH: Path = PROJECT_ROOT / "data" / "index.sqlite3"
DEFAULT_RELEVANCE_PATH: Path = PROJECT_ROOT / "data" / "relevance.json"
DEFAULT_EXPORT_DIR: Path = PROJECT_ROOT / "exports"

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
)
