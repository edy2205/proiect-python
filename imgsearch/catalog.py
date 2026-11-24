from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import SUPPORTED_EXTENSIONS
from .descriptor import histogram_from_array, load_rgb, validate_bins
from .exceptions import CatalogNotFoundError, InvalidImageError
from .hashing import sha256_file
from .models import ImportReport
from .storage import Storage

SOURCES_FILE = "sources.json"


def list_image_files(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        raise CatalogNotFoundError(f"Folderul catalogului nu există: {root}")
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def load_sources(folder: str | Path) -> dict[str, str]:
    path = Path(folder) / SOURCES_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def index_folder(folder: str | Path, storage: Storage, bins: int) -> ImportReport:
    bins = validate_bins(bins)
    root = Path(folder).resolve()
    sources = load_sources(root)
    report = ImportReport()
    storage.set_meta("catalog_root", str(root))

    for path in list_image_files(root):
        rel = path.relative_to(root).as_posix()
        try:
            img = load_rgb(path)
            digest = sha256_file(path)
            vector = histogram_from_array(np.asarray(img, dtype=np.uint8), bins)
        except (InvalidImageError, OSError) as exc:
            report.invalid.append((rel, str(exc)))
            continue

        existing = storage.get_image_by_hash(digest)
        if existing is not None:
            report.skipped_duplicates += 1
            if not storage.has_features(existing.id, bins):
                storage.add_features(existing.id, bins, vector)
            continue

        source = sources.get(path.name, f"folder:{path.parent.name or 'catalog'}")
        image_id = storage.add_image(
            file_name=path.name,
            rel_path=rel,
            source=source,
            sha256=digest,
            width=img.width,
            height=img.height,
        )
        storage.add_features(image_id, bins, vector)
        report.added += 1

    return report