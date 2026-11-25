from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .exceptions import InvalidBinsError, InvalidImageError


def validate_bins(bins: int) -> int:
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins <= 0:
        raise InvalidBinsError(f"Numărul de intervale trebuie să fie întreg pozitiv, primit: {bins!r}")
    return int(bins)


def load_rgb(path: str | Path) -> Image.Image:
    try:
        with Image.open(path) as img:
            img.load()
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise InvalidImageError(f"Imagine invalidă sau coruptă: {path}") from exc


def histogram_from_array(pixels: np.ndarray, bins: int) -> np.ndarray:
    bins = validate_bins(bins)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.size == 0:
        raise InvalidImageError("Tablou de pixeli invalid (se așteaptă HxWx3 nevid).")
    flat = pixels.reshape(-1, 3).astype(np.int64)
    idx = (flat * bins) // 256
    parts = []
    total = flat.shape[0]
    for ch in range(3):
        counts = np.bincount(idx[:, ch], minlength=bins).astype(np.float64)
        parts.append(counts / total)
    return np.concatenate(parts).astype(np.float32)


def compute_descriptor(path: str | Path, bins: int) -> np.ndarray:
    bins = validate_bins(bins)
    img = load_rgb(path)
    return histogram_from_array(np.asarray(img, dtype=np.uint8), bins)
