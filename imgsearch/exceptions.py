from __future__ import annotations


class ImgSearchError(Exception):
    """Baza excepțiilor aplicației."""


class InvalidImageError(ImgSearchError):
    """Fișierul nu este o imagine validă sau nu poate fi citit/decodat."""


class IndexNotFoundError(ImgSearchError):
    """Indexul (baza SQLite) nu există sau este gol."""


class InvalidBinsError(ImgSearchError):
    """Numărul de intervale este invalid (trebuie să fie întreg pozitiv)."""


class CatalogNotFoundError(ImgSearchError):
    """Folderul catalogului nu există."""