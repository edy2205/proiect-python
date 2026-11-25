from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImageRecord:
    id: int
    file_name: str
    rel_path: str
    source: str
    sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class SearchResult:
    image_id: int
    file_name: str
    rel_path: str
    distance: float


@dataclass
class ImportReport:
    added: int = 0
    skipped_duplicates: int = 0
    invalid: list[tuple[str, str]] = field(default_factory=list)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid)
