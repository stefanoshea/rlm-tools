import json
import os
import pathlib
import time

CACHE_DIR = ".rlm_cache"
METADATA_FILE = "metadata.json"
FILE_INDEX_FILE = "file_index.json"
CACHE_VERSION = 2


class DirectoryCache:
    """Кэш метаданных и индекса файлов для директории, хранится в .rlm_cache/.
    
    Кэш считается валидным пока существует и имеет правильную версию.
    Инвалидация — только через force_rescan=true в rlm_start.
    """

    def __init__(self, base_path: str):
        self.base = pathlib.Path(base_path).resolve()
        self.cache_dir = self.base / CACHE_DIR

    def _read_json(self, filename: str) -> dict | None:
        cache_file = self.cache_dir / filename
        try:
            with open(cache_file, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _write_json(self, filename: str, data: dict | list) -> bool:
        try:
            self.cache_dir.mkdir(exist_ok=True)
            cache_file = self.cache_dir / filename
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except OSError:
            return False

    def _is_cache_valid(self, entry: dict) -> bool:
        return entry.get("version") == CACHE_VERSION

    def get_metadata(self) -> dict | None:
        """Вернуть кэш метаданных, если валиден, иначе None."""
        entry = self._read_json(METADATA_FILE)
        if entry is None or not self._is_cache_valid(entry):
            return None
        return entry.get("data")

    def save_metadata(self, metadata: dict) -> None:
        """Сохранить метаданные в кэш."""
        entry = {
            "version": CACHE_VERSION,
            "saved_at": time.time(),
            "data": metadata,
        }
        self._write_json(METADATA_FILE, entry)

    def get_file_index(self) -> list[str] | None:
        """Вернуть кэшированный список файлов (пути относительно base), если валиден."""
        entry = self._read_json(FILE_INDEX_FILE)
        if entry is None or not self._is_cache_valid(entry):
            return None
        return entry.get("files")

    def save_file_index(self, files: list[str]) -> None:
        """Сохранить индекс файлов в кэш."""
        entry = {
            "version": CACHE_VERSION,
            "saved_at": time.time(),
            "files": files,
        }
        self._write_json(FILE_INDEX_FILE, entry)

    def invalidate(self) -> None:
        """Удалить все файлы кэша."""
        for filename in (METADATA_FILE, FILE_INDEX_FILE):
            cache_file = self.cache_dir / filename
            try:
                cache_file.unlink(missing_ok=True)
            except OSError:
                pass
