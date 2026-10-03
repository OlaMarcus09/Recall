from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "Recall"
    environment: str = "development"
    database_path: Path = Path("data/recall.sqlite3")
    upload_dir: Path = Path("uploads")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            environment=os.getenv("RECALL_ENV", cls.environment),
            database_path=Path(os.getenv("RECALL_DATABASE_PATH", str(cls.database_path))),
            upload_dir=Path(os.getenv("RECALL_UPLOAD_DIR", str(cls.upload_dir))),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
