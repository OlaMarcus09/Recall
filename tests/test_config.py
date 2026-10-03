from pathlib import Path

from recall.config import Settings


def test_settings_read_safe_environment_values(monkeypatch, tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    upload_dir = tmp_path / "uploads"

    monkeypatch.setenv("RECALL_ENV", "test")
    monkeypatch.setenv("RECALL_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("RECALL_UPLOAD_DIR", str(upload_dir))

    settings = Settings.from_env()

    assert settings.environment == "test"
    assert settings.database_path == database_path
    assert settings.upload_dir == upload_dir
    assert isinstance(settings.database_path, Path)
