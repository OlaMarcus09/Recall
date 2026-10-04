from pathlib import Path

from recall.config import Settings


def test_settings_read_safe_environment_values(monkeypatch, tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    upload_dir = tmp_path / "uploads"
    fastembed_cache_dir = tmp_path / "fastembed-cache"

    monkeypatch.setenv("RECALL_ENV", "test")
    monkeypatch.setenv("RECALL_DATABASE_PATH", str(database_path))
    monkeypatch.setenv("RECALL_UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("RECALL_EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("RECALL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("RECALL_FASTEMBED_CACHE_DIR", str(fastembed_cache_dir))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = Settings.from_env()

    assert settings.environment == "test"
    assert settings.database_path == database_path
    assert settings.upload_dir == upload_dir
    assert settings.embedding_provider == "fastembed"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.fastembed_cache_dir == fastembed_cache_dir
    assert settings.openai_api_key is None
    assert isinstance(settings.database_path, Path)


def test_openai_key_is_read_when_configured(monkeypatch):
    monkeypatch.setenv("RECALL_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("RECALL_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_API_KEY", "safe-test-placeholder")
    monkeypatch.delenv("RECALL_FASTEMBED_CACHE_DIR", raising=False)

    settings = Settings.from_env()

    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.openai_api_key == "safe-test-placeholder"
    assert settings.fastembed_cache_dir is None
