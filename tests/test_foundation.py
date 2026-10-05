from fastapi.testclient import TestClient

from recall.config import Settings
from recall.db import current_schema_version
from recall.main import create_app


def make_client(tmp_path) -> tuple[TestClient, Settings]:
    settings = Settings(
        environment="test",
        database_path=tmp_path / "recall.sqlite3",
        upload_dir=tmp_path / "uploads",
    )
    return TestClient(create_app(settings)), settings


def test_health_endpoint_bootstraps_sqlite(tmp_path):
    with make_client(tmp_path)[0] as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "Recall",
        "environment": "test",
        "database": "ok",
        "schema_version": 4,
    }


def test_sqlite_schema_version_is_available_after_startup(tmp_path):
    client, settings = make_client(tmp_path)

    with client:
        pass

    assert current_schema_version(settings.database_path) == 4


def test_home_page_renders(tmp_path):
    with make_client(tmp_path)[0] as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Recall" in response.text
    assert "Ask Recall" in response.text
