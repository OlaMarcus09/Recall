from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from recall.config import Settings, get_settings
from recall.db import initialize_database
from recall.web.routes import create_router


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        initialize_database(app_settings.database_path)
        app.state.settings = app_settings
        yield

    app = FastAPI(title=app_settings.app_name, lifespan=lifespan)
    app.include_router(create_router(app_settings))
    return app


app = create_app()
