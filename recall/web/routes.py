from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from recall.config import Settings
from recall.db import current_schema_version


TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def create_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "app_name": settings.app_name,
                "environment": settings.environment,
            },
        )

    @router.get("/health")
    async def health() -> dict[str, str | int | None]:
        return {
            "status": "ok",
            "app": settings.app_name,
            "environment": settings.environment,
            "database": "ok",
            "schema_version": current_schema_version(settings.database_path),
        }

    return router
