from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

from fastapi import APIRouter, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from recall.answering import (
    LearnerAnswer,
    QuestionValidationError,
    answer_question,
    normalize_question,
)
from recall.config import Settings
from recall.confusion.service import ConfusionCluster, list_recurring_confusions
from recall.db import current_schema_version
from recall.generation.service import REFUSAL_MESSAGE, SupportStatus
from recall.retrieval.embeddings import build_embedding_provider


TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def create_router(settings: Settings) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=page_context(settings, request=request),
        )

    @router.post("/ask", response_class=HTMLResponse)
    async def ask(request: Request) -> HTMLResponse:
        question = await question_from_form(request)
        result = None
        error = None
        status_code = status.HTTP_200_OK

        try:
            normalized_question = normalize_question(question)
            result = answer_question(
                database_path=settings.database_path,
                question=normalized_question,
                embedding_provider=build_embedding_provider(settings),
                remember_question=True,
            )
            question = result.question
        except QuestionValidationError as exc:
            error = str(exc)
            question = question.strip()
            status_code = status.HTTP_400_BAD_REQUEST
        except Exception:
            question = question.strip()
            result = LearnerAnswer(
                question=question,
                status=SupportStatus.NOT_FOUND,
                answer=REFUSAL_MESSAGE,
                citations=[],
            )

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=page_context(
                settings,
                request=request,
                question=question,
                result=result,
                error=error,
            ),
            status_code=status_code,
        )

    @router.post("/api/ask")
    async def ask_json(request: Request) -> JSONResponse:
        payload = await request.json()
        question = str(payload.get("question", ""))
        try:
            normalized_question = normalize_question(question)
            result = answer_question(
                database_path=settings.database_path,
                question=normalized_question,
                embedding_provider=build_embedding_provider(settings),
                remember_question=True,
            )
        except QuestionValidationError as exc:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": str(exc)},
            )
        except Exception:
            normalized_question = question.strip()
            result = LearnerAnswer(
                question=normalized_question,
                status=SupportStatus.NOT_FOUND,
                answer=REFUSAL_MESSAGE,
                citations=[],
            )

        return JSONResponse(
            content={
                "question": result.question,
                "status": result.status.value,
                "answer": result.answer,
                "citations": [
                    {
                        "knowledge_unit_id": citation.knowledge_unit_id,
                        "material_id": citation.material_id,
                        "document_title": citation.document_title,
                        "original_filename": citation.original_filename,
                        "page_number": citation.page_number,
                        "chunk_index": citation.chunk_index,
                        "evidence_excerpt": citation.evidence_excerpt,
                    }
                    for citation in result.citations
                ],
            }
        )

    @router.get("/api/confusions")
    async def confusions() -> JSONResponse:
        clusters = list_recurring_confusions(settings.database_path)
        return JSONResponse(
            content={
                "clusters": [
                    {
                        "cluster_id": cluster.cluster_id,
                        "representative_question": cluster.representative_question,
                        "recurrence_count": cluster.recurrence_count,
                        "first_seen": cluster.first_seen,
                        "last_seen": cluster.last_seen,
                    }
                    for cluster in clusters
                ]
            }
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


def page_context(
    settings: Settings,
    *,
    request: Request,
    question: str = "",
    result: LearnerAnswer | None = None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "request": request,
        "app_name": settings.app_name,
        "environment": settings.environment,
        "question": question,
        "result": result,
        "error": error,
        "confusion_clusters": safe_recurring_confusions(settings),
    }


def safe_recurring_confusions(settings: Settings) -> list[ConfusionCluster]:
    try:
        return list_recurring_confusions(settings.database_path)
    except Exception:
        return []


async def question_from_form(request: Request) -> str:
    body = (await request.body()).decode("utf-8")
    values = parse_qs(body, keep_blank_values=True)
    return values.get("question", [""])[0]
