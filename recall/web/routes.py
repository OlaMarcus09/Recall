from __future__ import annotations

import ipaddress
import tempfile
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default
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
from recall import db
from recall.db import current_schema_version
from recall.generation.service import REFUSAL_MESSAGE, SupportStatus
from recall.ingestion.pdf import PdfExtractionError
from recall.ingestion.service import IngestionResult, ingest_pdf
from recall.knowledge.service import (
    KnowledgeUnit,
    KnowledgeUnitGenerationError,
    generate_knowledge_units_for_material,
)
from recall.retrieval.embeddings import build_embedding_provider
from recall.retrieval.index import IndexingError, IndexingResult, index_material_embeddings


TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_MULTIPART_BODY_BYTES = MAX_UPLOAD_BYTES + 64 * 1024


class UploadError(ValueError):
    """Raised when a local PDF upload cannot be accepted."""


class UploadTooLargeError(UploadError):
    """Raised when an uploaded PDF exceeds the configured size limit."""


@dataclass(frozen=True)
class MaterialUploadResult:
    ingestion: IngestionResult
    knowledge_units: list[KnowledgeUnit]
    index: IndexingResult

    @property
    def created(self) -> bool:
        return self.ingestion.created


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

    @router.post("/materials/upload", response_class=HTMLResponse)
    async def upload_material(request: Request) -> HTMLResponse:
        upload_result: MaterialUploadResult | None = None
        upload_error: str | None = None
        status_code = status.HTTP_200_OK

        try:
            ensure_loopback_client(request)
            filename, pdf_bytes = await uploaded_pdf_from_request(request)
            upload_result = prepare_uploaded_material(
                settings=settings,
                filename=filename,
                pdf_bytes=pdf_bytes,
            )
        except UploadTooLargeError as exc:
            upload_error = str(exc)
            status_code = status.HTTP_413_CONTENT_TOO_LARGE
        except PermissionError:
            upload_error = "PDF upload is available only from this local machine."
            status_code = status.HTTP_403_FORBIDDEN
        except (UploadError, PdfExtractionError) as exc:
            upload_error = upload_error_message(exc)
            status_code = status.HTTP_400_BAD_REQUEST
        except (KnowledgeUnitGenerationError, IndexingError):
            upload_error = (
                "The PDF was accepted, but Recall could not finish preparing "
                "it for search."
            )
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        except Exception:
            upload_error = "Recall could not prepare that PDF for search."
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=page_context(
                settings,
                request=request,
                upload_result=upload_result,
                upload_error=upload_error,
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
    upload_result: MaterialUploadResult | None = None,
    upload_error: str | None = None,
) -> dict[str, object]:
    return {
        "request": request,
        "app_name": settings.app_name,
        "environment": settings.environment,
        "question": question,
        "result": result,
        "error": error,
        "upload_result": upload_result,
        "upload_error": upload_error,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
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


def ensure_loopback_client(request: Request) -> None:
    host = request.client.host if request.client else ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise PermissionError("Upload client is not a loopback address.") from exc
    if not address.is_loopback:
        raise PermissionError("Upload client is not a loopback address.")


async def uploaded_pdf_from_request(request: Request) -> tuple[str, bytes]:
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type.lower():
        raise UploadError("Choose a PDF file before uploading.")

    body = await limited_request_body(request)
    filename, pdf_bytes = parse_uploaded_pdf(content_type, body)
    if not pdf_bytes:
        raise UploadError("Choose a PDF file before uploading.")
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError("PDF uploads are limited to 10 MB.")
    return filename, pdf_bytes


async def limited_request_body(request: Request) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_MULTIPART_BODY_BYTES:
            raise UploadTooLargeError("PDF uploads are limited to 10 MB.")
        chunks.append(chunk)
    return b"".join(chunks)


def parse_uploaded_pdf(content_type: str, body: bytes) -> tuple[str, bytes]:
    header = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n"
        "\r\n"
    ).encode("utf-8")
    message = BytesParser(policy=default).parsebytes(header + body)
    if not message.is_multipart():
        raise UploadError("Choose a PDF file before uploading.")

    for part in message.iter_parts():
        disposition = part.get_content_disposition()
        name = part.get_param("name", header="content-disposition")
        if disposition != "form-data" or name != "material_pdf":
            continue
        filename = part.get_filename() or "uploaded.pdf"
        payload = part.get_payload(decode=True) or b""
        return filename, payload

    raise UploadError("Choose a PDF file before uploading.")


def prepare_uploaded_material(
    *,
    settings: Settings,
    filename: str,
    pdf_bytes: bytes,
) -> MaterialUploadResult:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="recall-upload-",
            suffix=".pdf",
            dir=settings.upload_dir,
            delete=False,
        ) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = Path(temp_file.name)

        db.initialize_database(settings.database_path)
        saved_pdf = temp_path.read_bytes()
        ingestion = ingest_pdf(
            database_path=settings.database_path,
            pdf_bytes=saved_pdf,
            original_filename=filename,
        )
        knowledge_units = generate_knowledge_units_for_material(
            settings.database_path,
            ingestion.material.id,
        )
        index = index_material_embeddings(
            settings.database_path,
            ingestion.material.id,
            build_embedding_provider(settings),
        )
        return MaterialUploadResult(
            ingestion=ingestion,
            knowledge_units=knowledge_units,
            index=index,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def upload_error_message(exc: Exception) -> str:
    message = str(exc)
    if "no meaningful extractable text" in message:
        return "The uploaded PDF has no extractable text."
    if "Encrypted PDFs are not supported" in message:
        return "Encrypted PDFs are not supported."
    if "Only PDF files" in message:
        return "Upload a valid text-extractable PDF file."
    if not message:
        return "Recall could not prepare that PDF for search."
    return message
