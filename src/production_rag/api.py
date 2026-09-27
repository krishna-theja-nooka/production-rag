import json
import logging
import secrets
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from threading import BoundedSemaphore, Lock

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .generation import GenerationError
from .service import RAGService

logger = logging.getLogger("production_rag")
key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=8)

    @field_validator("question")
    @classmethod
    def meaningful(cls, value: str):
        if len(value.strip()) < 3:
            raise ValueError("Question must contain at least three non-space characters")
        return value.strip()


class Citation(BaseModel):
    citation: int
    chunk_id: str
    source: str
    page: int
    text: str


class Answer(BaseModel):
    answer: str
    abstained: bool
    mode: str
    retrieval: str
    citations: list[Citation]


def create_app(settings: Settings | None = None, service: RAGService | None = None):
    cfg = settings or Settings()
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    logger.propagate = False
    lock = Lock()
    calls: deque[float] = deque()
    slots = BoundedSemaphore(cfg.concurrent_queries)
    metrics = {"queries": 0, "abstentions": 0, "errors": 0}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if len(cfg.api_key) < 24:
            raise RuntimeError("Set RAG_API_KEY to a random secret of at least 24 characters")
        app.state.rag = service or RAGService(cfg)
        yield

    app = FastAPI(title="Production RAG Reference API", version="0.2.0", lifespan=lifespan)
    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_methods=["POST", "GET"],
            allow_headers=["X-API-Key", "Content-Type"],
        )

    @app.middleware("http")
    async def envelope(request: Request, call_next):
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        if request.method == "POST":
            body = bytearray()
            async for part in request.stream():
                body.extend(part)
                if len(body) > 16_384:
                    return JSONResponse({"detail": "Request body too large"}, status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        # Do not log request bodies, headers, questions, excerpts, or arbitrary URL paths.
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "status": response.status_code,
                    "duration_ms": round((time.monotonic() - started) * 1000),
                }
            )
        )
        return response

    def authenticate(key: str | None = Security(key_header)):
        if not key or not secrets.compare_digest(key, cfg.api_key):
            raise HTTPException(401, "Invalid API key")

    def rate_limit():
        now = time.monotonic()
        with lock:
            while calls and calls[0] <= now - 60:
                calls.popleft()
            if len(calls) >= cfg.requests_per_minute:
                raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})
            calls.append(now)

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready", dependencies=[Depends(authenticate)])
    def ready():
        docs = app.state.rag.store.documents()
        if not docs:
            raise HTTPException(503, "No indexed documents")
        # Dependency readiness is checked separately with `rag doctor`.
        return {
            "status": "index_ready",
            "documents": len(docs),
            "generator": cfg.generator,
            "retrieval": cfg.retrieval,
        }

    @app.post("/v1/chat", response_model=Answer, dependencies=[Depends(authenticate)])
    def chat(body: Question):
        rate_limit()
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "Server busy", headers={"Retry-After": "5"})
        try:
            result = app.state.rag.ask(body.question, body.top_k)
            with lock:
                metrics["queries"] += 1
                metrics["abstentions"] += int(result["abstained"])
            return result
        except GenerationError as exc:
            with lock:
                metrics["errors"] += 1
            raise HTTPException(502, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, "Question or index configuration is invalid") from exc
        finally:
            slots.release()

    @app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(authenticate)])
    def observe():
        with lock:
            return "".join(f"rag_{key}_total {value}\n" for key, value in metrics.items())

    return app
