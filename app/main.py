"""ASGI app factory.

`uvicorn app.main:app` uses the module-level app, which reads `API_KEY` from the
environment already set by the process (or `uvicorn --env-file .env`).
`python -m app.main` loads a local `.env` first, without overriding real env vars.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

from app.config import Settings, load_settings
from app.errors import ApiError, OrjsonResponse, error_body
from app.inference import Engine
from app.jobs import JobService
from app.middleware import ContractMiddleware
from app.routes import register_routes

logger = logging.getLogger("tensorforge")


class Runtime:
    def __init__(
        self, settings: Settings, engine: Any | None = None, autoload: bool = True
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.autoload = autoload and engine is None
        self.load_error: str | None = None
        self.executor = ThreadPoolExecutor(
            max_workers=settings.inference_threads,
            thread_name_prefix="infer",
        )

    def load_model(self) -> None:
        try:
            engine = Engine.load()
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("model failed to load; staying unhealthy")
            return
        self.engine = engine
        logger.info("model loaded version=%s config=%s", engine.model_version, engine.config_name)


def _configure_logging(level: str) -> None:
    log = logging.getLogger("tensorforge")
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.propagate = False


def _load_dotenv_defaults(path: Path | None = None) -> None:
    """Fill empty environment keys from `.env`. Never overrides, never logs values."""
    env_path = path or Path(".env")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def create_app(
    settings: Settings | None = None,
    engine: Any | None = None,
    autoload: bool | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    _configure_logging(settings.log_level)
    if autoload is None:
        autoload = engine is None
    runtime = Runtime(settings, engine=engine, autoload=autoload)
    runtime.jobs = JobService(
        settings.job_db_path,
        retention_seconds=settings.job_retention_seconds,
        tombstone_seconds=settings.job_tombstone_seconds,
        engine_getter=lambda: runtime.engine,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if runtime.autoload:
            threading.Thread(target=runtime.load_model, name="model-loader", daemon=True).start()
        runtime.jobs.start()
        try:
            yield
        finally:
            runtime.jobs.stop()
            runtime.executor.shutdown(wait=False, cancel_futures=True)

    api = FastAPI(
        title="TensorForge",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        redirect_slashes=False,
        lifespan=lifespan,
        default_response_class=OrjsonResponse,
    )
    api.state.runtime = runtime
    api.state.settings = settings

    @api.exception_handler(ApiError)
    async def _api_error(_request: Any, exc: ApiError) -> OrjsonResponse:
        return OrjsonResponse(
            error_body(exc.code, exc.message, exc.details),
            status_code=exc.status,
            headers=exc.headers,
        )

    @api.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Any, exc: StarletteHTTPException) -> OrjsonResponse:
        if exc.status_code == 405:
            allow = None
            if exc.headers:
                allow = exc.headers.get("allow")
            headers = {"Allow": allow} if allow else None
            return OrjsonResponse(
                error_body("method_not_allowed", "Method not allowed."),
                status_code=405,
                headers=headers,
            )
        if exc.status_code == 404:
            return OrjsonResponse(error_body("not_found", "Not found."), status_code=404)
        return OrjsonResponse(
            error_body("internal_error", "Internal server error."), status_code=500
        )

    @api.exception_handler(Exception)
    async def _unhandled(_request: Any, _exc: Exception) -> OrjsonResponse:
        logger.exception("unhandled request error")
        return OrjsonResponse(
            error_body("internal_error", "Internal server error."), status_code=500
        )

    register_routes(api, runtime)
    ui_dir = Path(__file__).resolve().parents[1] / "ui"
    if ui_dir.is_dir():
        api.mount("/demo", StaticFiles(directory=ui_dir, html=True), name="demo")
    api.add_middleware(ContractMiddleware, settings=settings)
    return api


app = create_app()


def main() -> None:
    import uvicorn

    _load_dotenv_defaults()
    settings = load_settings()
    uvicorn.run(
        create_app(settings),
        host="0.0.0.0",  # noqa: S104  # container must listen on all interfaces
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
