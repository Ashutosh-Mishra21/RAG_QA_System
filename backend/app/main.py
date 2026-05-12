from pathlib import Path
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import logging
import time

# Load environment variables from repo root .env
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.api.routes import chat, evaluation, health, upload
from backend.app.api.responses import error_response
from backend.app.core.logging import configure_logging
from backend.app.core.config import settings

configure_logging()

logger = logging.getLogger(__name__)
# =========================================
# =========================================


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and not Path(path).suffix:
                return await super().get_response("index.html", scope)
            raise


class AppPaths:
    """Resolves key paths inside the repo."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir  # backend/app
        self.repo_root = base_dir.parents[1]  # repo root
        self.frontend_dir = self.repo_root / "frontend"
        self.frontend_dist_dir = self.frontend_dir / "dist"


class WebApp:
    """FastAPI application.

    - Serves the Vite-built React SPA at `/` from `frontend/dist`
      (SPA fallback enabled via `html=True`).
    - Exposes REST APIs under `/api`.
    """

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.app = FastAPI(
            title="Enterprise RAG QA System",
            lifespan=self.lifespan,
            docs_url="/docs",
            redoc_url="/redoc",
        )

        self._register_api_routes()
        self._register_frontend()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self._register_middlewares()
        lifespan = self.lifespan
        self._register_exception_handlers()

    # =========================================
    # =========================================

    def _register_middlewares(self):

        @self.app.middleware("http")
        async def log_requests(request, call_next):

            start = time.time()

            response = await call_next(request)

            duration = round(time.time() - start, 3)

            logger.info(
                "%s %s completed in %ss",
                request.method,
                request.url.path,
                duration,
            )

            return response

    # =========================================
    # =========================================

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):

        logger.info("Starting Enterprise RAG QA System")

        if settings.OPENROUTER_API_KEY:
            logger.info("OpenRouter configured")
        else:
            logger.warning("OPENROUTER_API_KEY missing")

        logger.info("Startup validation complete")

        yield

        logger.info("Shutting down Enterprise RAG QA System")

    # def _register_events(self):

    #     @self.app.on_event("startup")
    #     async def startup_event():

    #         logger.info("Starting Enterprise RAG QA System")

    #         if settings.OPENROUTER_API_KEY:
    #             logger.info("OpenRouter configured")
    #         else:
    #             logger.warning("OPENROUTER_API_KEY missing")

    #         logger.info("Startup validation complete")

    # =========================================
    # =========================================

    def _register_exception_handlers(self):

        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request, exc):
            logger.warning(
                "HTTP error on %s %s: %s",
                request.method,
                request.url.path,
                exc.detail,
            )
            return JSONResponse(
                status_code=exc.status_code,
                content=error_response(str(exc.detail)),
            )

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request, exc):
            logger.warning(
                "Validation error on %s %s: %s",
                request.method,
                request.url.path,
                exc,
            )
            return JSONResponse(
                status_code=422,
                content=error_response("Invalid request payload"),
            )

        @self.app.exception_handler(Exception)
        async def global_exception_handler(request, exc):

            logger.exception("Unhandled exception: %s", exc)

            return JSONResponse(
                status_code=500,
                content=error_response("Internal server error"),
            )

    # =========================================
    # =========================================

    def _register_api_routes(self) -> None:
        self.app.include_router(health.router, prefix="/api")
        self.app.include_router(chat.router, prefix="/api")
        self.app.include_router(upload.router, prefix="/api")
        self.app.include_router(evaluation.router, prefix="/api")

    # =========================================
    # =========================================

    def _register_frontend(self) -> None:
        """Mount the Vite production build if it exists.

        Run `npm run build` in `frontend/` to generate `frontend/dist/`.
        """
        dist = self.paths.frontend_dist_dir
        if dist.exists() and (dist / "index.html").exists():
            self.app.mount(
                "/",
                SPAStaticFiles(directory=str(dist), html=True),
                name="frontend",
            )
        else:
            logger.warning("Frontend build not found at %s", dist)


app = WebApp(AppPaths(Path(__file__).resolve().parent)).app
