from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from repo root .env
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import chat, evaluation, health, upload


class AppPaths:
    """Resolves key paths inside the repo."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir                          # backend/app
        self.repo_root = base_dir.parents[1]              # repo root
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
        self.app = FastAPI(title="Enterprise RAG QA System")

        self._register_api_routes()
        self._register_frontend()

    def _register_api_routes(self) -> None:
        self.app.include_router(health.router, prefix="/api")
        self.app.include_router(chat.router, prefix="/api")
        self.app.include_router(upload.router, prefix="/api")
        self.app.include_router(evaluation.router, prefix="/api")

    def _register_frontend(self) -> None:
        """Mount the Vite production build if it exists.

        Run `npm run build` in `frontend/` to generate `frontend/dist/`.
        """
        dist = self.paths.frontend_dist_dir
        if dist.exists() and (dist / "index.html").exists():
            self.app.mount(
                "/",
                StaticFiles(directory=str(dist), html=True),
                name="frontend",
            )


app = WebApp(AppPaths(Path(__file__).resolve().parent)).app
