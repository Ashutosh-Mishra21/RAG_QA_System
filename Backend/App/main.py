from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .api.routes import chat, evaluation, health, upload


class AppPaths:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.repo_root = base_dir.parents[1]
        self.frontend_dir = self.repo_root / "frontend"
        self.static_dir = self.frontend_dir / "static"
        self.templates_dir = self.frontend_dir / "templates"


class WebApp:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.app = FastAPI(title="Enterprise RAG QA System")
        self.templates = Jinja2Templates(directory=str(self.paths.templates_dir))

        self.app.mount(
            "/static",
            StaticFiles(directory=str(self.paths.static_dir)),
            name="static",
        )
        self._register_routes()
        self._register_api_routes()

    def _register_routes(self) -> None:
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})

        @self.app.get("/chat", response_class=HTMLResponse)
        async def chat_page(request: Request):
            return self.templates.TemplateResponse("chat.html", {"request": request})

    def _register_api_routes(self) -> None:
        self.app.include_router(health.router, prefix="/api")
        self.app.include_router(chat.router, prefix="/api")
        self.app.include_router(upload.router, prefix="/api")
        self.app.include_router(evaluation.router, prefix="/api")


app = WebApp(AppPaths(Path(__file__).resolve().parent)).app
