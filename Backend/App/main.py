from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


class AppPaths:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.static_dir = base_dir / "Static"
        self.templates_dir = base_dir / "Templates"


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

    def _register_routes(self) -> None:
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})

        @self.app.get("/chat", response_class=HTMLResponse)
        async def chat(request: Request):
            return self.templates.TemplateResponse("chat.html", {"request": request})


app = WebApp(AppPaths(Path(__file__).resolve().parent)).app
