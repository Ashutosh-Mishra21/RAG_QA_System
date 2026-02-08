from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(title="Enterprise RAG QA System")

# Mount Static directory (CASE-SENSITIVE)
app.mount("/static", StaticFiles(directory="Backend/App/Static"), name="static")

# Templates directory (CASE-SENSITIVE)
templates = Jinja2Templates(directory="Backend/App/Templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})
