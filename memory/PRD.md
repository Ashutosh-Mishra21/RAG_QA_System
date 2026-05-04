# Agentic RAG QA System — Frontend PRD

## 1. Original Problem Statement
Build a complete React + Tailwind CSS frontend (SPA with routing) for an AI-powered
document question-answering system that connects to a fixed, pre-existing FastAPI
backend at `http://localhost:8000` with:
- `POST /api/chat` → `{ answer, citations[], confidence, sources[] }`

Follow-up (2026-05-04): Migrate from CRA → **Vite**, remove Jinja2 from backend,
serve the Vite production build via FastAPI `StaticFiles` at `/`. Do NOT modify
API routes or business logic.

## 2. Architecture
- **Build tool**: Vite 5 (migrated from Create React App)
- **Frontend**: React 18 + React Router 6 + Tailwind CSS + axios + lucide-react
- **Structure**: Single Page App, two routes: `/` (Landing), `/chat` (Chat)
- **Dev server**: `vite --host 0.0.0.0 --port 3000` (run by supervisor via `yarn start`)
- **Dev proxy**: `/api/*` → `http://localhost:8000` (configurable via `VITE_API_PROXY_TARGET`)
- **Prod**: FastAPI mounts `frontend/dist/` at `/` with SPA fallback
- **Styling**: Dark theme (#0B0F19 / #121826), gradient primary (#6366F1 → #22D3EE), glass-morphism cards, Bricolage Grotesque display font, Geist body font

## 3. User Personas
- **Researcher**: Uploads papers, asks grounded questions, verifies citations
- **Enterprise analyst**: Needs confidence signals + source tracking for compliance
- **Developer**: Integrates the UI on top of a pre-existing RAG backend

## 4. Core Requirements (Static)
- Two pages: Landing + Chat (SPA)
- Landing: Hero with chat preview, Features grid (4), How-it-Works (6 steps), CTA, Footer
- Chat: Sidebar (upload, documents, history) + messages area + fixed bottom input
- User bubble (right, primary blue), AI bubble (left, dark #1F2937) with sources + confidence bar
- Typing indicator, fade/slide animations, error banner on API failure
- All interactive elements have unique `data-testid`
- No mocks, no fake data, real `/api/chat` calls only

## 5. What's Been Implemented
### 2026-05-04 (initial MVP on CRA)
- React + Tailwind SPA scaffold, dark theme with custom design tokens, Google Fonts
- Landing + Chat pages with all required sections and interactions
- axios client (`src/lib/api.js`) — POST `/api/chat` + `/api/upload`
- Graceful error handling, `data-testid` coverage
- Testing agent: 14/14 flows passed (100%)

### 2026-05-04 (Vite migration + FastAPI cleanup)
- ✅ Replaced CRA with **Vite 5** (@vitejs/plugin-react)
- ✅ `public/index.html` → root `index.html` with `<script type="module" src="/src/main.jsx">`
- ✅ `src/index.js` → `src/main.jsx`; `src/App.js` → `src/App.jsx`
- ✅ Env vars migrated `REACT_APP_*` → `VITE_*` (uses `import.meta.env`)
- ✅ `vite.config.js` with dev proxy (`/api` → `http://localhost:8000`), HMR wss, port 3000
- ✅ `postcss.config.js`, `tailwind.config.js` converted to ESM (`export default`)
- ✅ Backend `backend/app/main.py`: **removed Jinja2Templates, HTMLResponse, `_register_routes()`, old `/static` mount**; now only mounts `/api/*` routes and serves `frontend/dist/` at `/` with `html=True` (SPA fallback). API routes unchanged.
- ✅ Production build verified: `frontend/dist/index.html` + `frontend/dist/assets/*.{js,css}` generated
- ✅ Testing agent: 36/36 assertions, 13/13 flows passed (100%)

## 6. Running the App
### Development
```bash
# Terminal 1 — backend (from repo root)
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000

# Terminal 2 — frontend dev server (hot reload)
cd frontend
yarn install
yarn dev            # → http://localhost:3000 (proxies /api to :8000)
```

### Production
```bash
# Build the SPA
cd frontend && yarn build    # outputs frontend/dist/

# Run FastAPI — it serves /api AND the built frontend at /
uvicorn backend.app.main:app --port 8000
# → open http://localhost:8000
```

## 7. Prioritized Backlog
### P0 — None (MVP + migration complete)
### P1 (Next)
- Streaming responses (show tokens as they arrive) — requires backend SSE support
- Per-conversation persistence (localStorage) so chats survive reloads
- Keyboard shortcut cheat-sheet (⌘K) for power users
- Light-mode toggle
### P2 (Nice-to-have)
- Source expansion modal — click a source chip to view chunk text
- Multi-document filters in the sidebar
- Share-chat link + export conversation to Markdown/PDF
- Voice input via Web Speech API

## 8. Known Non-Issues (reviewed)
- HMR `wss://localhost/` warning in dev console — harmless artifact of the preview proxy
- React Router v6 future-flag warnings — cosmetic
- Typing indicator brief flash when backend is offline — network error resolves immediately

## 9. Files Created/Updated (latest)
- `/app/frontend/package.json` (Vite + scripts)
- `/app/frontend/vite.config.js`, `index.html` (root), `.env`
- `/app/frontend/postcss.config.js`, `tailwind.config.js` (ESM)
- `/app/frontend/src/main.jsx`, `App.jsx`, `lib/api.js`, `components/Sidebar.jsx`
- `/app/backend/app/main.py` (Jinja2 removed; serves Vite dist)
- `/app/frontend/dist/` (built)
