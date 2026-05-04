# Agentic RAG QA System — Frontend PRD

## 1. Original Problem Statement
Build a complete React + Tailwind CSS frontend (SPA with routing) for an AI-powered
document question-answering system that connects to a fixed, pre-existing FastAPI
backend at `http://localhost:8000` with:
- `POST /api/chat` → `{ answer, citations[], confidence, sources[] }`

Strict rules: no backend changes, no mocks, no simulated data. On API failure the
UI must show a clear error message.

## 2. Architecture
- **Frontend**: React 18 + React Router 6 + Tailwind CSS + axios + lucide-react
- **Structure**: Single Page App, two routes: `/` (Landing), `/chat` (Chat)
- **Backend URL**: `REACT_APP_API_URL` (defaults to `http://localhost:8000`) — runs on user's own machine
- **Styling**: Dark theme (#0B0F19 / #121826), gradient primary (#6366F1 → #22D3EE), glass-morphism cards, Bricolage Grotesque display font, Geist body font
- **Services**: Supervisor-managed `yarn start` on port 3000

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

## 5. What's Been Implemented (2026-05-04)
- ✅ React + Tailwind SPA scaffold (`/app/frontend/` full rewrite)
- ✅ Dark theme with custom design tokens, Google Fonts (Bricolage Grotesque, Geist, JetBrains Mono)
- ✅ Landing page: Navbar, Hero (title/subtitle/Try-Demo/Upload buttons), Hero chat preview mock, Features (Retrieval, Reasoning, Fast, Documents), How-it-Works horizontal flow (Upload→Index→Query→Retrieve→Reason→Answer), CTA, Footer
- ✅ Chat page: Sidebar (New Chat, Upload wired to `/api/upload`, empty-state docs list, history), message area with user/AI bubbles, sources chips, confidence bar, typing indicator, error banner, starter suggestions, fixed bottom input with Enter-to-send / Shift+Enter newline
- ✅ axios client at `src/lib/api.js` — POST `/api/chat` + `/api/upload` with progress
- ✅ Error handling: network failure → friendly banner + red AI bubble mentioning the API URL
- ✅ `data-testid` on all interactive/user-facing elements
- ✅ Testing agent: 14/14 tests passed (100% frontend success rate)

## 6. Prioritized Backlog
### P0 — None (MVP complete)
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

## 7. Known Non-Issues (reviewed)
- `ws://localhost:443/ws` warning in dev console — CRA HMR artifact, harmless
- React Router v6 future-flag warnings — cosmetic
- Typing indicator brief flash when backend is offline — network error resolves immediately

## 8. Files Created/Updated
- `/app/frontend/package.json`, `.env`, `tailwind.config.js`, `postcss.config.js`
- `/app/frontend/public/index.html`
- `/app/frontend/src/index.js`, `index.css`, `App.js`, `lib/api.js`
- `/app/frontend/src/pages/Landing.jsx`, `pages/Chat.jsx`
- `/app/frontend/src/components/{Navbar,Footer,HeroPreview,FeatureCard,HowItWorks,Sidebar,ChatMessage,TypingIndicator,ConfidenceBar}.jsx`

## 9. How to Use Locally
1. Run the RAG backend on `http://localhost:8000` (FastAPI — `uvicorn backend.app.main:app --reload`)
2. Open the preview URL → Click **Try Demo** → Ask a question
3. If the backend is offline the UI will display a clear error message — this is expected behavior
