import React, { useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Send, Sparkles, AlertTriangle, ArrowLeft, Menu, X } from "lucide-react";
import Sidebar from "../components/Sidebar";
import ChatMessage from "../components/ChatMessage";
import TypingIndicator from "../components/TypingIndicator";
import { sendChatMessage, API_BASE_URL } from "../lib/api";

const SUGGESTIONS = [
  "Summarize the main findings of the uploaded document.",
  "Compare the methodology in section 2 and section 4.",
  "What are the key limitations mentioned in the conclusion?",
  "Extract the experimental results with their metrics.",
];

export default function Chat() {
  const location = useLocation();
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);      // [{id, role, content, citations, sources, confidence, error}]
  const [isSending, setIsSending] = useState(false);
  const [globalError, setGlobalError] = useState(null);
  const [mobileSidebar, setMobileSidebar] = useState(false);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isSending]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const hasMessages = messages.length > 0;

  const send = async (text) => {
    const msg = (text ?? input).trim();
    if (!msg || isSending) return;

    setGlobalError(null);
    const userMsg = { id: `u-${Date.now()}`, role: "user", content: msg };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setIsSending(true);

    try {
      const data = await sendChatMessage(msg);
      const aiMsg = {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: data?.answer ?? "(no answer returned)",
        citations: data?.citations ?? [],
        sources: data?.sources ?? [],
        confidence: typeof data?.confidence === "number" ? data.confidence : undefined,
      };
      setMessages((m) => [...m, aiMsg]);
    } catch (err) {
      const detail =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        err?.message ||
        "Unable to reach the backend.";
      const status = err?.response?.status ? ` (HTTP ${err.response.status})` : "";
      const friendly =
        err?.code === "ERR_NETWORK"
          ? `Couldn't reach ${API_BASE_URL}. Make sure the backend is running.`
          : `${detail}${status}`;
      setGlobalError(friendly);
      setMessages((m) => [
        ...m,
        {
          id: `e-${Date.now()}`,
          role: "assistant",
          content: friendly,
          error: true,
        },
      ]);
    } finally {
      setIsSending(false);
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    send();
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const newChat = () => {
    setMessages([]);
    setGlobalError(null);
    setInput("");
    inputRef.current?.focus();
  };

  const history = useMemo(() => {
    // Build a light history from user messages
    const firsts = messages.filter((m) => m.role === "user");
    if (!firsts.length) return [];
    return [{ id: "current", title: firsts[0].content.slice(0, 60) }];
  }, [messages]);

  return (
    <div data-testid="chat-page" className="h-screen w-full flex flex-col bg-bg">
      {/* Mobile top bar */}
      <div className="lg:hidden flex items-center justify-between px-4 h-14 border-b border-white/5 bg-bg/80 backdrop-blur">
        <button
          className="btn-ghost w-9 h-9 rounded-lg flex items-center justify-center"
          onClick={() => setMobileSidebar(true)}
          data-testid="mobile-open-sidebar-btn"
        >
          <Menu className="w-4 h-4" />
        </button>
        <div className="font-display font-semibold text-sm tracking-tight">
          Agentic<span className="text-accent">.</span>RAG
        </div>
        <Link to="/" className="btn-ghost w-9 h-9 rounded-lg flex items-center justify-center">
          <ArrowLeft className="w-4 h-4" />
        </Link>
      </div>

      <div className="flex flex-1 min-h-0">
        {/* Desktop sidebar */}
        <Sidebar
          history={history}
          onNewChat={newChat}
          onSelect={() => {}}
          activeId="current"
        />

        {/* Mobile sidebar drawer */}
        {mobileSidebar && (
          <div className="fixed inset-0 z-50 lg:hidden" data-testid="mobile-sidebar-drawer">
            <div
              className="absolute inset-0 bg-black/60"
              onClick={() => setMobileSidebar(false)}
            />
            <div className="absolute left-0 top-0 bottom-0 w-80 bg-bg border-r border-white/10 shadow-2xl animate-slide-up">
              <div className="flex items-center justify-between p-3 border-b border-white/5">
                <div className="font-display font-semibold text-sm">Menu</div>
                <button
                  onClick={() => setMobileSidebar(false)}
                  className="btn-ghost w-8 h-8 rounded-md flex items-center justify-center"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="lg:hidden block">
                <Sidebar
                  history={history}
                  onNewChat={() => {
                    newChat();
                    setMobileSidebar(false);
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Main chat area */}
        <main className="flex-1 flex flex-col min-w-0 min-h-0 relative">
          {/* Top meta */}
          <div className="hidden lg:flex items-center justify-between px-6 h-14 border-b border-white/5 bg-bg/60 backdrop-blur">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-accent/15 border border-accent/25 flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-accent" />
              </div>
              <div>
                <div className="text-sm font-medium">Document Q&amp;A</div>
                <div className="text-[11px] text-muted">
                  Connected to <span className="font-mono">{API_BASE_URL}</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={newChat}
                data-testid="chat-clear-btn"
                className="btn-ghost rounded-full px-3 py-1.5 text-xs"
              >
                Clear
              </button>
              <Link
                to="/"
                data-testid="chat-back-home-btn"
                className="btn-ghost rounded-full px-3 py-1.5 text-xs"
              >
                ← Home
              </Link>
            </div>
          </div>

          {/* Global error banner */}
          {globalError && (
            <div
              data-testid="chat-error-banner"
              className="mx-6 mt-4 rounded-xl border border-red-500/30 bg-red-500/10 text-red-200 px-4 py-2.5 text-sm flex items-start gap-2"
            >
              <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
              <div className="flex-1 break-words">{globalError}</div>
              <button
                className="text-red-200/70 hover:text-red-100 text-xs"
                onClick={() => setGlobalError(null)}
                data-testid="chat-error-dismiss-btn"
              >
                dismiss
              </button>
            </div>
          )}

          {/* Messages */}
          <div
            ref={scrollRef}
            data-testid="chat-messages"
            className="flex-1 overflow-y-auto px-4 sm:px-6 lg:px-10 py-6"
          >
            {!hasMessages ? (
              <EmptyState onPick={(q) => send(q)} />
            ) : (
              <div className="max-w-3xl mx-auto flex flex-col gap-4">
                {messages.map((m) => (
                  <ChatMessage
                    key={m.id}
                    role={m.role}
                    content={m.content}
                    citations={m.citations}
                    sources={m.sources}
                    confidence={m.confidence}
                    error={m.error}
                  />
                ))}
                {isSending && <TypingIndicator />}
                <div className="h-4" />
              </div>
            )}
          </div>

          {/* Input */}
          <form
            onSubmit={handleSubmit}
            data-testid="chat-input-form"
            className="px-4 sm:px-6 lg:px-10 pb-6"
          >
            <div className="max-w-3xl mx-auto">
              <div className="glass rounded-2xl p-2 flex items-end gap-2 focus-within:border-primary/40 transition-colors">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKey}
                  rows={1}
                  placeholder="Ask anything about your documents..."
                  data-testid="chat-input"
                  className="flex-1 resize-none bg-transparent outline-none text-[15px] leading-6 px-3 py-2.5 placeholder:text-muted max-h-40"
                />
                <button
                  type="submit"
                  disabled={!input.trim() || isSending}
                  data-testid="chat-send-btn"
                  className="btn-gradient rounded-xl h-10 px-4 flex items-center justify-center gap-1.5 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none"
                >
                  <Send className="w-4 h-4" />
                  <span className="hidden sm:inline">Send</span>
                </button>
              </div>
              <div className="mt-2 flex items-center justify-between text-[11px] text-muted">
                <span>Press Enter to send · Shift+Enter for newline</span>
                <span className="font-mono">{input.length} chars</span>
              </div>
            </div>
          </form>
        </main>
      </div>
    </div>
  );
}

function EmptyState({ onPick }) {
  return (
    <div
      data-testid="chat-empty-state"
      className="max-w-2xl mx-auto mt-8 sm:mt-16 text-center"
    >
      <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-white/10 bg-white/[0.03] text-[11px] uppercase tracking-[0.22em] text-muted mb-6">
        <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
        Ready to answer
      </div>
      <h2 className="font-display text-3xl sm:text-4xl font-semibold tracking-tight">
        What should we <span className="text-accent">explore</span> today?
      </h2>
      <p className="text-muted mt-3 max-w-lg mx-auto">
        Drop a document in the sidebar or pick a starter prompt. Every answer comes with sources and a confidence signal.
      </p>

      <div className="mt-8 grid sm:grid-cols-2 gap-3 text-left">
        {SUGGESTIONS.map((s, i) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            data-testid={`suggestion-${i}`}
            className="glass rounded-xl p-4 text-sm card-hover"
          >
            <div className="text-[11px] uppercase tracking-wider text-accent mb-1">
              Starter · 0{i + 1}
            </div>
            <div className="text-text/90">{s}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
