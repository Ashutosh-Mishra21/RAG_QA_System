import React from "react";
import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const { pathname } = useLocation();
  const onChat = pathname === "/chat";
  return (
    <header
      data-testid="site-navbar"
      className="sticky top-0 z-40 backdrop-blur-xl bg-bg/60 border-b border-white/5"
    >
      <div className="max-w-7xl mx-auto px-6 lg:px-10 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2.5 group" data-testid="brand-link">
          <div className="w-8 h-8 rounded-lg bg-grad-primary flex items-center justify-center shadow-glow">
            <svg viewBox="0 0 24 24" className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 6h16M4 12h10M4 18h16" />
            </svg>
          </div>
          <div className="leading-tight">
            <div className="font-display text-[15px] font-semibold tracking-tight">
              Agentic<span className="text-accent">.</span>RAG
            </div>
            <div className="text-[10px] uppercase tracking-[0.18em] text-muted">QA System</div>
          </div>
        </Link>

        <nav className="hidden md:flex items-center gap-8 text-sm text-muted">
          <a href="/#features" className="hover:text-text transition-colors">Features</a>
          <a href="/#how" className="hover:text-text transition-colors">How it works</a>
          <a
            href="https://github.com/"
            target="_blank"
            rel="noreferrer"
            className="hover:text-text transition-colors"
          >
            GitHub
          </a>
        </nav>

        <div className="flex items-center gap-2">
          {!onChat ? (
            <Link
              to="/chat"
              data-testid="nav-open-chat-btn"
              className="btn-gradient px-4 py-2 rounded-full text-sm font-medium"
            >
              Open Chat
            </Link>
          ) : (
            <Link
              to="/"
              data-testid="nav-back-home-btn"
              className="btn-ghost px-4 py-2 rounded-full text-sm"
            >
              ← Home
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}
