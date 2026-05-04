import React from "react";

export default function Footer() {
  return (
    <footer data-testid="site-footer" className="mt-24 border-t border-white/5">
      <div className="max-w-7xl mx-auto px-6 lg:px-10 py-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-grad-primary flex items-center justify-center">
            <svg viewBox="0 0 24 24" className="w-4 h-4 text-white" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 6h16M4 12h10M4 18h16" />
            </svg>
          </div>
          <div>
            <div className="font-display font-semibold tracking-tight">Agentic RAG QA System</div>
            <div className="text-xs text-muted">Grounded answers. Cited sources. Agentic reasoning.</div>
          </div>
        </div>

        <div className="flex items-center gap-6 text-sm text-muted">
          <a href="/#features" className="hover:text-text transition-colors">Features</a>
          <a href="/#how" className="hover:text-text transition-colors">How it works</a>
          <a href="/chat" className="hover:text-text transition-colors">Chat</a>
        </div>

        <div className="text-xs text-muted">
          © {new Date().getFullYear()} RAG QA — Built with FastAPI + React.
        </div>
      </div>
    </footer>
  );
}
