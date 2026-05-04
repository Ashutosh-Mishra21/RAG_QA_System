import React from "react";
import { FileText, Bot, Quote } from "lucide-react";

/**
 * Static, stylized mock of the chat UI shown on the hero.
 * No API calls — pure visual preview.
 */
export default function HeroPreview() {
  return (
    <div
      data-testid="hero-chat-preview"
      className="relative glass rounded-3xl p-4 sm:p-5 shadow-card overflow-hidden"
    >
      <div className="grain pointer-events-none" />
      {/* Top bar */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-red-400/70" />
          <span className="w-2.5 h-2.5 rounded-full bg-yellow-400/70" />
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-400/70" />
          <span className="ml-3 text-[11px] uppercase tracking-[0.2em] text-muted">live · demo</span>
        </div>
        <div className="text-[11px] text-muted font-mono">/api/chat</div>
      </div>

      {/* Messages */}
      <div className="space-y-3">
        {/* user */}
        <div className="flex justify-end animate-slide-up" style={{ animationDelay: "50ms" }}>
          <div className="max-w-[80%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-white text-sm shadow-glow">
            What's the main finding in section 3 of the paper?
          </div>
        </div>

        {/* ai */}
        <div className="flex justify-start animate-slide-up" style={{ animationDelay: "180ms" }}>
          <div className="max-w-[88%] rounded-2xl rounded-bl-md bg-bubble-ai/90 px-4 py-3 text-sm border border-white/5">
            <div className="flex items-center gap-2 mb-2 text-muted text-[11px] uppercase tracking-wider">
              <Bot className="w-3.5 h-3.5 text-accent" /> Agentic RAG
            </div>
            <p className="leading-relaxed">
              The authors report a <span className="text-accent font-medium">12.8% improvement</span> in
              retrieval accuracy using hybrid BM25 + dense vectors, with a <span className="text-accent font-medium">34% reduction</span> in hallucinations.
            </p>

            {/* sources */}
            <div className="mt-3 pt-3 border-t border-white/5">
              <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-muted mb-1.5">
                <Quote className="w-3 h-3" /> Sources
              </div>
              <div className="flex flex-wrap gap-1.5">
                {["§3.1 Methodology", "§3.4 Results", "Abstract"].map((s) => (
                  <span
                    key={s}
                    className="text-[11px] px-2 py-1 rounded-full bg-white/5 border border-white/5 text-muted hover:text-text transition"
                  >
                    {s}
                  </span>
                ))}
              </div>
            </div>

            {/* confidence */}
            <div className="mt-3">
              <div className="flex items-center justify-between text-[11px] text-muted mb-1">
                <span>Confidence</span>
                <span className="font-mono">82%</span>
              </div>
              <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                <div
                  className="h-full bg-grad-primary rounded-full"
                  style={{ width: "82%" }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* input mock */}
        <div className="mt-4 flex items-center gap-2 rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2.5">
          <FileText className="w-4 h-4 text-muted" />
          <span className="text-sm text-muted flex-1">Ask anything about your documents…</span>
          <kbd className="text-[10px] font-mono text-muted border border-white/10 rounded px-1.5 py-0.5">⏎</kbd>
        </div>
      </div>
    </div>
  );
}
