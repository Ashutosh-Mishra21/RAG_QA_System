import React from "react";
import { Bot } from "lucide-react";

export default function TypingIndicator() {
  return (
    <div
      data-testid="typing-indicator"
      className="flex justify-start animate-fade-in"
    >
      <div className="max-w-[88%] rounded-2xl rounded-bl-md bg-bubble-ai/90 px-4 py-3 border border-white/5">
        <div className="flex items-center gap-2 text-muted text-[11px] uppercase tracking-wider mb-1.5">
          <Bot className="w-3.5 h-3.5 text-accent" />
          <span>Agentic RAG is thinking</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
      </div>
    </div>
  );
}
