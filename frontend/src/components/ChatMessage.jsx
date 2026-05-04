import React from "react";
import ReactMarkdown from "react-markdown";
import { Bot, User, Quote, AlertTriangle } from "lucide-react";
import ConfidenceBar from "./ConfidenceBar";

function SourceChip({ label, idx }) {
  return (
    <span
      data-testid={`source-chip-${idx}`}
      className="text-[11px] px-2 py-1 rounded-full bg-white/5 border border-white/5 text-muted hover:text-text hover:border-white/15 transition"
      title={label}
    >
      {label}
    </span>
  );
}

function normalizeSourceLabel(s, idx) {
  if (s == null) return `Source ${idx + 1}`;
  if (typeof s === "string") return s;
  if (typeof s === "object") {
    return (
      s.section ||
      s.title ||
      s.document_id ||
      s.source ||
      s.name ||
      `Source ${idx + 1}`
    );
  }
  return String(s);
}

export default function ChatMessage({ role, content, citations, sources, confidence, error }) {
  const isUser = role === "user";

  if (isUser) {
    return (
      <div data-testid="chat-message-user" className="flex justify-end animate-slide-up">
        <div className="flex items-start gap-2 max-w-[82%]">
          <div className="rounded-2xl rounded-br-md bg-primary text-white px-4 py-2.5 text-sm shadow-glow whitespace-pre-wrap break-words">
            {content}
          </div>
          <div className="w-8 h-8 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center shrink-0">
            <User className="w-4 h-4 text-primary" />
          </div>
        </div>
      </div>
    );
  }

  // AI message
  const citationList = Array.isArray(citations) ? citations : [];
  const sourceList = Array.isArray(sources) ? sources : [];
  const chips = [
    ...citationList.map((c) => (typeof c === "string" ? c : normalizeSourceLabel(c, 0))),
    ...sourceList.map((s, i) => normalizeSourceLabel(s, i)),
  ].filter(Boolean);

  const hasSources = chips.length > 0;
  const hasConfidence = typeof confidence === "number";

  return (
    <div data-testid="chat-message-ai" className="flex justify-start animate-slide-up">
      <div className="flex items-start gap-2 max-w-[88%]">
        <div className="w-8 h-8 rounded-full bg-accent/15 border border-accent/30 flex items-center justify-center shrink-0">
          <Bot className="w-4 h-4 text-accent" />
        </div>
        <div
          className={`rounded-2xl rounded-bl-md border px-4 py-3 text-sm w-full ${
            error
              ? "bg-red-500/10 border-red-500/30 text-red-200"
              : "bg-bubble-ai/90 border-white/5 text-text"
          }`}
        >
          {error ? (
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-red-300 mt-0.5 shrink-0" />
              <div>
                <div className="font-medium text-red-200 mb-0.5">Request failed</div>
                <div className="text-xs text-red-200/80 break-words">{content}</div>
              </div>
            </div>
          ) : (
            <div className="prose-chat">
              <ReactMarkdown>{content || ""}</ReactMarkdown>
            </div>
          )}

          {!error && hasSources && (
            <div className="mt-3 pt-3 border-t border-white/5">
              <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-muted mb-1.5">
                <Quote className="w-3 h-3" /> Sources
              </div>
              <div className="flex flex-wrap gap-1.5" data-testid="sources-list">
                {chips.map((label, i) => (
                  <SourceChip key={`${label}-${i}`} label={label} idx={i} />
                ))}
              </div>
            </div>
          )}

          {!error && hasConfidence && <ConfidenceBar confidence={confidence} />}
        </div>
      </div>
    </div>
  );
}
