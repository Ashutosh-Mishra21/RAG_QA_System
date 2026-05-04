import React from "react";
import { Upload, Database, Search, Network, Brain, Sparkles } from "lucide-react";

const steps = [
  { key: "upload", label: "Upload", icon: Upload, hint: "PDF, Docs, Text" },
  { key: "index", label: "Index", icon: Database, hint: "Chunk + Embed" },
  { key: "query", label: "Query", icon: Search, hint: "Natural language" },
  { key: "retrieve", label: "Retrieve", icon: Network, hint: "Hybrid search" },
  { key: "reason", label: "Reason", icon: Brain, hint: "Agentic loop" },
  { key: "answer", label: "Answer", icon: Sparkles, hint: "Grounded + cited" },
];

export default function HowItWorks() {
  return (
    <section
      id="how"
      data-testid="how-it-works-section"
      className="max-w-7xl mx-auto px-6 lg:px-10 py-20"
    >
      <div className="mb-12 flex items-end justify-between flex-wrap gap-4">
        <div className="max-w-xl">
          <div className="text-[11px] uppercase tracking-[0.25em] text-accent mb-3">Pipeline</div>
          <h2 className="font-display text-3xl md:text-4xl font-semibold tracking-tight">
            From raw document to <span className="text-accent">grounded answer</span>
          </h2>
        </div>
        <p className="text-sm text-muted max-w-md">
          Each step is observable, swappable, and optimized for retrieval quality — not just LLM output.
        </p>
      </div>

      <div className="relative">
        {/* connecting line */}
        <div className="hidden md:block absolute left-0 right-0 top-[44px] h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

        <ol className="grid grid-cols-2 md:grid-cols-6 gap-4 md:gap-3 relative">
          {steps.map(({ key, label, icon: Icon, hint }, idx) => (
            <li
              key={key}
              data-testid={`flow-step-${key}`}
              className="animate-slide-up"
              style={{ animationDelay: `${idx * 80}ms` }}
            >
              <div className="flex flex-col items-start">
                <div className="relative z-10 w-[88px] h-[88px] rounded-2xl glass flex flex-col items-center justify-center gap-1 card-hover">
                  <Icon className="w-5 h-5 text-accent" />
                  <span className="text-[10px] uppercase tracking-wider text-muted">
                    0{idx + 1}
                  </span>
                </div>
                <div className="mt-3">
                  <div className="font-display text-[15px] font-semibold tracking-tight">{label}</div>
                  <div className="text-xs text-muted">{hint}</div>
                </div>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
