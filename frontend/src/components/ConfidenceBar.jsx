import React from "react";

/**
 * Visual confidence bar.
 * Accepts confidence in [0,1] OR [0,100]. Gracefully handles unknowns.
 */
export default function ConfidenceBar({ confidence }) {
  const num = typeof confidence === "number" ? confidence : parseFloat(confidence);
  const pct = Number.isFinite(num)
    ? Math.max(0, Math.min(100, num <= 1 ? num * 100 : num))
    : null;

  return (
    <div data-testid="confidence-bar" className="mt-3">
      <div className="flex items-center justify-between text-[11px] text-muted mb-1">
        <span className="uppercase tracking-wider">Confidence</span>
        <span className="font-mono" data-testid="confidence-value">
          {pct === null ? "—" : `${Math.round(pct)}%`}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
        <div
          className="h-full rounded-full bg-grad-primary transition-[width] duration-700 ease-out"
          style={{ width: `${pct ?? 0}%` }}
        />
      </div>
    </div>
  );
}
