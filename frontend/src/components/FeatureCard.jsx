import React from "react";

export default function FeatureCard({ icon: Icon, title, description, accent, delay = 0, testid }) {
  return (
    <div
      data-testid={testid}
      className="card-hover relative glass rounded-2xl p-6 overflow-hidden animate-slide-up"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="grain" />
      <div
        className="w-11 h-11 rounded-xl flex items-center justify-center mb-4 border border-white/5"
        style={{ background: accent || "rgba(99,102,241,0.12)" }}
      >
        <Icon className="w-5 h-5 text-accent" />
      </div>
      <h3 className="font-display text-lg font-semibold tracking-tight mb-1.5">{title}</h3>
      <p className="text-sm text-muted leading-relaxed">{description}</p>
    </div>
  );
}
