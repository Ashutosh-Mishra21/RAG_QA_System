import React from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Upload, Search, Network, Brain, Zap, FileText } from "lucide-react";
import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import HeroPreview from "../components/HeroPreview";
import FeatureCard from "../components/FeatureCard";
import HowItWorks from "../components/HowItWorks";

const features = [
  {
    icon: Search,
    title: "Intelligent Retrieval",
    description:
      "Hybrid BM25 + dense vector search with cross-encoder reranking surfaces the most relevant passages — not just the closest match.",
    accent: "rgba(99,102,241,0.14)",
    testid: "feature-card-retrieval",
  },
  {
    icon: Brain,
    title: "Agentic Reasoning",
    description:
      "Multi-hop decomposition, query rewriting, and iterative follow-ups let the system think through complex questions step by step.",
    accent: "rgba(34,211,238,0.12)",
    testid: "feature-card-reasoning",
  },
  {
    icon: Zap,
    title: "Fast & Efficient",
    description:
      "Response and LLM caches make repeat questions near-instant. Streaming-ready for low-latency UX.",
    accent: "rgba(99,102,241,0.14)",
    testid: "feature-card-fast",
  },
  {
    icon: FileText,
    title: "Document Understanding",
    description:
      "Layout-aware parsing preserves structure — sections, tables, and headings become first-class retrieval signals.",
    accent: "rgba(34,211,238,0.12)",
    testid: "feature-card-documents",
  },
];

export default function Landing() {
  return (
    <div data-testid="landing-page" className="min-h-screen flex flex-col">
      <Navbar />

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 grid-dots opacity-40 pointer-events-none" />
        <div className="max-w-7xl mx-auto px-6 lg:px-10 pt-16 pb-24 grid lg:grid-cols-[1.05fr_1fr] gap-12 items-center">
          {/* Left */}
          <div className="animate-slide-up">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-white/10 bg-white/[0.03] text-[11px] uppercase tracking-[0.22em] text-muted mb-6">
              <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
              Retrieval-Augmented · Agentic · Grounded
            </div>

            <h1 className="font-display text-4xl sm:text-5xl lg:text-[64px] leading-[1.02] tracking-tight font-semibold">
              Agentic RAG
              <br />
              <span className="bg-clip-text text-transparent bg-grad-primary">QA System</span>
            </h1>

            <p className="mt-6 text-[17px] leading-relaxed text-muted max-w-xl">
              Ask questions. Get grounded, multi-step reasoning answers from your documents — with
              citations, confidence signals, and sources you can verify.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                to="/chat"
                data-testid="hero-try-demo-btn"
                className="btn-gradient inline-flex items-center gap-2 rounded-full px-5 py-3 text-sm font-medium"
              >
                Try Demo <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                to="/chat?upload=1"
                data-testid="hero-upload-doc-btn"
                className="btn-ghost inline-flex items-center gap-2 rounded-full px-5 py-3 text-sm"
              >
                <Upload className="w-4 h-4" /> Upload Document
              </Link>
            </div>

            <dl className="mt-12 grid grid-cols-3 gap-6 max-w-lg">
              {[
                { k: "Retrieval", v: "Hybrid", hint: "BM25 + Dense" },
                { k: "Reasoning", v: "Agentic", hint: "Multi-hop" },
                { k: "Output", v: "Grounded", hint: "With sources" },
              ].map((s) => (
                <div key={s.k}>
                  <dt className="text-[10px] uppercase tracking-[0.2em] text-muted">{s.k}</dt>
                  <dd className="font-display text-2xl font-semibold tracking-tight mt-1">
                    {s.v}
                  </dd>
                  <div className="text-[11px] text-muted">{s.hint}</div>
                </div>
              ))}
            </dl>
          </div>

          {/* Right */}
          <div className="animate-slide-up" style={{ animationDelay: "160ms" }}>
            <HeroPreview />
          </div>
        </div>
      </section>

      {/* Features */}
      <section
        id="features"
        data-testid="features-section"
        className="max-w-7xl mx-auto px-6 lg:px-10 py-16"
      >
        <div className="mb-12 flex items-end justify-between flex-wrap gap-4">
          <div className="max-w-xl">
            <div className="text-[11px] uppercase tracking-[0.25em] text-accent mb-3">Capabilities</div>
            <h2 className="font-display text-3xl md:text-4xl font-semibold tracking-tight">
              Built for retrieval quality, <br />
              not just LLM fluency.
            </h2>
          </div>
          <p className="text-sm text-muted max-w-md">
            Every component of the pipeline is observable and optimized — from ingestion to generation.
          </p>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {features.map((f, i) => (
            <FeatureCard key={f.title} {...f} delay={i * 90} />
          ))}
        </div>
      </section>

      {/* How it works */}
      <HowItWorks />

      {/* CTA */}
      <section className="max-w-7xl mx-auto px-6 lg:px-10 py-16">
        <div className="relative glass rounded-3xl p-8 md:p-12 overflow-hidden">
          <div className="grain" />
          <div className="relative z-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
            <div>
              <h3 className="font-display text-2xl md:text-3xl font-semibold tracking-tight">
                Ready to chat with your documents?
              </h3>
              <p className="text-muted mt-2 max-w-xl">
                Open the chat, drop in a PDF, and start asking. The system cites every claim it makes.
              </p>
            </div>
            <Link
              to="/chat"
              data-testid="cta-open-chat-btn"
              className="btn-gradient inline-flex items-center gap-2 rounded-full px-6 py-3 text-sm font-medium"
            >
              Open Chat <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}
