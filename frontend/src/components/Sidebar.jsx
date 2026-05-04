import React, { useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Upload, FileText, Plus, Sparkles, Trash2, CheckCircle2, Loader2, AlertCircle } from "lucide-react";
import { uploadDocument, API_BASE_URL } from "../lib/api";

export default function Sidebar({ history = [], onNewChat, onSelect, activeId, onDocumentUploaded }) {
  const fileRef = useRef(null);
  const [docs, setDocs] = useState([]);            // [{name, size, status, progress, error}]
  const [isUploading, setIsUploading] = useState(false);

  const triggerFile = () => fileRef.current?.click();

  const handleFiles = async (files) => {
    const list = Array.from(files || []);
    if (!list.length) return;
    setIsUploading(true);

    for (const file of list) {
      const entry = { name: file.name, size: file.size, status: "uploading", progress: 0, error: null };
      setDocs((d) => [entry, ...d]);

      try {
        const res = await uploadDocument(file, (p) => {
          setDocs((d) =>
            d.map((x) => (x.name === file.name && x.status === "uploading" ? { ...x, progress: p } : x))
          );
        });
        setDocs((d) =>
          d.map((x) =>
            x.name === file.name ? { ...x, status: "ready", progress: 100 } : x
          )
        );
        onDocumentUploaded?.(res);
      } catch (err) {
        const msg =
          err?.response?.data?.detail ||
          err?.message ||
          "Upload failed — backend not reachable.";
        setDocs((d) =>
          d.map((x) => (x.name === file.name ? { ...x, status: "error", error: msg } : x))
        );
      }
    }
    setIsUploading(false);
    if (fileRef.current) fileRef.current.value = "";
  };

  const removeDoc = (name) =>
    setDocs((d) => d.filter((x) => x.name !== name));

  const prettySize = (bytes) => {
    if (!bytes && bytes !== 0) return "";
    const k = 1024;
    const units = ["B", "KB", "MB", "GB"];
    const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(k)));
    return `${(bytes / Math.pow(k, i)).toFixed(i ? 1 : 0)} ${units[i]}`;
  };

  return (
    <aside
      data-testid="chat-sidebar"
      className="hidden lg:flex w-72 shrink-0 flex-col h-full border-r border-white/5 bg-bg/60 backdrop-blur-xl"
    >
      <div className="p-4 border-b border-white/5">
        <Link to="/" className="flex items-center gap-2 mb-4" data-testid="sidebar-home-link">
          <div className="w-7 h-7 rounded-md bg-grad-primary flex items-center justify-center">
            <Sparkles className="w-3.5 h-3.5 text-white" />
          </div>
          <div className="font-display text-[14px] font-semibold tracking-tight">
            Agentic<span className="text-accent">.</span>RAG
          </div>
        </Link>

        <button
          type="button"
          onClick={onNewChat}
          data-testid="sidebar-new-chat-btn"
          className="w-full btn-gradient rounded-xl py-2.5 text-sm font-medium flex items-center justify-center gap-2"
        >
          <Plus className="w-4 h-4" /> New Chat
        </button>

        <button
          type="button"
          onClick={triggerFile}
          disabled={isUploading}
          data-testid="sidebar-upload-btn"
          className="w-full mt-2 btn-ghost rounded-xl py-2.5 text-sm font-medium flex items-center justify-center gap-2 disabled:opacity-60"
        >
          {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
          {isUploading ? "Uploading…" : "Upload Document"}
        </button>
        <input
          ref={fileRef}
          type="file"
          multiple
          hidden
          data-testid="sidebar-file-input"
          onChange={(e) => handleFiles(e.target.files)}
          accept=".pdf,.txt,.md,.doc,.docx,.html,.json,.csv"
        />
      </div>

      {/* Documents */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="text-[11px] uppercase tracking-[0.2em] text-muted">Documents</div>
          <div className="text-[10px] font-mono text-muted">{docs.length}</div>
        </div>

        {docs.length === 0 ? (
          <div
            data-testid="sidebar-empty-docs"
            onClick={triggerFile}
            className="cursor-pointer rounded-xl border border-dashed border-white/10 p-4 text-center hover:border-accent/40 transition"
          >
            <FileText className="w-5 h-5 text-muted mx-auto mb-1.5" />
            <div className="text-sm text-text/90">No documents yet</div>
            <div className="text-xs text-muted mt-0.5">
              Click to upload your first PDF or text file
            </div>
          </div>
        ) : (
          <ul className="space-y-1.5">
            {docs.map((doc) => (
              <li
                key={doc.name}
                data-testid={`sidebar-doc-item`}
                className="group rounded-lg px-2.5 py-2 hover:bg-white/[0.04] border border-transparent hover:border-white/5 transition"
              >
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-muted shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="text-[13px] truncate" title={doc.name}>
                      {doc.name}
                    </div>
                    <div className="text-[10px] text-muted flex items-center gap-2">
                      <span>{prettySize(doc.size)}</span>
                      {doc.status === "uploading" && (
                        <span className="text-accent flex items-center gap-1">
                          <Loader2 className="w-3 h-3 animate-spin" /> {doc.progress}%
                        </span>
                      )}
                      {doc.status === "ready" && (
                        <span className="text-emerald-400 flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3" /> indexed
                        </span>
                      )}
                      {doc.status === "error" && (
                        <span className="text-red-300 flex items-center gap-1" title={doc.error}>
                          <AlertCircle className="w-3 h-3" /> failed
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => removeDoc(doc.name)}
                    className="opacity-0 group-hover:opacity-100 transition text-muted hover:text-red-300"
                    title="Remove"
                    data-testid="sidebar-remove-doc-btn"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* History */}
      <div className="px-4 pb-4 flex-1 overflow-y-auto">
        <div className="text-[11px] uppercase tracking-[0.2em] text-muted mb-2">History</div>
        {history.length === 0 ? (
          <div className="text-xs text-muted/80">Your recent chats will appear here.</div>
        ) : (
          <ul className="space-y-1">
            {history.map((h) => (
              <li key={h.id}>
                <button
                  onClick={() => onSelect?.(h.id)}
                  data-testid="sidebar-history-item"
                  className={`w-full text-left truncate text-[13px] rounded-lg px-2.5 py-1.5 transition ${
                    activeId === h.id
                      ? "bg-primary/15 text-text border border-primary/25"
                      : "text-muted hover:text-text hover:bg-white/[0.04]"
                  }`}
                  title={h.title}
                >
                  {h.title || "Untitled"}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="p-4 border-t border-white/5">
        <div className="text-[11px] text-muted">
          API:&nbsp;
          <span className="font-mono text-text/80 break-all">
            {API_BASE_URL || "(same origin / proxy)"}
          </span>
        </div>
      </div>
    </aside>
  );
}
