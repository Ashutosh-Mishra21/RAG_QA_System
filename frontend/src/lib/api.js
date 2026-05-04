import axios from "axios";

/**
 * Base URL for the RAG backend.
 *
 * - When empty (default in production build), the frontend is served by
 *   FastAPI itself, so relative "/api" paths hit the same origin.
 * - In local dev, Vite proxies "/api" to VITE_API_PROXY_TARGET (see vite.config.js).
 * - Set VITE_API_URL to override with a full URL (e.g. http://localhost:8000).
 */
export const API_BASE_URL = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
  timeout: 120000,
});

/**
 * POST /api/chat
 * body: { message: string }
 * returns: { answer, citations, confidence, sources }
 */
export async function sendChatMessage(message) {
  const res = await client.post("/api/chat", { message });
  return res.data;
}

/**
 * POST /api/upload   (multipart/form-data)
 * Used by the sidebar upload button.
 */
export async function uploadDocument(file, onProgress) {
  const form = new FormData();
  form.append("file", file);
  const url = `${API_BASE_URL}/api/upload`;
  const res = await axios.post(url, form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded * 100) / e.total));
    },
    timeout: 300000,
  });
  return res.data;
}
