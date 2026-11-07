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
 * @typedef {Object} ChatData
 * @property {string} answer
 * @property {string[]} citations
 * @property {number} confidence
 * @property {Object[]} sources
 *
 * @typedef {Object} ChatApiResponse
 * @property {true} success
 * @property {ChatData} data
 * @property {null} error
 */

/**
 * POST /api/chat
 * body: { message: string }
 * returns: { success: true, data: { answer, citations, confidence, sources }, error: null }
 *
 * @param {string} message
 * @returns {Promise<ChatApiResponse>}
 */
export async function sendChatMessage(message) {
  const res = await client.post("/api/chat", { message });
  return res.data;
}

/**
 * POST /api/chat/stream using Fetch so the browser can consume the SSE body.
 * Event callback receives { event, data }; the function resolves with final data.
 */
export async function streamChatMessage(message, onEvent, { signal } = {}) {
  const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ message }),
    signal,
  });

  if (!response.ok) {
    let detail = `Chat request failed (HTTP ${response.status}).`;
    try {
      const payload = await response.json();
      detail = payload?.error || payload?.detail || detail;
    } catch {
      // Keep the HTTP status message when the response is not JSON.
    }
    throw new Error(detail);
  }
  if (!response.body) {
    throw new Error("The browser could not read the answer stream.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalData = null;
  let completed = false;

  const dispatchFrame = (frame) => {
    if (!frame.trim()) return;
    let event = "message";
    const dataLines = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return;

    let data;
    try {
      data = JSON.parse(dataLines.join("\n"));
    } catch {
      throw new Error("The server sent an invalid streaming event.");
    }
    if (event === "error") {
      throw new Error(data?.message || "Answer generation failed.");
    }
    if (event === "final") finalData = data;
    if (event === "done") completed = true;
    onEvent?.({ event, data });
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      let separator;
      while ((separator = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, separator);
        buffer = buffer.slice(separator + 2);
        dispatchFrame(frame);
      }
    }
    buffer += decoder.decode();
    buffer = buffer.replace(/\r\n/g, "\n");
    if (buffer.trim()) dispatchFrame(buffer);
  } finally {
    if (!completed) await reader.cancel().catch(() => {});
    reader.releaseLock();
  }

  if (!completed || !finalData) {
    throw new Error("The answer stream ended before completion.");
  }
  return finalData;
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
