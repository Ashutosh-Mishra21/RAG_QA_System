import axios from "axios";

// Base URL for the RAG backend. Defaults to http://localhost:8000 as specified.
export const API_BASE_URL =
  process.env.REACT_APP_API_URL ||
  process.env.REACT_APP_BACKEND_URL ||
  "http://localhost:8000";

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
 * Optional — used by the sidebar upload button if the backend supports it.
 */
export async function uploadDocument(file, onProgress) {
  const form = new FormData();
  form.append("file", file);
  const res = await axios.post(`${API_BASE_URL}/api/upload`, form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded * 100) / e.total));
    },
    timeout: 300000,
  });
  return res.data;
}
