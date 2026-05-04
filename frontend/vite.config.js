import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiTarget = env.VITE_API_PROXY_TARGET || "http://localhost:8000";

  return {
    plugins: [react()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      host: "0.0.0.0",
      port: 3000,
      strictPort: true,
      // allow the emergent preview host in dev
      allowedHosts: true,
      // Proxy /api calls to FastAPI backend during dev
      proxy: {
        "/api": {
          target: apiTarget,
          changeOrigin: true,
          secure: false,
        },
      },
      // HMR over HTTPS proxy (emergent preview)
      hmr: {
        clientPort: 443,
        protocol: "wss",
      },
    },
    preview: {
      host: "0.0.0.0",
      port: 3000,
      strictPort: true,
    },
    build: {
      outDir: "dist",
      sourcemap: false,
      emptyOutDir: true,
    },
  };
});
