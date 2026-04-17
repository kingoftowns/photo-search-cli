import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy API + image routes to the FastAPI service during local dev so the
// browser can fetch same-origin.
//
// Override the target with VITE_API_PROXY=http://host:port when the API
// isn't on localhost:8000.
const apiTarget = process.env.VITE_API_PROXY ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
      "/thumbs": { target: apiTarget, changeOrigin: true },
      "/originals": { target: apiTarget, changeOrigin: true },
    },
  },
});
