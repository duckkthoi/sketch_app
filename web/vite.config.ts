import { defineConfig } from "vite";

// Build ra ../static để Flask phục vụ trực tiếp; khi dev thì proxy /api sang Flask (cổng 5000).
export default defineConfig({
  build: { outDir: "../static", emptyOutDir: true, assetsDir: "assets", target: "es2020" },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:5000", "/healthz": "http://127.0.0.1:5000" },
  },
});
