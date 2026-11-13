import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: {
    rollupOptions: {
      input: {
        draw: resolve(__dirname, "index.html"),
        gallery: resolve(__dirname, "gallery.html"),
        results: resolve(__dirname, "results.html"),
        notes: resolve(__dirname, "notes.html"),
        status: resolve(__dirname, "status.html"),
        learn: resolve(__dirname, "learn.html"),
        learn1: resolve(__dirname, "learn-1-basics.html"),
        learn2: resolve(__dirname, "learn-2-scoring.html"),
        learn3: resolve(__dirname, "learn-3-tsfm.html"),
        learn4: resolve(__dirname, "learn-4-distillation.html"),
        learn5: resolve(__dirname, "learn-5-deployment.html"),
        glossary: resolve(__dirname, "glossary.html"),
      },
    },
  },
});
