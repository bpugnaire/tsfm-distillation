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
      },
    },
  },
});
