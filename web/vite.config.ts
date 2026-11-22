import { resolve } from "node:path";
import { defineConfig } from "vite";

const page = (f: string) => resolve(__dirname, f);
export default defineConfig({
  base: "./",
  build: {
    target: "es2022", // top-level await in the page scripts
    rollupOptions: {
      input: {
        home: page("index.html"),
        results: page("results.html"),
        posts: page("posts.html"),
        cards: page("cards.html"),
        evaluation: page("post-evaluation.html"),
        data: page("post-data.html"),
        student: page("post-student.html"),
        distillation: page("post-distillation.html"),
        compression: page("post-compression.html"),
        browser: page("post-browser.html"),
      },
    },
  },
});
