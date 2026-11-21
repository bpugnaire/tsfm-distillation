// KaTeX formulas and highlighted code cells for the posts. Code comes from code.json (extracted from the repo at
// build time by scripts/build_web_data.py), so what you read is what runs.
import "katex/dist/katex.min.css";
import renderMathInElement from "katex/contrib/auto-render";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import typescript from "highlight.js/lib/languages/typescript";
import type { Snippet } from "./data";

hljs.registerLanguage("python", python);
hljs.registerLanguage("typescript", typescript);
const REPO = "https://github.com/bpugnaire/tsfm-distillation/blob/main";

export function renderMath(root: HTMLElement = document.body): void {
  renderMathInElement(root, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
  });
}

const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

/** Highlight static <pre><code class="language-x"> blocks written in the post itself (usage examples). */
export function highlightStatic(root: HTMLElement = document.body): void {
  for (const el of root.querySelectorAll<HTMLElement>("pre code[class*='language-']")) {
    const lang = el.className.match(/language-(\w+)/)?.[1];
    if (lang && hljs.getLanguage(lang)) el.innerHTML = hljs.highlight(el.textContent ?? "", { language: lang }).value;
  }
}

/** Fill a <figure class="code" data-ref="..."> with the repo's source, collapsible past `maxLines`. */
export function renderCode(fig: HTMLElement, c: Snippet, maxLines = 28): void {
  const name = c.lang === "python" ? (c.code.match(/^(?:class|def)\s+(\w+)/m)?.[1] ?? "") : "";
  const html = hljs.highlight(c.code.trimEnd(), { language: c.lang }).value;
  const n = c.code.trimEnd().split("\n").length;
  const long = n > maxLines;
  fig.innerHTML = `<figcaption><span>${name ? `<code>${esc(name)}</code> · ` : ""}<a href="${REPO}/${c.file}#L${c.line}">${esc(c.file)}:${c.line}</a></span>
      <button class="copy" type="button">copy</button></figcaption>
    <div class="${long ? "collapsed" : ""}"><pre><code class="hljs">${html}</code></pre></div>
    ${long ? `<button class="more" type="button">show all ${n} lines</button>` : ""}`;
  const copy = fig.querySelector<HTMLButtonElement>(".copy")!;
  copy.onclick = async () => {
    try { await navigator.clipboard.writeText(c.code); copy.textContent = "copied"; } catch { copy.textContent = "select to copy"; }
    setTimeout(() => (copy.textContent = "copy"), 1500);
  };
  const more = fig.querySelector<HTMLButtonElement>(".more");
  if (more) more.onclick = () => {
    const box = more.previousElementSibling!;
    const c2 = box.classList.toggle("collapsed");
    more.textContent = c2 ? `show all ${n} lines` : "collapse";
  };
}
