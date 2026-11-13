// Readability layer for the Learn pages: KaTeX formulas and highlighted, numbered, collapsible code cells.
import "katex/dist/katex.min.css";
import renderMathInElement from "katex/contrib/auto-render";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";

hljs.registerLanguage("python", python);

/** Render every $$...$$ (display) and \(...\) (inline) formula inside `root`. */
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

/** First paragraph of a Python docstring, if any: shown above the code as "what it does". */
function docstringOf(code: string): string | null {
  const m = code.match(/^[^\n]*\n\s*(?:"""|''')([\s\S]*?)(?:"""|''')/);
  if (!m) return null;
  return m[1].trim().split(/\n\s*\n/)[0].replace(/\s+/g, " ");
}

/** Fill a <figure class="code"> with a highlighted, line-numbered, collapsible code cell. */
export function renderCode(fig: HTMLElement, c: { file: string; line: number; code: string }, maxLines = 22): void {
  const name = c.code.match(/^(?:class|def)\s+(\w+)/m)?.[1] ?? "code";
  const doc = docstringOf(c.code);
  const html = hljs.highlight(c.code, { language: "python" }).value;
  // keep highlight spans balanced per line: close open spans at line end and reopen on the next line
  const lines: string[] = [];
  let open: string[] = [];
  for (const raw of html.split("\n")) {
    const prefix = open.join("");
    const tags = raw.match(/<span[^>]*>|<\/span>/g) ?? [];
    for (const t of tags) {
      if (t === "</span>") open.pop();
      else open.push(t);
    }
    lines.push(`<span class="ln">${prefix}${raw}${"</span>".repeat(open.length)}</span>`);
    open = [...open];
  }
  const long = lines.length > maxLines;
  fig.innerHTML = `
    <figcaption><span class="cap-title"><b>In our code</b><code>${esc(name)}</code> · <span class="muted">${esc(c.file)}:${c.line}</span></span>
      <button class="copy" type="button">copy</button></figcaption>
    ${doc ? `<div class="doc">${esc(doc)}</div>` : ""}
    <div class="scroll${long ? " collapsed" : ""}"><pre><code class="hljs">${lines.join("")}</code></pre></div>
    ${long ? `<button class="more" type="button">show all ${lines.length} lines ▾</button>` : ""}`;
  const copy = fig.querySelector<HTMLButtonElement>(".copy")!;
  copy.onclick = async () => {
    try {
      await navigator.clipboard.writeText(c.code);
      copy.textContent = "copied ✓";
    } catch {
      copy.textContent = "select & copy";
    }
    setTimeout(() => (copy.textContent = "copy"), 1500);
  };
  const more = fig.querySelector<HTMLButtonElement>(".more");
  if (more) {
    more.onclick = () => {
      const box = fig.querySelector(".scroll")!;
      const collapsed = box.classList.toggle("collapsed");
      more.textContent = collapsed ? `show all ${lines.length} lines ▾` : "collapse ▴";
    };
  }
}
