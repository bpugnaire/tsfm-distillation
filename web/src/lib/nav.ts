const pages = [
  ["index.html", "Draw"],
  ["play.html", "Play"],
  ["gallery.html", "Gallery"],
  ["results.html", "Results"],
  ["study.html", "Study"],
  ["learn.html", "Learn"],
  ["notes.html", "Lab notes"],
  ["status.html", "Status"],
] as const;

export function header(): void {
  const file = location.pathname.split("/").pop() || "index.html";
  const here = file.startsWith("learn") || file === "glossary.html" ? "learn.html" : file;
  const h = document.createElement("header");
  h.className = "site";
  h.innerHTML = `<span class="brand">tinycast</span><nav>${pages
    .map(([href, label]) => `<a href="${href}"${href === here ? ' aria-current="page"' : ""}>${label}</a>`)
    .join("")}</nav><span class="muted small tagline">big forecasting models, distilled into your browser</span>`;
  document.body.prepend(h);
}
