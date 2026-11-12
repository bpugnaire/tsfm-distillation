const pages = [
  ["index.html", "Draw"],
  ["gallery.html", "Gallery"],
  ["results.html", "Results"],
  ["notes.html", "Lab notes"],
] as const;

export function header(): void {
  const here = location.pathname.split("/").pop() || "index.html";
  const h = document.createElement("header");
  h.className = "site";
  h.innerHTML = `<span class="brand">tinycast</span><nav>${pages
    .map(([href, label]) => `<a href="${href}"${href === here ? ' aria-current="page"' : ""}>${label}</a>`)
    .join("")}</nav><span class="muted small">distilling forecasting foundation models into the browser · work in progress</span>`;
  document.body.prepend(h);
}
