const PAGES = [
  ["index.html", "Demo"],
  ["results.html", "Results"],
  ["posts.html", "Posts"],
] as const;
const REPO = "https://github.com/bpugnaire/tsfm-distillation";

/** Site header + footer. Posts (post-*.html) highlight "Posts". */
export function header(): void {
  const file = location.pathname.split("/").pop() || "index.html";
  const here = file.startsWith("post-") ? "posts.html" : file;
  const h = document.createElement("header");
  h.className = "site";
  h.innerHTML = `<div class="in"><a class="brand" href="index.html">tinycast</a><nav>${PAGES.map(
    ([href, label]) => `<a href="${href}"${href === here ? ' aria-current="page"' : ""}>${label}</a>`,
  ).join("")}<a href="${REPO}">GitHub</a></nav></div>`;
  document.body.prepend(h);
  const f = document.createElement("footer");
  f.className = "site";
  f.innerHTML = `Every number on this site is read from a results file in the <a href="${REPO}">repository</a>, produced by a committed script and config. Electricity data: RTE éCO2mix (Licence Ouverte).`;
  document.body.append(f);
}
