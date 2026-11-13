"""Wrap each Learn chapter body (learn-src/*.html) in the shared page shell -> web/<name>.html.

Run: python3 web/learn-src/build.py   (no dependencies; also run by `make web-data`).
"""

from pathlib import Path

HERE = Path(__file__).parent
SHELL = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="stylesheet" href="/src/style.css" />
</head>
<body>
  <!-- GENERATED from web/learn-src/{src} by web/learn-src/build.py: edit the source, not this file -->
  <div class="course">
    <aside id="sidebar"></aside>
    <article>
{body}
      <nav class="chapter-nav" id="chapter-nav"></nav>
    </article>
  </div>
  <script type="module" src="/src/pages/learn.ts"></script>
</body>
</html>
"""

for src in sorted(HERE.glob("*.html")):
    text = src.read_text()
    first, body = text.split("\n", 1)
    assert first.startswith("<!-- title: "), f"{src.name}: first line must be <!-- title: ... -->"
    title = first[len("<!-- title: ") : -len(" -->")]
    out = HERE.parent / src.name
    out.write_text(SHELL.format(title=title, src=src.name, body=body))
    print("wrote", out.name)
