# Browser playground

The static site at <https://92kareeem.github.io/sentinel-rag/>. It runs the
`sentinel-proof` wheel built from this checkout inside [Pyodide](https://pyodide.org),
so the verdict in the browser is the verdict `pip install sentinel-proof` gives.
Nothing a visitor types leaves their tab.

```bash
pip install build playwright && playwright install chromium
python playground/build.py              # -> playground/dist
python playground/test_playground.py    # end-to-end check in headless Chromium
python -m http.server -d playground/dist 8000
```

- `index.html`, `style.css`, `app.js`: the page. No framework, no bundler.
- `worker.js`: loads Pyodide and the wheel in a Web Worker and runs `verify()`.
- `build.py`: copies the static files, builds the wheel, writes `manifest.json`.
- `test_playground.py`: what CI runs on every pull request that touches the
  playground or the package (`.github/workflows/pages.yml`). It loads every
  example, edits the inputs, round-trips a share link, and asserts the page's
  JSON equals the Python package's `Verdict.to_dict()` for the same input.

The Pyodide version is pinned in `worker.js`. `dist/` is a build output and is
not committed.
