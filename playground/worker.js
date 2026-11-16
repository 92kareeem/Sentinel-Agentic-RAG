// The verification runs in a Web Worker so the page stays responsive while
// Pyodide downloads and while a verdict is computed. The worker loads the
// sentinel-proof wheel that build.py placed next to this file: the same code
// that is published to PyPI, not a JavaScript rewrite of it.

"use strict";

const PYODIDE_VERSION = "0.28.3";
const PYODIDE_INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

const PY_SETUP = `
import json
from sentinel_proof import __version__, passages_from_texts, verify

def _run(payload: str) -> str:
    p = json.loads(payload)
    passages = passages_from_texts(list(p["passages"]))
    v = verify(
        p["answer"],
        passages,
        p.get("question") or "",
        citation_style=p.get("style") or "numeric",
        max_stripped_ratio=float(p.get("max_stripped", 0.5)),
    )
    return json.dumps(v.to_dict())

__version__
`;

let runVerify = null;
let packageVersion = null;

function status(text) {
  self.postMessage({ type: "status", text });
}

async function init(wheelUrl) {
  status("Loading the Python runtime (about 10 MB on the first visit, cached afterwards)…");
  importScripts(`${PYODIDE_INDEX_URL}pyodide.js`);
  const pyodide = await self.loadPyodide({ indexURL: PYODIDE_INDEX_URL });

  status("Installing sentinel-proof…");
  await pyodide.loadPackage(wheelUrl, { messageCallback: () => {} });

  packageVersion = String(pyodide.runPython(PY_SETUP));
  runVerify = pyodide.globals.get("_run");
  self.postMessage({
    type: "ready",
    packageVersion,
    pyodideVersion: PYODIDE_VERSION,
    pythonVersion: String(pyodide.runPython("import sys; '.'.join(map(str, sys.version_info[:3]))")),
  });
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  try {
    if (msg.type === "init") {
      await init(msg.wheelUrl);
    } else if (msg.type === "verify") {
      if (!runVerify) throw new Error("The runtime is not ready yet.");
      const out = runVerify(JSON.stringify(msg.payload));
      self.postMessage({ type: "result", id: msg.id, verdict: JSON.parse(out) });
    }
  } catch (err) {
    const message = err && err.message ? err.message : String(err);
    self.postMessage({ type: "error", id: msg.id, phase: msg.type, message });
  }
};
