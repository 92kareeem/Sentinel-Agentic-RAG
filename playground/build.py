"""Assemble the playground into ``playground/dist``.

The playground is static files plus the sentinel-proof wheel built from this
checkout, so what the page runs is exactly the code in the repository. Run:

    python playground/build.py            # writes playground/dist
    python -m http.server -d playground/dist 8000

Requires the ``build`` package (``pip install build``).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "packages" / "sentinel-proof"
HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"
STATIC = ("index.html", "style.css", "app.js", "worker.js")


def package_version() -> str:
    text = (PACKAGE / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise SystemExit("could not read the package version from pyproject.toml")
    return m.group(1)


def build_wheel(out: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out), str(PACKAGE)],
        check=True,
    )
    # setuptools leaves a build/ tree inside the package; it is not source.
    shutil.rmtree(PACKAGE / "build", ignore_errors=True)
    wheels = sorted(out.glob("sentinel_proof-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {out}, found {wheels}")
    return wheels[0]


def main() -> int:
    # ignore_errors: a sync client (OneDrive) can hold the directory itself
    # open on Windows; the files inside are still removed, which is what
    # matters, and stale wheels are cleared explicitly below.
    shutil.rmtree(DIST, ignore_errors=True)
    (DIST / "wheels").mkdir(parents=True, exist_ok=True)
    for stale in (DIST / "wheels").glob("*.whl"):
        stale.unlink()

    for name in STATIC:
        shutil.copyfile(HERE / name, DIST / name)

    wheel = build_wheel(DIST / "wheels")
    version = package_version()
    if version not in wheel.name:
        raise SystemExit(f"wheel {wheel.name} does not match pyproject version {version}")

    manifest = {
        "wheel": f"wheels/{wheel.name}",
        "version": version,
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    (DIST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # GitHub Pages would otherwise run Jekyll over the output.
    (DIST / ".nojekyll").write_text("", encoding="utf-8")

    print(f"playground built: {DIST} (sentinel-proof {version}, {wheel.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
