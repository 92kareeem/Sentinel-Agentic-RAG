"""Open the built playground in a real browser and check it end to end.

    python playground/build.py
    pip install playwright && playwright install chromium
    python playground/test_playground.py

Serves ``playground/dist`` on a local port, loads the page in headless
Chromium, waits for Pyodide and the wheel, runs every example and asserts
the verdicts the examples were written to demonstrate. Also round-trips a
share link and checks that the page's JSON matches what the Python package
returns for the same input, so the browser can never disagree with pip.
"""

from __future__ import annotations

import functools
import http.server
import json
import sys
import threading
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

DIST = Path(__file__).resolve().parent / "dist"
READY_TIMEOUT_MS = 180_000  # the first Pyodide download on a cold CI runner


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # the request log is noise here
        pass


def serve(directory: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    handler = functools.partial(_QuietHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def wait_ready(page: Page) -> None:
    page.wait_for_function("() => window.__playground && window.__playground.isReady()", timeout=READY_TIMEOUT_MS)


def verdict_after(page: Page, action) -> dict:  # type: ignore[no-untyped-def]
    page.evaluate("() => { window.__playground.__seen = window.__playground.lastVerdict(); }")
    action()
    page.wait_for_function(
        "() => window.__playground.lastVerdict() && window.__playground.lastVerdict() !== window.__playground.__seen",
        timeout=30_000,
    )
    return page.evaluate("() => window.__playground.lastVerdict()")


def main() -> int:
    if not (DIST / "manifest.json").exists():
        raise SystemExit("run `python playground/build.py` first")
    server, port = serve(DIST)
    base = f"http://127.0.0.1:{port}/"
    errors: list[str] = []
    failures: list[str] = []

    def check(cond: bool, what: str) -> None:
        print(("  ok   " if cond else "  FAIL ") + what)
        if not cond:
            failures.append(what)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        print("load")
        page.goto(base)
        wait_ready(page)
        status = page.locator("#status").inner_text()
        check(status.startswith("Ready."), f"status says ready: {status!r}")
        page.wait_for_function("() => window.__playground.lastVerdict() !== null", timeout=30_000)

        # Example 1: one figure invented (stripped), one citation re-pointed.
        v = page.evaluate("() => window.__playground.lastVerdict()")
        check(v["ok"] is True, "example 1 verified")
        check(v["repaired_citations"] == 1, "example 1 repaired one citation")
        statuses = [c["status"] for c in v["claims"]]
        check(statuses == ["supported", "repaired", "stripped"], f"example 1 statuses {statuses}")
        check(page.locator("#verdict").get_attribute("class") == "verdict ok", "example 1 verdict card is green")
        check(page.locator(".claim.stripped").count() == 1, "example 1 renders one stripped row")
        check("2 business days" not in page.locator("#safe-text").inner_text(), "example 1 safe text drops the fabrication")

        # Example 2: refused, draft not shown.
        v = verdict_after(page, lambda: page.select_option("#preset", "1"))
        check(v["ok"] is False and v["refusal"] == "UNVERIFIABLE_ANSWER", "example 2 refused with reason")
        safe = page.locator("#safe-text").inner_text()
        check("9,999" not in safe and "7%" not in safe, "example 2 safe text does not leak the draft")
        check(page.locator("#verdict").get_attribute("class") == "verdict fail", "example 2 verdict card is red")

        # Example 3: everything supported.
        v = verdict_after(page, lambda: page.select_option("#preset", "2"))
        check(v["ok"] and all(c["status"] == "supported" for c in v["claims"]), "example 3 fully supported")

        # Example 4: tag style, one citation repaired.
        v = verdict_after(page, lambda: page.select_option("#preset", "3"))
        check(v["ok"] and v["repaired_citations"] == 1, "example 4 (tag style) repaired one citation")
        check("[chunk:p1]" in page.locator("#safe-text").inner_text(), "example 4 keeps tag citations")

        # Example 5: model refusal passes through.
        v = verdict_after(page, lambda: page.select_option("#preset", "4"))
        check(v["is_refusal"] is True and v["refusal"] == "INSUFFICIENT_EVIDENCE", "example 5 passes the refusal through")
        check(page.locator("#verdict").get_attribute("class") == "verdict refusal", "example 5 verdict card is amber")

        # Typing: edits re-verify, and the empty state is handled.
        page.select_option("#preset", "0")
        page.wait_for_function("() => window.__playground.lastVerdict() && window.__playground.lastVerdict().ok === true")
        v = verdict_after(page, lambda: page.fill("#answer", "Customers can get a refund within 45 days [1]."))
        check(v["ok"] is False, "typed answer with a wrong figure is refused")
        page.fill("#answer", "")
        page.wait_for_function("() => window.__playground.lastVerdict() === null")
        check(page.locator("#verdict-note").inner_text() == "Add the answer to check.", "empty answer shows a hint, not an error")

        # Share link round trip.
        page.select_option("#preset", "1")
        page.wait_for_function("() => window.__playground.lastVerdict() && window.__playground.lastVerdict().ok === false")
        page.context.grant_permissions(["clipboard-read", "clipboard-write"], origin=base)
        page.click("#share")
        link = page.evaluate("() => window.location.href")
        check("#s=" in link, "share link written to the address bar")
        page2 = browser.new_page()
        page2.goto(link)
        wait_ready(page2)
        page2.wait_for_function("() => window.__playground.lastVerdict() !== null", timeout=30_000)
        check(page2.locator("#answer").input_value().startswith("The premium plan costs $9,999"), "share link restores the inputs")
        v2 = page2.evaluate("() => window.__playground.lastVerdict()")
        check(v2["refusal"] == "UNVERIFIABLE_ANSWER", "share link reproduces the verdict")

        # The browser and pip agree, byte for byte, on the same input.
        try:
            from sentinel_proof import passages_from_texts, verify
        except ImportError:
            sys.path.insert(0, str(DIST.parent.parent / "packages" / "sentinel-proof" / "src"))
            from sentinel_proof import passages_from_texts, verify
        page.select_option("#preset", "0")
        page.wait_for_function("() => window.__playground.lastVerdict() && window.__playground.lastVerdict().ok === true")
        browser_json = page.locator("#json").inner_text()
        local = verify(
            "Customers can get a refund within 30 days [1]. "
            "Requests above 500 USD need manager approval [1]. "
            "Refunds are processed in 2 business days [2].",
            passages_from_texts(
                [
                    "Customers may request a refund within 30 days of purchase.",
                    "Refund requests above 500 USD require manager approval.",
                ]
            ),
            "What is the refund policy?",
            citation_style="numeric",
        ).to_dict()
        check(json.loads(browser_json) == local, "browser verdict equals the Python package's verdict")

        # Mobile width: no horizontal overflow.
        page.set_viewport_size({"width": 390, "height": 844})
        overflow = page.evaluate("() => document.documentElement.scrollWidth > document.documentElement.clientWidth")
        check(not overflow, "no horizontal scroll at phone width")

        browser.close()
    server.shutdown()

    check(not errors, "no console or page errors" + (": " + "; ".join(errors[:3]) if errors else ""))
    if failures:
        print(f"\n{len(failures)} check(s) failed")
        return 1
    print("\nall playground checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
