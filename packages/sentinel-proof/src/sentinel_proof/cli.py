"""``sentinel-proof``: run the gate from a shell or a CI step.

    sentinel-proof check --answer answer.txt --passages passages.json
    sentinel-proof check --answer answer.txt --passages passages.json --numeric --json
    echo "Refunds take 30 days [1]." | sentinel-proof check --passages passages.json --numeric

``passages.json`` is a list of strings, or a list of objects with ``text`` and
any other :class:`Passage` field. Exit code 0 when the answer is verified,
1 when it is not, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .grounding import MAX_STRIPPED_RATIO, Verdict, verify
from .passages import Passage, _with_ids, passages_from_texts


def _load_passages(path: Path) -> list[Passage]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise SystemExit("passages file must be a non-empty JSON list")
    if all(isinstance(x, str) for x in raw):
        return passages_from_texts(raw)
    fields = {f.name for f in dataclasses.fields(Passage)} - {"metadata"}
    out: list[Passage] = []
    for x in raw:
        if isinstance(x, str):
            x = {"text": x}
        if not isinstance(x, dict) or "text" not in x:
            raise SystemExit("each passage must be a string or an object with a 'text' field")
        known = {k: v for k, v in x.items() if k in fields}
        extra = {k: v for k, v in x.items() if k not in fields and k != "metadata"}
        out.append(Passage(**known, metadata={**x.get("metadata", {}), **extra}))
    return _with_ids(out)


def _as_dict(v: Verdict) -> dict[str, Any]:
    d = dataclasses.asdict(v)
    d["refusal"] = v.refusal.value if v.refusal else None
    d["passed"] = v.passed
    d["safe_text"] = v.safe_text()
    for c in d["claims"]:
        c["status"] = str(c["status"])
    return d


def _print_human(v: Verdict) -> None:
    if v.is_refusal:
        print("REFUSED by the model: the evidence does not answer the question.")
        return
    label = "VERIFIED" if v.ok else "NOT VERIFIED"
    print(
        f"{label}: {len(v.claims) - len(v.stripped)}/{len(v.claims)} claims supported, "
        f"{v.repaired_citations} citation(s) repaired, "
        f"{v.stripped_ratio:.0%} stripped (limit {MAX_STRIPPED_RATIO:.0%})"
    )
    for c in v.claims:
        mark = {"supported": "+", "repaired": "~", "stripped": "-"}[str(c.status)]
        detail = ""
        if c.status.value == "stripped" and c.unsupported_numbers:
            detail = f"  (figures not in cited evidence: {', '.join(c.unsupported_numbers)})"
        elif c.status.value == "repaired":
            cited = ", ".join(c.cited) or "nothing"
            detail = f"  (cited {cited}; supported by {', '.join(c.supported_by)})"
        print(f"  {mark} {c.text or '<no text>'}{detail}")
    if v.ok:
        print("\n" + v.answer)
    else:
        print("\n" + v.safe_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sentinel-proof", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="verify an answer against its passages")
    check.add_argument("--answer", type=Path, help="file with the answer (default: stdin)")
    check.add_argument("--passages", type=Path, required=True, help="JSON list of passages")
    check.add_argument("--question", default="", help="the user's question, if any")
    check.add_argument(
        "--numeric", action="store_true", help="citations are [1], [2]... not [chunk:id]"
    )
    check.add_argument(
        "--max-stripped",
        type=float,
        default=MAX_STRIPPED_RATIO,
        help="fail when more than this fraction of claims is stripped",
    )
    check.add_argument("--json", action="store_true", help="machine-readable output")

    args = parser.parse_args(argv)
    answer = args.answer.read_text(encoding="utf-8") if args.answer else sys.stdin.read()
    verdict = verify(
        answer,
        _load_passages(args.passages),
        args.question,
        citation_style="numeric" if args.numeric else "tag",
        max_stripped_ratio=args.max_stripped,
    )
    if args.json:
        print(json.dumps(_as_dict(verdict), indent=2))
    else:
        _print_human(verdict)
    return 0 if verdict.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
