"""Review failures and promote them into the regression suite.

Usage (from repo root, venv python):

    python evals/promote.py list                      # what could become a test
    python evals/promote.py promote <case_id>         # reader's correction as the reference
    python evals/promote.py promote <case_id> --reference "It is 30 days."
    python evals/promote.py promote <case_id> --expect refuse --note "Payroll is never in scope."
    python evals/promote.py lock R004                 # pending -> enforced
    python evals/promote.py retire R002 --reason "Refund window changed to 60 days."

Reads the casebook the server writes: in local mode ./index/cases.json (set
INDEX_DIR to point elsewhere), in AWS mode the documents table, scoped to
--owner.

This is deliberately a person's tool, not a background job. Every
regression it writes becomes something CI holds the system to forever, and
the reference answers come from readers' corrections — claims by whoever
typed them. Promoting them unreviewed would let any user write the test
suite. See backend/app/learning/ratchet.py for the full reasoning.

Subcommands rather than interactive prompts so every step is scriptable,
reviewable in shell history, and testable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.learning import casebook, ratchet  # noqa: E402
from app.models.schemas import Case  # noqa: E402

SUITE_PATH = Path(__file__).resolve().parent / "regressions.json"


def _candidates(owner: str, suite: ratchet.RegressionSuite) -> list[Case]:
    promoted = {i.source.case_id for i in suite.active}
    return [
        c for c in reversed(casebook.list_for_owner(owner))  # newest first
        if c.case_id not in promoted and c.diagnosis not in ratchet.UNPROMOTABLE
    ]


def cmd_list(args: argparse.Namespace) -> int:
    suite = ratchet.load(args.suite)
    cases = _candidates(args.owner, suite)
    if not cases:
        print(f"Nothing to review for owner '{args.owner}'.")
        return 0
    print(f"{len(cases)} case(s) for owner '{args.owner}' not yet in the regression suite:\n")
    for c in cases:
        print(f"  {c.case_id}")
        print(f"    {c.diagnosis.value}  ({c.outcome.value.lower()}, asked {c.created_at})")
        print(f"    Q: {c.question}")
        if c.correction:
            print(f"    reader says: {c.correction}")
            print(f"    -> python evals/promote.py promote {c.case_id}")
        else:
            print("    no correction: promote with --reference, or --expect refuse --note")
        print()
    return 0


def _find_case(owner: str, case_id: str) -> Case:
    for c in casebook.list_for_owner(owner):
        if c.case_id == case_id:
            return c
    raise ratchet.PromotionError(
        f"no case {case_id} for owner '{owner}'. Run `list` to see what exists; in AWS "
        "mode check --owner, since cases are scoped per owner."
    )


def cmd_promote(args: argparse.Namespace) -> int:
    suite = ratchet.load(args.suite)
    case = _find_case(args.owner, args.case_id)
    item = ratchet.promote(
        suite, case, expect=args.expect, reference=args.reference, note=args.note
    )
    ratchet.save(suite, args.suite)
    print(f"Promoted {case.case_id} as {item.id} (pending).")
    print(f"  Q: {item.question}")
    if item.reference_answer:
        print(f"  must contain: {item.reference_answer}")
    else:
        print(f"  must refuse — {item.note}")
    print(
        "\nPending means it is tracked but does not block CI yet. Once a run of "
        "evals/run.py shows it passing, lock it:\n"
        f"  python evals/promote.py lock {item.id}   (or: python evals/run.py --lock)"
    )
    return 0


def cmd_lock(args: argparse.Namespace) -> int:
    suite = ratchet.load(args.suite)
    item = ratchet.lock(suite, args.id)
    ratchet.save(suite, args.suite)
    print(f"{item.id} is enforced. From now on, if it fails, the eval run fails.")
    return 0


def cmd_retire(args: argparse.Namespace) -> int:
    suite = ratchet.load(args.suite)
    retired = ratchet.retire(suite, args.id, args.reason)
    ratchet.save(suite, args.suite)
    print(f"{retired.id} retired: {retired.reason}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite", type=Path, default=SUITE_PATH, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="cases not yet promoted, newest first")
    p_list.add_argument("--owner", default="demo", help="casebook owner (default: demo)")
    p_list.set_defaults(func=cmd_list)

    p_prom = sub.add_parser("promote", help="turn a reviewed case into a pending regression")
    p_prom.add_argument("case_id")
    p_prom.add_argument("--owner", default="demo")
    p_prom.add_argument("--expect", choices=["answer", "refuse"], default="answer")
    p_prom.add_argument(
        "--reference", help="the correct answer; overrides the reader's correction"
    )
    p_prom.add_argument("--note", help="required for --expect refuse: why it is out of scope")
    p_prom.set_defaults(func=cmd_promote)

    p_lock = sub.add_parser("lock", help="pending -> enforced")
    p_lock.add_argument("id")
    p_lock.set_defaults(func=cmd_lock)

    p_ret = sub.add_parser("retire", help="remove a regression from the suite, with a reason")
    p_ret.add_argument("id")
    p_ret.add_argument("--reason", required=True)
    p_ret.set_defaults(func=cmd_retire)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ratchet.PromotionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
