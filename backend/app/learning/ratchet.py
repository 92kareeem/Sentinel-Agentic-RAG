"""The regression ratchet: every failure a person confirms becomes a test.

Role in architecture: the last link in the learning loop. The repair loop
heals one request. The casebook remembers what failed. This makes the fix
permanent — a question the system once got wrong, once fixed, can never be
got wrong the same way again without CI going red.

Why this cannot be the existing eval gate. That gate is an AVERAGE:
faithfulness >= 0.75 across the golden set. One question can fall from 1.0
to 0.0, move the mean by 0.05, and pass. An average is the right gate for
"is the system good overall" and structurally incapable of "did this
specific mistake come back". A ratchet needs a per-question gate, so it is
a separate suite with separate semantics rather than more rows in the
golden set — mixing them would also quietly change what the golden set's
historical numbers mean.

The trust boundary. Cases come from readers; a correction is a claim by
whoever typed it. Promotion is therefore a HUMAN act (evals/promote.py),
never automatic: auto-promoting corrections would let any user write the
test suite, and a wrong correction ("refunds are unlimited") would become a
test that fails the system for being right.

The refusal trap, and why `expect` is never inferred. The obvious move —
promote every refusal as "must refuse forever" — puts the ratchet at war
with the knowledge-gap report. That report tells the owner to add the
missing document; the moment they do, the correct behaviour flips to
answering, and a must-refuse test would fail CI for the exact fix the
product asked for. So `refuse` is only for questions a person has decided
are permanently out of scope, and it requires a written reason. A closed
gap is promoted as `answer`, which locks the fix in.

Two loops close here:

    refusal -> gap report -> owner adds the doc -> promote(answer)
            -> can never go back to refusing
    wrong answer -> reader's correction -> human review -> promote(answer)
            -> can never be wrong that way again

Nothing is deleted. A lesson that stops being true (the refund window
changed from 30 days to 60) is RETIRED with a reason, and ids are dense
across active and retired, so a quiet deletion leaves a gap the validator
catches. The only way to make a regression go away is on the record.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.schemas import Case, CaseDiagnosis

Expect = Literal["answer", "refuse"]
Status = Literal["pending", "enforced"]

# Per-item bars, deliberately below the golden set's aggregate 0.75. A single
# judge call is noisier than a mean over twenty, and a ratchet that fails on
# judge wobble gets switched off — the worst possible outcome for a mechanism
# whose whole value is that it is never switched off. The bar is "the
# corrected fact is there and grounded", not "the prose is ideal".
MIN_FAITHFULNESS = 0.7
MIN_COMPLETENESS = 0.7

# Operational, not a quality signal: the request ran out of budget before the
# question got a fair attempt. A test built on it would assert a rate limit.
UNPROMOTABLE = frozenset({CaseDiagnosis.CAPACITY_EXCEEDED})

_ID_RE = re.compile(r"^R(\d{3,})$")


class Provenance(BaseModel):
    """Where a regression came from — the case a person reviewed."""

    case_id: str
    diagnosis: CaseDiagnosis
    owner_id: str
    reported_at: str  # when the original question was asked


class RegressionItem(BaseModel):
    id: str
    question: str
    expect: Expect
    # The fact the answer must contain. For a reader-reported case this starts
    # as their correction, reviewed at promotion — see the module docstring.
    reference_answer: str | None = None
    # Required for `refuse`: why this question is permanently out of scope.
    # Optional context otherwise.
    note: str | None = None
    status: Status = "pending"
    source: Provenance
    promoted_at: str
    locked_at: str | None = None  # when it went from pending to enforced


class RetiredItem(BaseModel):
    id: str
    question: str
    retired_at: str
    reason: str = Field(min_length=1)


class RegressionSuite(BaseModel):
    # How many ids have ever been handed out. Only promote() moves it. It is
    # what makes deleting the most recent regression detectable — see
    # validate(). Editing it down by hand to hide a deletion is possible, but
    # it is a deliberate two-line change in a reviewed diff, not an accident.
    issued: int = 0
    active: list[RegressionItem] = Field(default_factory=list)
    retired: list[RetiredItem] = Field(default_factory=list)


class PromotionError(ValueError):
    """A case cannot become a regression as requested. The message says why,
    in terms the person running promote.py can act on."""


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ storage


def load(path: Path) -> RegressionSuite:
    if not path.exists():
        return RegressionSuite()
    return RegressionSuite.model_validate_json(path.read_text(encoding="utf-8"))


def save(suite: RegressionSuite, path: Path) -> None:
    """Validate, then write atomically. A suite that breaks an invariant is
    never written — the file on disk is always one validate() accepts."""
    problems = validate(suite)
    if problems:
        raise PromotionError("refusing to write an invalid suite:\n  " + "\n  ".join(problems))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(suite.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------- invariants


def validate(suite: RegressionSuite) -> list[str]:
    """Every way the suite can be wrong, as readable problems. Empty = valid.

    Run by save() and by a unit test against the committed file, so a hand
    edit that breaks the ratchet fails CI rather than weakening it quietly.
    """
    problems: list[str] = []
    ids = [i.id for i in suite.active] + [r.id for r in suite.retired]

    numbers: list[int] = []
    for item_id in ids:
        m = _ID_RE.match(item_id)
        if not m:
            problems.append(f"{item_id}: ids look like R001")
        else:
            numbers.append(int(m.group(1)))
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        problems.append(f"duplicate ids: {', '.join(dupes)}")

    # Two guards against a quiet deletion, because each alone has a blind spot.
    #
    # `issued` counts every id ever handed out. Without it, deleting the NEWEST
    # regression leaves R001, R002 — dense, and apparently fine. That is the
    # blind spot that matters most, because the newest lesson is the one still
    # failing and the one someone is most tempted to delete.
    if len(ids) != suite.issued:
        problems.append(
            f"{suite.issued} regression id(s) issued but {len(ids)} present: a regression "
            "was deleted rather than retired. Restore it, or retire it with a reason."
        )
    # Dense numbering catches a middle deletion even if `issued` was edited
    # down to match: R001, R002, R004 has a hole that is itself the tell.
    if numbers and sorted(set(numbers)) != list(range(1, max(numbers) + 1)):
        missing = sorted(set(range(1, max(numbers) + 1)) - set(numbers))
        problems.append(
            "ids have gaps (" + ", ".join(f"R{n:03d}" for n in missing) + "): a regression "
            "was deleted rather than retired. Restore it, or retire it with a reason."
        )

    seen_cases: set[str] = set()
    for item in suite.active:
        if item.expect == "answer" and not (item.reference_answer or "").strip():
            problems.append(
                f"{item.id}: expects an answer but has no reference_answer to judge it against"
            )
        if item.expect == "refuse" and not (item.note or "").strip():
            problems.append(
                f"{item.id}: expects a refusal but has no note saying why the question is "
                "permanently out of scope"
            )
        if item.status == "enforced" and not item.locked_at:
            problems.append(f"{item.id}: enforced but never locked (locked_at missing)")
        if item.source.case_id in seen_cases:
            problems.append(f"{item.id}: case {item.source.case_id} is promoted twice")
        seen_cases.add(item.source.case_id)
        if item.source.diagnosis in UNPROMOTABLE:
            problems.append(f"{item.id}: {item.source.diagnosis} is not a quality signal")
    return problems


def next_id(suite: RegressionSuite) -> str:
    """From the issued counter, so an id is never reused — not after a
    retirement, and not after a deletion either. History reads unambiguously:
    "R004 was retired" always means the same R004."""
    return f"R{suite.issued + 1:03d}"


# ---------------------------------------------------------------- promotion


def promote(
    suite: RegressionSuite,
    case: Case,
    *,
    expect: Expect = "answer",
    reference: str | None = None,
    note: str | None = None,
) -> RegressionItem:
    """Turn a reviewed case into a pending regression. Mutates `suite`.

    Always lands as `pending`. A case is promoted at the moment someone
    confirms it is a real failure — usually before it is fixed — and a
    pending item is tracked without blocking CI. It becomes `enforced` once
    it passes (see lock()); from then on it can never fail again unnoticed.
    """
    if case.diagnosis in UNPROMOTABLE:
        raise PromotionError(
            f"{case.diagnosis} means the request ran out of budget before the question got "
            "a fair attempt. That is a capacity problem, not a wrong answer, and a test "
            "built on it would assert a rate limit."
        )
    if any(i.source.case_id == case.case_id for i in suite.active):
        raise PromotionError(f"case {case.case_id} is already a regression")

    if expect == "answer":
        # An explicit reference wins over the reader's correction: promotion is
        # where a human reviews the correction, and may well fix it.
        ref = (reference or case.correction or "").strip()
        if not ref:
            raise PromotionError(
                "an 'answer' regression needs the correct answer to judge against. This "
                "case has no reader correction, so pass --reference with what the right "
                "answer is."
            )
    else:
        ref = None
        if not (note or "").strip():
            raise PromotionError(
                "a 'refuse' regression asserts the documents will NEVER answer this. That "
                "must be a decision, not an inference — the gap report exists to get missing "
                "documents written. Pass --note saying why it is permanently out of scope."
            )

    item = RegressionItem(
        id=next_id(suite),
        question=case.question,
        expect=expect,
        reference_answer=ref,
        note=(note or "").strip() or None,
        status="pending",
        source=Provenance(
            case_id=case.case_id,
            diagnosis=case.diagnosis,
            owner_id=case.owner_id,
            reported_at=case.created_at,
        ),
        promoted_at=now(),
    )
    suite.active.append(item)
    suite.issued += 1
    return item


def lock(suite: RegressionSuite, item_id: str) -> RegressionItem:
    """pending -> enforced. The one-way step: from here a failure is a red CI."""
    item = _find(suite, item_id)
    if item.status == "enforced":
        raise PromotionError(f"{item_id} is already enforced")
    item.status = "enforced"
    item.locked_at = now()
    return item


def retire(suite: RegressionSuite, item_id: str, reason: str) -> RetiredItem:
    """Move a regression out of the active suite, on the record."""
    if not reason.strip():
        raise PromotionError(
            "retiring a regression needs a reason — e.g. 'refund window changed to 60 "
            "days in the 2026 policy'. A lesson can stop being true; it cannot quietly vanish."
        )
    item = _find(suite, item_id)
    suite.active.remove(item)
    retired = RetiredItem(id=item.id, question=item.question, retired_at=now(), reason=reason)
    suite.retired.append(retired)
    return retired


def _find(suite: RegressionSuite, item_id: str) -> RegressionItem:
    for item in suite.active:
        if item.id == item_id:
            return item
    if any(r.id == item_id for r in suite.retired):
        raise PromotionError(f"{item_id} is retired")
    raise PromotionError(f"no active regression {item_id}")


# ------------------------------------------------------------------ scoring


class Outcome(BaseModel):
    """One regression's result in one eval run."""

    id: str
    status: Status
    passed: bool | None  # None = could not be measured (rate limit, outage)
    reason: str


def evaluate(item: RegressionItem, row: Mapping[str, Any]) -> Outcome:
    """Judge one eval row against what the regression expects.

    Pure — no LLM, no I/O — so the rules the ratchet enforces are unit-tested
    rather than trusted. `row` is what evals/run.py's run_item() returns.
    """
    if row.get("error"):
        # An upstream failure says nothing about whether the mistake came back.
        # Reporting it as a regression would teach people to ignore the gate.
        return Outcome(
            id=item.id, status=item.status, passed=None,
            reason=f"unmeasured: {str(row['error'])[:100]}",
        )

    refused = bool(row.get("refused"))
    if item.expect == "refuse":
        if refused:
            return Outcome(
                id=item.id, status=item.status, passed=True, reason="refused, as expected"
            )
        return Outcome(
            id=item.id, status=item.status, passed=False,
            reason="answered a question that is out of scope by decision",
        )

    if refused:
        return Outcome(
            id=item.id, status=item.status, passed=False,
            reason="refused a question it is expected to answer",
        )
    faith = float(row.get("faithfulness", 0.0))
    compl = float(row.get("completeness", 0.0))
    if compl < MIN_COMPLETENESS:
        return Outcome(
            id=item.id, status=item.status, passed=False,
            reason=f"answer is missing the corrected fact (completeness {compl:.2f})",
        )
    if faith < MIN_FAITHFULNESS:
        return Outcome(
            id=item.id, status=item.status, passed=False,
            reason=f"answer is not grounded in the retrieved documents (faithfulness {faith:.2f})",
        )
    return Outcome(
        id=item.id, status=item.status, passed=True,
        reason=f"faithfulness {faith:.2f}, completeness {compl:.2f}",
    )


def blocking_failures(outcomes: list[Outcome]) -> list[Outcome]:
    """Enforced regressions that failed. Any of these fails the run."""
    return [o for o in outcomes if o.status == "enforced" and o.passed is False]


def ready_to_lock(outcomes: list[Outcome]) -> list[Outcome]:
    """Pending regressions that now pass — fixed, and waiting to be locked in."""
    return [o for o in outcomes if o.status == "pending" and o.passed is True]
