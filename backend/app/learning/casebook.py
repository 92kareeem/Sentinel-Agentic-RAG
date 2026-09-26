"""The casebook: every question the system handled badly, kept and classified.

Role in architecture: this is what makes the system improve rather than merely
recover. The repair loop heals ONE request — it rewrites a query, retries, and
the next identical question starts from scratch and fails the same way. The
casebook is the other half: a durable record of what failed and why, so the
same gap is not rediscovered by every user who happens to ask.

The business value is in who the record is FOR. A refusal is a fact about the
corpus — "nothing here covers parental leave for contractors" — and today
nobody owns that fact. The person asking rephrases or gives up; the document
owner, the one person who could fix it, never hears about it. A knowledge-gap
report turns those refusals into a ranked work list, which is the difference
between a chatbot and a documentation product.

Storage, and why it costs nothing:

  * local_mode -> a JSON file beside the registry, written atomically
  * AWS        -> the EXISTING documents table, pk="CASE#<owner>",
                  sk="<created_at>#<case_id>"

No new table, no GSI, no new IAM statement. The sort key is time-ordered, so
"the last 30 days for this owner" is one Query with a key condition and no
filter scan — the same access pattern the registry already relies on. The
alternative, a dedicated table, would have been cleaner in isolation and is
the right call at volume; at this scale it buys nothing and costs a resource.

Writes are best-effort by design. A case is a byproduct of answering, and a
user who asked a hard question should not ALSO lose their response because
the byproduct could not be persisted.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.models.schemas import (
    Case,
    CaseDiagnosis,
    CaseOutcome,
    FeedbackVerdict,
    KnowledgeGap,
    KnowledgeGapReport,
)
from app.observability.logging import get_logger

_LOCAL_FILE = "cases.json"
_STATS_FILE = "case_stats.json"
_CLAIMS_FILE = "feedback_claims.json"
_PK_PREFIX = "CASE#"
_STATS_PK_PREFIX = "STATS#"
_CLAIM_PK_PREFIX = "FEEDBACK#"

_local_lock = threading.RLock()
_logger = get_logger()

# What a document owner should do about each diagnosis. Written for someone who
# owns the documentation and does not know how retrieval works — the report is
# useless if acting on it requires reading this module.
RECOMMENDED_ACTION: dict[CaseDiagnosis, str] = {
    CaseDiagnosis.NO_EVIDENCE_FOUND: (
        "No document in this workspace covers this. Add or upload one, or "
        "confirm it is out of scope."
    ),
    CaseDiagnosis.EVIDENCE_OFF_TOPIC: (
        "Related documents exist but none answer this specific case. The "
        "topic is probably covered in general terms and missing the detail "
        "people are asking for."
    ),
    CaseDiagnosis.ANSWER_UNVERIFIABLE: (
        "The evidence is present but an answer could not be verified against "
        "it. Check whether the passage is ambiguous, contradicts another "
        "document, or is split awkwardly across sections."
    ),
    CaseDiagnosis.CITATION_MISATTRIBUTED: (
        "Answers here are landing on the wrong passage before correction, "
        "which usually means several sections say similar things. Worth a "
        "review for duplicated or superseded content."
    ),
    CaseDiagnosis.CAPACITY_EXCEEDED: (
        "These questions ran out of processing budget rather than evidence. "
        "This is a system limit, not a documentation gap."
    ),
    CaseDiagnosis.USER_REPORTED_INCORRECT: (
        "Readers say the answer here was wrong. The evidence was found and "
        "cited, so this is usually two documents disagreeing, or a passage "
        "that has been superseded and not removed. Check the cited sections "
        "against each other first."
    ),
    CaseDiagnosis.USER_REPORTED_INCOMPLETE: (
        "Readers say the answer here was true but partial. Usually the topic "
        "is split across documents, or a section stops short of the case "
        "people actually have. Consider consolidating it in one place."
    ),
}

# Diagnoses only a human can assert. Separate from GAP_DIAGNOSES because they
# mean something different: a gap is "the documents do not cover this", while
# these are "the documents covered it and the answer was still wrong". Both
# belong on the owner's work list, so the report draws from both sets, but
# conflating them would let a correctness bug hide inside a content metric.
FEEDBACK_DIAGNOSES = frozenset(
    {CaseDiagnosis.USER_REPORTED_INCORRECT, CaseDiagnosis.USER_REPORTED_INCOMPLETE}
)

# Which diagnosis a verdict opens. HELPFUL is absent deliberately: a good
# answer is not a case. Recording one would turn the casebook into a query log
# and bury the failures it exists to surface — the positive signal lives in
# the daily counters instead, where it serves as a denominator.
VERDICT_DIAGNOSIS: dict[FeedbackVerdict, CaseDiagnosis] = {
    FeedbackVerdict.INCORRECT: CaseDiagnosis.USER_REPORTED_INCORRECT,
    FeedbackVerdict.INCOMPLETE: CaseDiagnosis.USER_REPORTED_INCOMPLETE,
}

# Diagnoses that mean "the documents cannot answer this". Only these count
# against the answer rate as knowledge gaps; a capacity limit is an operational
# problem and saying otherwise would send an owner to write a document that
# would not have helped.
GAP_DIAGNOSES = frozenset(
    {CaseDiagnosis.NO_EVIDENCE_FOUND, CaseDiagnosis.EVIDENCE_OFF_TOPIC}
)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------- diagnosis


def diagnose(
    *,
    outcome: CaseOutcome,
    refusal_reason: str | None,
    retrieved_chunks: int,
    repaired_citations: int,
) -> CaseDiagnosis | None:
    """Classify one completed request, or None if there is nothing to record.

    Deterministic on purpose. An LLM classifier would be more nuanced and
    would also mean every failure costs another provider call — precisely when
    the system is already struggling — and would put diagnosis on the same
    rate limit as answering. A burst of hard questions must not make the
    thing that explains the burst the next thing to fail.

    Returns None for a clean answer: recording every success would turn the
    casebook into a query log, and the gap report into something nobody scans.
    """
    if outcome == CaseOutcome.ANSWERED:
        # Shipped, so not a failure — but citations that needed re-pointing
        # are a real attribution signal and the only way anyone learns the
        # corpus has near-duplicate passages.
        if repaired_citations > 0:
            return CaseDiagnosis.CITATION_MISATTRIBUTED
        return None

    if refusal_reason == "BUDGET_EXHAUSTED":
        return CaseDiagnosis.CAPACITY_EXCEEDED
    if retrieved_chunks == 0:
        # Nothing in scope matched at all — the clearest possible statement
        # that the corpus does not cover this.
        return CaseDiagnosis.NO_EVIDENCE_FOUND
    if refusal_reason == "UNVERIFIABLE_ANSWER":
        # Evidence was there and a draft existed; the failure is ours.
        return CaseDiagnosis.ANSWER_UNVERIFIABLE
    return CaseDiagnosis.EVIDENCE_OFF_TOPIC


# ------------------------------------------------------------- persistence


def _local_path() -> Path:
    return Path(get_settings().index_dir) / _LOCAL_FILE


def _local_read_all() -> list[dict[str, Any]]:
    path = _local_path()
    if not path.exists():
        return []
    try:
        data: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return data
    except (json.JSONDecodeError, OSError):
        return []  # a truncated casebook is not a reason to fail a request


def _local_write_all(rows: list[dict[str, Any]]) -> None:
    path = _local_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # a reader never observes a half-written file


def _table() -> Any:
    import boto3

    settings = get_settings()
    return boto3.resource("dynamodb", region_name=settings.aws_region).Table(
        settings.ddb_table_documents
    )


def record(case: Case) -> None:
    """Persist one case. Never raises.

    A case is a byproduct of answering a question. Letting its write fail the
    request would mean a user who asked something hard loses their answer
    BECAUSE it was hard — the worst possible failure mode for a feature whose
    entire purpose is to make hard questions better over time.
    """
    try:
        if get_settings().local_mode:
            with _local_lock:
                rows = _local_read_all()
                rows.append(case.model_dump(mode="json"))
                _local_write_all(rows)
            return
        item = case.model_dump(mode="json")
        item["pk"] = f"{_PK_PREFIX}{case.owner_id}"
        item["sk"] = f"{case.created_at}#{case.case_id}"
        # Matches the traces table's retention. Harmless if TTL is not enabled
        # on the table: DynamoDB simply stores the attribute.
        item["ttl"] = int(time.time()) + 90 * 86400
        _table().put_item(Item=item)
    except Exception as exc:  # noqa: BLE001 — see docstring
        _logger.warning("casebook_write_failed", extra={"data": f"{type(exc).__name__}: {exc}"})


def feedback_case_id(trace_id: str) -> str:
    """The one case id a trace's feedback can ever have.

    Deterministic, not random, because it is what makes feedback idempotent:
    one reader, one answer, one verdict. The first click wins and every later
    one is a no-op -- see record_feedback().
    """
    return f"fb-{trace_id}"


def _claims_path() -> Path:
    return Path(get_settings().index_dir) / _CLAIMS_FILE


def record_feedback(
    *,
    owner_id: str,
    trace_id: str,
    verdict: FeedbackVerdict,
    day: str,
    case: Case | None,
) -> bool:
    """Record one reader's verdict on one answer, exactly once, atomically.

    Returns True if this call recorded it, False if this reader had already
    rated this answer (any verdict). Three writes happen together or not at
    all:

      1. a claim on (owner, trace) -- the idempotency key, for EVERY verdict
      2. the case, for INCORRECT / INCOMPLETE (HELPFUL opens none)
      3. the daily counters: rated, and helpful or wrong

    Why atomic rather than three sequential writes, each individually safe:
    sequencing them leaves two failures that look harmless and are not.
    Claim-then-case can lose the report -- the claim lands, the case write
    fails, the reader retries, and the claim now says "already rated" while
    the case that justified it does not exist. And keying dedupe on the case
    alone leaves HELPFUL unguarded, since it opens no case: a reader who
    clicks helpful and then wrong is counted twice, and the one KPI a manager
    reads from this feature (how often readers flag answers) drifts upward
    for reasons that have nothing to do with answer quality.

    In DynamoDB this is one TransactWriteItems. It costs twice the write
    units of a plain put, which on this table's volume is nothing, and it is
    the only way to make "claim if absent AND write the rest" a single
    decision two concurrent Lambdas cannot both win.

    RAISES on storage failure, unlike record(). record() is a byproduct of
    answering and must not cost the answer; here the write IS the request,
    and telling a reader "thanks, noted" while their report was dropped means
    the next reader meets the same wrong answer and nobody knows why.
    """
    field = "helpful" if verdict == FeedbackVerdict.HELPFUL else "wrong"

    if get_settings().local_mode:
        with _local_lock:
            claims_path = _claims_path()
            claims: dict[str, str] = {}
            if claims_path.exists():
                try:
                    claims = json.loads(claims_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    claims = {}
            key = f"{owner_id}#{trace_id}"
            if key in claims:
                return False

            # Build every new state first, then publish. Each file write is
            # atomic on its own (tmp + os.replace); the lock is what makes the
            # three appear together to any other request in this process.
            rows = _local_read_all()
            if case is not None:
                rows.append(case.model_dump(mode="json"))

            stats_path = _stats_path()
            stats: dict[str, dict[str, int]] = {}
            if stats_path.exists():
                try:
                    stats = json.loads(stats_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    stats = {}
            row = stats.setdefault(f"{owner_id}#{day}", {"answered": 0, "refused": 0})
            row["rated"] = row.get("rated", 0) + 1
            row[field] = row.get(field, 0) + 1

            claims[key] = verdict.value

            if case is not None:
                _local_write_all(rows)
            _atomic_json_write(stats_path, stats)
            # The claim goes LAST. If anything above raised, no claim exists,
            # so a retry records normally instead of reporting "already rated"
            # for a verdict that was never stored.
            _atomic_json_write(claims_path, claims)
        return True

    import boto3
    from boto3.dynamodb.types import TypeSerializer
    from botocore.exceptions import ClientError

    settings = get_settings()
    table_name = settings.ddb_table_documents
    ser = TypeSerializer()

    def typed(item: dict[str, Any]) -> dict[str, Any]:
        return {k: ser.serialize(v) for k, v in item.items() if v is not None}

    ttl = int(time.time()) + 90 * 86400
    ops: list[dict[str, Any]] = [
        {
            "Put": {
                "TableName": table_name,
                "Item": typed({
                    "pk": f"{_CLAIM_PK_PREFIX}{owner_id}",
                    "sk": trace_id,
                    "verdict": verdict.value,
                    "created_at": utc_now(),
                    "ttl": ttl,
                }),
                "ConditionExpression": "attribute_not_exists(pk)",
            }
        },
        {
            "Update": {
                "TableName": table_name,
                "Key": typed({"pk": f"{_STATS_PK_PREFIX}{owner_id}", "sk": day}),
                "UpdateExpression": f"ADD rated :one, {field} :one",
                "ExpressionAttributeValues": {":one": {"N": "1"}},
            }
        },
    ]
    if case is not None:
        item = case.model_dump(mode="json")
        item["pk"] = f"{_PK_PREFIX}{case.owner_id}"
        item["sk"] = f"{case.created_at}#{case.case_id}"
        item["ttl"] = ttl
        ops.append({"Put": {"TableName": table_name, "Item": typed(item)}})

    client = boto3.client("dynamodb", region_name=settings.aws_region)
    try:
        client.transact_write_items(TransactItems=ops)
    except ClientError as exc:
        err = exc.response.get("Error", {})
        reasons = exc.response.get("CancellationReasons") or []
        # The claim is op 0. If IT is the one that failed its condition, this
        # reader already rated this answer; any other cancellation is a real
        # failure and must surface.
        if err.get("Code") == "TransactionCanceledException" and (
            reasons and reasons[0].get("Code") == "ConditionalCheckFailed"
        ):
            return False
        raise
    return True


def _atomic_json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def list_for_owner(owner_id: str, *, since: str | None = None) -> list[Case]:
    """Cases for one owner, oldest first. Never raises."""
    try:
        if get_settings().local_mode:
            with _local_lock:
                rows = [r for r in _local_read_all() if r.get("owner_id") == owner_id]
        else:
            from boto3.dynamodb.conditions import Key

            # Key condition only — the sort key starts with the timestamp, so
            # the time window is part of the query rather than a filter applied
            # after reading (and paying for) everything.
            cond = Key("pk").eq(f"{_PK_PREFIX}{owner_id}")
            if since:
                cond = cond & Key("sk").gte(since)
            rows = _table().query(KeyConditionExpression=cond).get("Items", [])
        cases = [Case(**{k: v for k, v in r.items() if k in Case.model_fields}) for r in rows]
    except Exception as exc:  # noqa: BLE001
        _logger.warning("casebook_read_failed", extra={"data": f"{type(exc).__name__}: {exc}"})
        return []
    if since:
        cases = [c for c in cases if c.created_at >= since]
    return sorted(cases, key=lambda c: c.created_at)


def reset_local() -> None:
    """Test helper: drop the local casebook."""
    if not get_settings().local_mode:
        raise RuntimeError("reset_local is local_mode only")
    with _local_lock:
        for path in (_local_path(), _claims_path()):
            if path.exists():
                path.unlink()


# -------------------------------------------------------------- daily totals
#
# The casebook records failures only — recording every success would turn it
# into a query log nobody scans, and a gap report is read for what is MISSING.
# But "answer rate" is the one number a manager acts on, and a rate needs a
# denominator that failures alone cannot provide.
#
# So volume is kept separately as a per-owner, per-day counter: two integers a
# day, incremented atomically, rather than a row per question. In AWS that is
# an UpdateItem with ADD, which is safe under concurrency without a read (two
# Lambdas answering at once both count) and is the same order of cost as the
# trace write already made on every request.


def _stats_path() -> Path:
    return Path(get_settings().index_dir) / _STATS_FILE


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def bump_totals(owner_id: str, outcome: CaseOutcome) -> None:
    """Count one completed request. Never raises — see record()."""
    field = "answered" if outcome == CaseOutcome.ANSWERED else "refused"
    try:
        if get_settings().local_mode:
            with _local_lock:
                path = _stats_path()
                data: dict[str, dict[str, int]] = {}
                if path.exists():
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        data = {}
                key = f"{owner_id}#{_today()}"
                row = data.setdefault(key, {"answered": 0, "refused": 0})
                row[field] = row.get(field, 0) + 1
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                os.replace(tmp, path)
            return
        _table().update_item(
            Key={"pk": f"{_STATS_PK_PREFIX}{owner_id}", "sk": _today()},
            # ADD, not SET: an atomic increment needs no prior read, so two
            # instances answering at the same moment cannot lose a count.
            UpdateExpression=f"ADD {field} :one",
            ExpressionAttributeValues={":one": 1},
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning("casebook_totals_failed", extra={"data": f"{type(exc).__name__}: {exc}"})


def _stats_rows(owner_id: str, since_date: str) -> list[dict[str, Any]]:
    """Daily counter rows for one owner since YYYY-MM-DD. Never raises."""
    try:
        if get_settings().local_mode:
            path = _stats_path()
            if not path.exists():
                return []
            with _local_lock:
                data = json.loads(path.read_text(encoding="utf-8"))
            return [
                v
                for k, v in data.items()
                if k.startswith(f"{owner_id}#") and k.split("#", 1)[1] >= since_date
            ]
        from boto3.dynamodb.conditions import Key

        rows: list[dict[str, Any]] = (
            _table()
            .query(
                KeyConditionExpression=Key("pk").eq(f"{_STATS_PK_PREFIX}{owner_id}")
                & Key("sk").gte(since_date)
            )
            .get("Items", [])
        )
        return rows
    except Exception as exc:  # noqa: BLE001
        _logger.warning(
            "casebook_totals_read_failed", extra={"data": f"{type(exc).__name__}: {exc}"}
        )
        return []


def _sum(rows: list[dict[str, Any]], field: str) -> int:
    return sum(int(r.get(field, 0)) for r in rows)


def totals_for_owner(owner_id: str, *, since_date: str) -> tuple[int, int]:
    """(answered, refused) since the given YYYY-MM-DD. Never raises."""
    rows = _stats_rows(owner_id, since_date)
    return (_sum(rows, "answered"), _sum(rows, "refused"))


def feedback_totals_for_owner(owner_id: str, *, since_date: str) -> tuple[int, int, int]:
    """(rated, helpful, wrong) since the given YYYY-MM-DD. Never raises."""
    rows = _stats_rows(owner_id, since_date)
    return (_sum(rows, "rated"), _sum(rows, "helpful"), _sum(rows, "wrong"))


# ------------------------------------------------------------ gap clustering

_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_STOPWORDS = frozenset({
    "and", "are", "can", "does", "for", "from", "has", "have", "how", "the",
    "there", "this", "that", "was", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your", "our", "any", "all", "get",
    "got", "its", "his", "her", "not", "but", "she", "him", "they", "them",
    "need", "want", "should", "would", "could", "about", "after", "before",
})

# Fraction of shared terms that makes two questions "the same question asked
# differently". Tuned to be generous: over-merging produces a slightly vague
# topic line, while under-merging produces the flat list of near-duplicates
# this clustering exists to avoid, and the example questions are shown anyway
# so an owner can always see what was actually asked.
_SIMILARITY = 0.4


def _terms(question: str) -> frozenset[str]:
    return frozenset(
        w.lower() for w in _WORD_RE.findall(question) if w.lower() not in _STOPWORDS
    )


def _similar(a: frozenset[str], b: frozenset[str]) -> bool:
    if not a or not b:
        return False
    # Overlap against the SMALLER question, not the union: "refund window?"
    # and "what is the refund window for annual plans bought in the EU?" are
    # the same gap, and Jaccard would score them far apart purely because one
    # is longer.
    return len(a & b) / min(len(a), len(b)) >= _SIMILARITY


def _cluster(cases: list[Case]) -> list[list[Case]]:
    """Greedy single-pass clustering on shared content words.

    Term overlap rather than embeddings, for two reasons. It is free and
    stateless, so a report costs no model time and cannot fail on a provider
    outage. And it is explainable: an owner looking at a cluster can see the
    words the questions share, which matters more for a report someone acts on
    than marginal accuracy would. If clusters get noisy at real volume,
    embedding the questions is the natural upgrade — the interface here does
    not change.
    """
    clusters: list[list[Case]] = []
    signatures: list[frozenset[str]] = []
    for case in cases:
        terms = _terms(case.question)
        for i, sig in enumerate(signatures):
            if _similar(terms, sig):
                clusters[i].append(case)
                signatures[i] = sig | terms
                break
        else:
            clusters.append([case])
            signatures.append(terms)
    return clusters


def knowledge_gaps(owner_id: str, *, window_days: int = 30) -> KnowledgeGapReport:
    """The document owner's work list: what people asked that the corpus could
    not answer, grouped by topic and ranked by how often it came up.

    Ranked by frequency because that is the owner's actual prioritisation
    question — one missing paragraph asked about eleven times is worth more
    than eleven one-off questions — and recency breaks ties so a gap that has
    stopped mattering sinks.
    """
    cutoff = datetime.now(UTC).timestamp() - window_days * 86400
    since = datetime.fromtimestamp(cutoff, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    cases = list_for_owner(owner_id, since=since)

    # From the daily rollup, not from the cases: the casebook holds failures
    # only, so counting it would report an answer rate near zero and make a
    # healthy system look broken.
    # One read serves both the answer counters and the feedback counters --
    # they share a row per day.
    stats = _stats_rows(owner_id, since[:10])
    answered, unanswered = _sum(stats, "answered"), _sum(stats, "refused")
    total = answered + unanswered

    gaps: list[KnowledgeGap] = []
    for diagnosis in CaseDiagnosis:
        # Reader-reported problems sit beside the gaps: an answer people say is
        # wrong is at least as urgent for an owner as a question nobody could
        # answer, and arguably more so -- a refusal wastes a reader's time, a
        # wrong answer gets acted on.
        if diagnosis not in GAP_DIAGNOSES and diagnosis not in FEEDBACK_DIAGNOSES:
            continue
        for cluster in _cluster([c for c in cases if c.diagnosis == diagnosis]):
            # The longest question is the most specific one, and therefore the
            # most useful label for someone deciding what to write.
            topic = max((c.question for c in cluster), key=len)
            gaps.append(
                KnowledgeGap(
                    topic=topic,
                    question_count=len(cluster),
                    example_questions=sorted({c.question for c in cluster})[:5],
                    diagnosis=diagnosis,
                    recommended_action=RECOMMENDED_ACTION[diagnosis],
                    first_seen=min(c.created_at for c in cluster),
                    last_seen=max(c.created_at for c in cluster),
                )
            )
    # Most-asked first, and among equally frequent gaps the most recent first:
    # that is the order an owner would pick work in. Sorted in two stable
    # passes because the two keys run in opposite directions and a timestamp
    # cannot be negated.
    gaps.sort(key=lambda g: g.last_seen, reverse=True)
    gaps.sort(key=lambda g: g.question_count, reverse=True)

    return KnowledgeGapReport(
        generated_at=utc_now(),
        window_days=window_days,
        total_questions=total,
        answered=answered,
        unanswered=unanswered,
        # Reported over recorded cases only, and stated as such by the field
        # names: this is the rate among questions the system had something to
        # say about, not a claim about every query ever made.
        answer_rate=round(answered / total, 4) if total else 1.0,
        answers_rated=_sum(stats, "rated"),
        marked_helpful=_sum(stats, "helpful"),
        marked_wrong=_sum(stats, "wrong"),
        gaps=gaps,
    )


def new_case_id() -> str:
    return uuid.uuid4().hex
