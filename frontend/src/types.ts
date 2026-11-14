// Mirrors backend/app/models/schemas.py — change both in the same commit.

export interface Citation {
  chunk_id: string;
  section_path: string;
  quote: string;
  page_number: number | null;
  source_filename: string;
  document_id: string;
  provenance_hash?: string;
  char_start?: number | null;
  char_end?: number | null;
}

// Mirrors DocumentStatus / DocumentErrorCode in schemas.py.
export type DocumentStatus =
  | "UPLOADING"
  | "UPLOADED"
  | "PROCESSING"
  | "INDEXED"
  | "FAILED"
  | "DELETED";

export interface DocumentSummary {
  document_id: string;
  filename: string;
  status: DocumentStatus;
  file_size: number;
  page_count: number | null;
  chunk_count: number;
  created_at: string;
  updated_at: string;
  error_code: string | null;
  error_message: string | null;
  classification?: ClassificationLevel;
  allowed_roles?: string[];
  owned_by_caller?: boolean;
  retryable?: boolean;
  ingest_attempts?: number;
}

// Human-readable explanations for ingestion failures. The backend returns a
// machine-readable code precisely so the UI can say what to do about it
// instead of showing a generic "upload failed".
export const DOCUMENT_ERROR_HELP: Record<string, string> = {
  ENCRYPTED_DOCUMENT:
    "This PDF is password-protected. Upload a copy without a password.",
  UNSUPPORTED_SCANNED_DOCUMENT:
    "This looks like a scanned/image-only PDF. Sentinel does not run OCR — upload a text-based PDF.",
  CORRUPT_DOCUMENT: "This file could not be opened as a PDF. It may be damaged.",
  EMPTY_DOCUMENT: "This document has no pages.",
  NO_EXTRACTABLE_TEXT:
    "Almost no text could be extracted, so this document cannot be answered from reliably.",
  TOO_MANY_PAGES: "This document has more pages than Sentinel will index.",
  UPLOAD_ABANDONED:
    "This upload never finished, so the file was never received. Upload it again.",
  TOO_LARGE: "This document is over the 1 MB size limit.",
  UNSUPPORTED_CONTENT_TYPE: "Only .pdf, .md and .txt files are supported.",
  INTERNAL_ERROR: "Indexing failed unexpectedly. Please try again.",
  PUBLISH_CONTENTION: "The index was busy with other uploads. The file is saved; retry it.",
};

export interface CriticScores {
  faithfulness: number;
  relevance: number;
}

export interface ConversationTurn {
  role: string;
  content: string;
}

export interface QueryResponse {
  trace_id: string;
  answer: string;
  citations: Citation[];
  critic: CriticScores;
  repair_count: number;
  model_used: string;
  tokens: { in: number; out: number };
  latency_ms: number;
}

// Mirrors RefusalReason in schemas.py. A refusal is a successful outcome, but
// these are not interchangeable: BUDGET_EXHAUSTED is retryable and says
// nothing about whether the documents cover the question, so telling the user
// "not in your documents" for it would send them to fix the wrong thing.
export type RefusalReason =
  | "INSUFFICIENT_EVIDENCE"
  | "UNVERIFIABLE_ANSWER"
  | "BUDGET_EXHAUSTED";

export interface RefusalResponse {
  trace_id: string;
  refusal: true;
  reason: string;
  reason_code?: RefusalReason; // optional: older backends don't send it
  best_effort_context: string[];
}

export const REFUSAL_COPY: Record<RefusalReason, { title: string; body: string; hint: string }> = {
  INSUFFICIENT_EVIDENCE: {
    title: "No supporting evidence found",
    body: "I couldn't find information in your documents that answers this question.",
    hint: "Try rephrasing, or upload a document that covers this topic.",
  },
  UNVERIFIABLE_ANSWER: {
    title: "Couldn't verify an answer",
    body:
      "I found related material, but couldn't confirm an answer was fully supported by it — " +
      "so I'm not going to guess.",
    hint: "Try asking something more specific about the document.",
  },
  BUDGET_EXHAUSTED: {
    title: "Ran out of time on this one",
    body: "This question hit the processing limit before a verified answer was ready.",
    hint: "Please try again — this is usually temporary.",
  },
};

export interface TraceStep {
  name: string;
  started_ms: number;
  duration_ms: number;
  tokens_in: number;
  tokens_out: number;
  meta: Record<string, string>;
}

export interface TraceRecord {
  trace_id: string;
  user_id: string;
  created_at?: string;
  query_redacted?: string;
  provider?: string;
  est_cost_usd?: number;
  cache_hit?: boolean;
  steps: TraceStep[];
  critic_scores: { faithfulness: number; relevance: number; attempt: number }[];
  repair_count: number;
  final_status: string;
  total_tokens: number;
  latency_ms: number;
  model_path: string[];
}

export type QueryResult = QueryResponse | RefusalResponse;

export function isRefusal(r: QueryResult): r is RefusalResponse {
  return (r as RefusalResponse).refusal === true;
}

// ---------------------------------------------------------------- chat UI
//
// Presentation-layer shape for one turn in the visible thread. Built purely
// from QueryResult/RefusalResponse — no new backend fields. Keeping this
// distinct from QueryResult means the chat renderer never has to branch on
// `isRefusal` again once a message is constructed.

export type ChatMessage =
  | { id: string; role: "user"; text: string }
  | {
      id: string;
      role: "assistant";
      kind: "answer";
      text: string;
      citations: Citation[];
      critic: CriticScores;
      repairCount: number;
      model: string;
      latencyMs: number;
      traceId: string;
    }
  | {
      id: string;
      role: "assistant";
      kind: "refusal";
      reason: string;
      reasonCode: RefusalReason;
      traceId: string;
    }
  | { id: string; role: "assistant"; kind: "pending"; stages: StageEvent[] }
  | { id: string; role: "assistant"; kind: "error"; message: string };

// ------------------------------------------------------- coverage insights
//
// Mirrors CaseDiagnosis / KnowledgeGap / KnowledgeGapReport in schemas.py.

export type CaseDiagnosis =
  | "NO_EVIDENCE_FOUND"
  | "EVIDENCE_OFF_TOPIC"
  | "ANSWER_UNVERIFIABLE"
  | "CITATION_MISATTRIBUTED"
  | "CAPACITY_EXCEEDED"
  | "USER_REPORTED_INCORRECT"
  | "USER_REPORTED_INCOMPLETE";

// Plain-language labels. The report is read by whoever owns the documents,
// who has no reason to know what "EVIDENCE_OFF_TOPIC" means — and a report
// that needs a glossary is one nobody acts on.
export const DIAGNOSIS_LABEL: Record<CaseDiagnosis, string> = {
  NO_EVIDENCE_FOUND: "Nothing covers this",
  EVIDENCE_OFF_TOPIC: "Covered, but not this case",
  ANSWER_UNVERIFIABLE: "Evidence unclear",
  CITATION_MISATTRIBUTED: "Overlapping sources",
  CAPACITY_EXCEEDED: "System limit",
  USER_REPORTED_INCORRECT: "Readers say it's wrong",
  USER_REPORTED_INCOMPLETE: "Readers say it's partial",
};

export interface KnowledgeGap {
  topic: string;
  question_count: number;
  example_questions: string[];
  diagnosis: CaseDiagnosis;
  recommended_action: string;
  first_seen: string;
  last_seen: string;
}

export interface KnowledgeGapReport {
  generated_at: string;
  window_days: number;
  total_questions: number;
  answered: number;
  unanswered: number;
  answer_rate: number;
  // Raw counts, not a rate: the sample is small and self-selected (people
  // rate answers that surprised them), so a percentage would claim a
  // precision it doesn't have. See KnowledgeGapReport in schemas.py.
  answers_rated: number;
  marked_helpful: number;
  marked_wrong: number;
  gaps: KnowledgeGap[];
}

// ----------------------------------------------------------------- feedback
//
// Mirrors FeedbackVerdict / FeedbackResponse in schemas.py.

export type FeedbackVerdict = "HELPFUL" | "INCORRECT" | "INCOMPLETE";

export interface FeedbackResponse {
  trace_id: string;
  verdict: FeedbackVerdict;
  // false when this reader already rated this answer — the first verdict
  // stands, and the UI should say so rather than claim a second one landed.
  recorded: boolean;
  case_id: string | null;
}

// ------------------------------------------------------- access + provenance
//
// Mirrors ClassificationLevel / ChunkDetail in schemas.py.

export type ClassificationLevel = "PUBLIC" | "INTERNAL" | "CONFIDENTIAL" | "RESTRICTED";
export const CLASSIFICATION_LEVELS: ClassificationLevel[] = [
  "PUBLIC",
  "INTERNAL",
  "CONFIDENTIAL",
  "RESTRICTED",
];

export interface ChunkDetail {
  chunk_id: string;
  document_id: string;
  source_filename: string;
  section_path: string;
  page_number: number | null;
  char_start: number;
  char_end: number;
  text: string;
  provenance_hash: string;
  document_checksum_sha256: string;
  hash_recipe: string;
  classification: ClassificationLevel;
}

// ------------------------------------------------------------------ health

export interface Health {
  status: string;
  index_version: string;
  model_warm: boolean;
  storage_backend: string;
  llm_provider: string;
  llm_configured: boolean;
  demo_mode: boolean;
  version: string;
}

// --------------------------------------------------------------- streaming
//
// The stage events POST /v1/query/stream sends. They carry telemetry only —
// never answer text — until the final `result` event (ADR 0005 / 0010).

export interface StageEvent {
  node: string;
  label: string;
  attempt: number;
  step: TraceStep | null;
}

// --------------------------------------------------------------- telemetry

export interface StageStat {
  name: string;
  calls: number;
  mean_ms: number;
  p95_ms: number;
  tokens: number;
}

export interface HourBucket {
  hour: string;
  requests: number;
  refused: number;
  p95_ms: number;
}

export interface TelemetryReport {
  generated_at: string;
  window_hours: number;
  requests: number;
  answered: number;
  refused: number;
  errors: number;
  refusal_rate: number;
  cache_hit_rate: number;
  repair_rate: number;
  downgrades: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  tokens_total: number;
  tokens_per_request: number;
  token_burn_per_hour: number;
  est_cost_usd: number;
  est_cost_per_answer_usd: number;
  providers: Record<string, number>;
  refusal_reasons: Record<string, number>;
  stages: StageStat[];
  hourly: HourBucket[];
}

export interface TraceSummary {
  trace_id: string;
  created_at: string;
  user_id: string;
  query_redacted: string;
  final_status: string;
  latency_ms: number;
  total_tokens: number;
  est_cost_usd: number;
  repair_count: number;
  cache_hit: boolean;
}

// ----------------------------------------------------------------- ratchet

export interface RatchetCandidate {
  case_id: string;
  question: string;
  diagnosis: CaseDiagnosis;
  outcome: "ANSWERED" | "REFUSED";
  correction: string | null;
  asked_at: string;
  owner_id: string;
  promote_command: string;
}

export interface RatchetItem {
  id: string;
  question: string;
  expect: "answer" | "refuse";
  status: "pending" | "enforced";
  reference_answer: string | null;
  note: string | null;
  diagnosis: string;
  promoted_at: string;
  locked_at: string | null;
}

export interface RatchetView {
  enforced: number;
  pending: number;
  retired: number;
  items: RatchetItem[];
  candidates: RatchetCandidate[];
  writable: boolean;
  verify_command: string;
}

// -------------------------------------------------------------- chaos demo

export type ChaosScenario =
  | "prompt_injection"
  | "pii_leak"
  | "hallucination_trap"
  | "out_of_scope"
  | "budget_exceeded";

export const CHAOS_SCENARIOS: ChaosScenario[] = [
  "prompt_injection",
  "pii_leak",
  "hallucination_trap",
  "out_of_scope",
  "budget_exceeded",
];

export interface ChaosResult {
  scenario: ChaosScenario;
  title: string;
  attack: string;
  caught_by: string;
  outcome: string;
  passed: boolean;
  http_status: number;
  detail: string;
  trace_id: string | null;
  refusal_reason: RefusalReason | null;
  evidence: Record<string, string>;
  steps: TraceStep[];
  adrs: string[];
  simulated: boolean;
}
