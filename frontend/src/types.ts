// Mirrors backend/app/models/schemas.py — change both in the same commit.

export interface Citation {
  chunk_id: string;
  section_path: string;
  quote: string;
  page_number: number | null;
  source_filename: string;
  document_id: string;
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
  TOO_LARGE: "This document is over the 1 MB size limit.",
  UNSUPPORTED_CONTENT_TYPE: "Only .pdf, .md and .txt files are supported.",
  INTERNAL_ERROR: "Indexing failed unexpectedly. Please try again.",
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

export interface RefusalResponse {
  trace_id: string;
  refusal: true;
  reason: string;
  best_effort_context: string[];
}

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
