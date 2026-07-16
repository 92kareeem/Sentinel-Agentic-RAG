// Mirrors backend/app/models/schemas.py — change both in the same commit.

export interface Citation {
  chunk_id: string;
  section_path: string;
  quote: string;
}

export interface CriticScores {
  faithfulness: number;
  relevance: number;
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
