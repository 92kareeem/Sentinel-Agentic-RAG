// Typed fetch client. VITE_API_BASE / VITE_API_KEY are baked at build time —
// the key here is the DEMO key only (quota-limited server-side, rotatable).
// The admin key must never appear in a deployed bundle.
//
// In local storage mode the server ships five fixture identities (two
// tenants, four roles) so access control can be demonstrated from a browser.
// The "acting as" switcher swaps between them; it exists only against a
// local API and is never offered against a deployed one.

import { feedbackBody } from "./lib/feedback";
import { SseParser } from "./lib/sse";
import type {
  ChaosResult,
  ChaosScenario,
  ChunkDetail,
  ClassificationLevel,
  ConversationTurn,
  DocumentSummary,
  FeedbackResponse,
  FeedbackVerdict,
  Health,
  KnowledgeGapReport,
  QueryResult,
  RatchetItem,
  RatchetView,
  StageEvent,
  TelemetryReport,
  TraceRecord,
  TraceSummary,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";
const BUILD_KEY = (import.meta.env.VITE_API_KEY as string | undefined) ?? "demo-local";

export const IS_LOCAL_API = /^https?:\/\/(localhost|127\.0\.0\.1)/.test(BASE);

// Mirrors _LOCAL_USERS in backend/app/guardrails/auth.py.
export const LOCAL_IDENTITIES = [
  { key: "demo-local", label: "Demo employee", detail: "acme · employee · INTERNAL" },
  { key: "finance-local", label: "Finance analyst", detail: "acme · finance · CONFIDENTIAL" },
  { key: "contractor-local", label: "Contractor", detail: "acme · contractor · PUBLIC" },
  { key: "admin-local", label: "Tenant admin", detail: "acme · admin · RESTRICTED" },
  { key: "globex-local", label: "Other company", detail: "globex · employee · INTERNAL" },
] as const;

const KEY_STORAGE = "sentinel-api-key";

function storedKey(): string | null {
  try {
    return localStorage.getItem(KEY_STORAGE);
  } catch {
    return null;
  }
}

let currentKey = (IS_LOCAL_API && storedKey()) || BUILD_KEY;

export function apiKey(): string {
  return currentKey;
}

export function setApiKey(key: string): void {
  currentKey = key;
  try {
    localStorage.setItem(KEY_STORAGE, key);
  } catch {
    // Non-fatal: the choice just won't survive a reload.
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
    public retryAfter?: string,
    // Set for ingestion rejections, which return a machine-readable reason so
    // the UI can explain what to do rather than showing a generic failure.
    public errorCode?: string,
  ) {
    super(detail);
  }
}

// FastAPI puts our structured ingestion errors in `detail`; everything else
// sends a plain string. Normalizes both into (message, code).
function parseDetail(body: unknown, fallback: string): { message: string; code?: string } {
  const problem = body as { detail?: string; title?: string; error_code?: string } | null;
  return {
    message: problem?.detail ?? problem?.title ?? fallback,
    code: problem?.error_code ?? undefined,
  };
}

async function failure(resp: Response, fallback: string): Promise<ApiError> {
  const body = await resp.json().catch(() => ({ detail: resp.statusText }));
  const { message, code } = parseDetail(body, fallback);
  return new ApiError(resp.status, message, resp.headers.get("retry-after") ?? undefined, code);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      "x-api-key": currentKey,
      ...init?.headers,
    },
  });
  if (!resp.ok) throw await failure(resp, "request failed");
  return resp.json() as Promise<T>;
}

function queryBody(query: string, docId?: string | null, history?: ConversationTurn[]): string {
  return JSON.stringify(
    docId
      ? { query, doc_id: docId, conversation_history: history ?? [] }
      : { query, conversation_history: history ?? [] },
  );
}

export function postQuery(
  query: string,
  docId?: string | null,
  conversationHistory?: ConversationTurn[],
): Promise<QueryResult> {
  return request<QueryResult>("/v1/query", {
    method: "POST",
    body: queryBody(query, docId, conversationHistory),
  });
}

/**
 * Ask over Server-Sent Events. `onStage` fires once per completed graph node;
 * the answer arrives only in the resolved result, after it has been verified.
 * Behind a proxy that buffers (AWS Lambda Function URLs do), every event lands
 * at once at the end — the result is identical, only the progress is lost.
 */
export async function streamQuery(
  query: string,
  docId: string | null,
  conversationHistory: ConversationTurn[],
  onStage: (stage: StageEvent) => void,
): Promise<QueryResult> {
  const resp = await fetch(`${BASE}/v1/query/stream`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": currentKey, accept: "text/event-stream" },
    body: queryBody(query, docId, conversationHistory),
  });
  if (!resp.ok) throw await failure(resp, "request failed");
  if (!resp.body) return postQuery(query, docId, conversationHistory);

  const parser = new SseParser();
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let result: QueryResult | null = null;

  const handle = (event: string, data: string) => {
    if (event === "stage") onStage(JSON.parse(data) as StageEvent);
    else if (event === "result") result = JSON.parse(data) as QueryResult;
    else if (event === "error") {
      const e = JSON.parse(data) as { status: number; detail: string };
      throw new ApiError(e.status, e.detail);
    }
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    for (const ev of parser.push(decoder.decode(value, { stream: true }))) handle(ev.event, ev.data);
  }
  for (const ev of parser.flush()) handle(ev.event, ev.data);
  if (!result) throw new ApiError(502, "The answer stream ended early. Please try again.");
  return result;
}

export function getTrace(traceId: string): Promise<TraceRecord> {
  return request<TraceRecord>(`/v1/traces/${traceId}`);
}

export function getHealth(): Promise<Health> {
  return request<Health>("/healthz");
}

// ---------------------------------------------------------------- uploads

interface UploadTicket {
  doc_id: string;
  upload_url: string;
  fields: Record<string, string>;
  filename: string;
  max_bytes: number;
  expires_in_seconds: number;
}

interface IndexResult {
  doc_id: string;
  chunks_indexed: number;
  index_version: string;
}

export interface UploadOptions {
  classification: ClassificationLevel;
  roles: string[];
}

function accessParams(opts: UploadOptions): string {
  const p = new URLSearchParams({ classification: opts.classification });
  if (opts.roles.length) p.set("roles", opts.roles.join(","));
  return p.toString();
}

// Three-step upload (production/AWS): get a signed S3 form, POST the file
// straight to S3 (no API key — different origin), then ask the API to index it.
async function uploadViaS3(file: File, opts: UploadOptions, onStage?: (stage: string) => void): Promise<IndexResult> {
  onStage?.("Requesting upload…");
  const ticket = await request<UploadTicket>(
    `/v1/documents?filename=${encodeURIComponent(file.name)}&${accessParams(opts)}`,
    { method: "POST" },
  );
  if (file.size > ticket.max_bytes) {
    throw new ApiError(413, `File too large — max ${(ticket.max_bytes / 1e6).toFixed(0)} MB`);
  }

  onStage?.("Uploading to S3…");
  const form = new FormData();
  for (const [k, v] of Object.entries(ticket.fields)) form.append(k, v);
  form.append("file", file);
  const s3resp = await fetch(ticket.upload_url, { method: "POST", body: form });
  if (!s3resp.ok) throw new ApiError(s3resp.status, "S3 upload rejected the file");

  onStage?.("Indexing (chunk + embed)…");
  return request<IndexResult>(`/v1/documents/${ticket.doc_id}/index`, { method: "POST" });
}

// Single-step upload (local dev and docker compose): the API chunks + embeds directly.
async function uploadLocal(file: File, opts: UploadOptions, onStage?: (stage: string) => void): Promise<IndexResult> {
  onStage?.("Uploading and indexing (chunk + embed)…");
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${BASE}/v1/documents/local-upload?${accessParams(opts)}`, {
    method: "POST",
    headers: { "x-api-key": currentKey },
    body: form,
  });
  if (!resp.ok) throw await failure(resp, "upload failed");
  return resp.json() as Promise<IndexResult>;
}

export function uploadDocument(
  file: File,
  opts: UploadOptions,
  onStage?: (stage: string) => void,
): Promise<IndexResult> {
  return IS_LOCAL_API ? uploadLocal(file, opts, onStage) : uploadViaS3(file, opts, onStage);
}

// ---------------------------------------------------------------- documents

export function listDocuments(): Promise<DocumentSummary[]> {
  return request<DocumentSummary[]>("/v1/documents");
}

export function getDocument(documentId: string): Promise<DocumentSummary> {
  return request<DocumentSummary>(`/v1/documents/${documentId}`);
}

export function deleteDocument(documentId: string): Promise<DocumentSummary> {
  return request<DocumentSummary>(`/v1/documents/${documentId}`, { method: "DELETE" });
}

export function retryDocument(documentId: string): Promise<IndexResult> {
  return request<IndexResult>(`/v1/documents/${documentId}/retry`, { method: "POST" });
}

export function getChunk(documentId: string, chunkId: string): Promise<ChunkDetail> {
  return request<ChunkDetail>(
    `/v1/documents/${encodeURIComponent(documentId)}/chunks/${encodeURIComponent(chunkId)}`,
  );
}

// ---------------------------------------------------------------- control plane

export function getKnowledgeGaps(windowDays = 30): Promise<KnowledgeGapReport> {
  return request<KnowledgeGapReport>(`/v1/insights/knowledge-gaps?window_days=${windowDays}`);
}

export function getTelemetry(windowHours = 24): Promise<TelemetryReport> {
  return request<TelemetryReport>(`/v1/insights/telemetry?window_hours=${windowHours}`);
}

export function getRecentTraces(limit = 25): Promise<TraceSummary[]> {
  return request<TraceSummary[]>(`/v1/insights/traces?limit=${limit}`);
}

export function getRatchet(): Promise<RatchetView> {
  return request<RatchetView>("/v1/insights/ratchet");
}

export function promoteCase(body: {
  case_id: string;
  expect: "answer" | "refuse";
  reference?: string;
  note?: string;
}): Promise<RatchetItem> {
  return request<RatchetItem>("/v1/insights/ratchet/promote", { method: "POST", body: JSON.stringify(body) });
}

export function retireRegression(id: string, reason: string): Promise<unknown> {
  return request(`/v1/insights/ratchet/${encodeURIComponent(id)}/retire`, {
    method: "POST",
    body: JSON.stringify({ reason }),
  });
}

export function runChaos(scenario: ChaosScenario): Promise<ChaosResult> {
  return request<ChaosResult>("/v1/demo/chaos", { method: "POST", body: JSON.stringify({ scenario }) });
}

// ---------------------------------------------------------------- feedback

export function postFeedback(
  traceId: string,
  verdict: FeedbackVerdict,
  correction?: string,
): Promise<FeedbackResponse> {
  return request<FeedbackResponse>("/v1/feedback", {
    method: "POST",
    body: JSON.stringify(feedbackBody(traceId, verdict, correction)),
  });
}
