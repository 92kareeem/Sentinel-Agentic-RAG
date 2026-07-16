// Typed fetch client. VITE_API_BASE / VITE_API_KEY are baked at build time —
// the key here is the DEMO key only (quota-limited server-side, rotatable).
// The admin key must never appear in this codebase.

import type { QueryResult, TraceRecord } from "./types";

const BASE = import.meta.env.VITE_API_BASE as string;
const KEY = import.meta.env.VITE_API_KEY as string;

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
    public retryAfter?: string,
  ) {
    super(detail);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      "x-api-key": KEY,
      ...init?.headers,
    },
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new ApiError(
      resp.status,
      body.detail ?? body.title ?? "request failed",
      resp.headers.get("retry-after") ?? undefined,
    );
  }
  return resp.json() as Promise<T>;
}

export function postQuery(query: string, docId?: string | null): Promise<QueryResult> {
  return request<QueryResult>("/v1/query", {
    method: "POST",
    body: JSON.stringify(docId ? { query, doc_id: docId } : { query }),
  });
}

export function getTrace(traceId: string): Promise<TraceRecord> {
  return request<TraceRecord>(`/v1/traces/${traceId}`);
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

// Three-step upload: get a signed S3 form, POST the file straight to S3
// (no API key — different origin), then ask the API to index it.
export async function uploadDocument(
  file: File,
  onStage?: (stage: string) => void,
): Promise<IndexResult> {
  onStage?.("Requesting upload…");
  const ticket = await request<UploadTicket>(
    `/v1/documents?filename=${encodeURIComponent(file.name)}`,
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
  return request<IndexResult>(
    `/v1/documents/${ticket.doc_id}/index?filename=${encodeURIComponent(ticket.filename)}`,
    { method: "POST" },
  );
}
