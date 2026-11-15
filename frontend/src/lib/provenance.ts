// Recompute a citation's provenance hash in the browser.
//
// The server says "this quote is characters 120–410 of the file whose sha256
// is X". This checks that claim instead of trusting it: the same recipe the
// server publishes (ChunkDetail.hash_recipe), computed locally with WebCrypto.
// A mismatch means the text, its position or the source file differ from what
// was indexed — and the drawer says so rather than showing a green tick.

import type { ChunkDetail } from "../types";

export function provenancePayload(c: Pick<ChunkDetail, "document_checksum_sha256" | "char_start" | "char_end" | "text">): string {
  return `${c.document_checksum_sha256}|${c.char_start}|${c.char_end}|${c.text}`;
}

export async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export type Verification =
  | { state: "verified"; hash: string }
  | { state: "mismatch"; expected: string; actual: string }
  | { state: "unavailable"; reason: string };

export async function verifyChunk(chunk: ChunkDetail): Promise<Verification> {
  if (!chunk.provenance_hash) {
    return { state: "unavailable", reason: "This passage was indexed before provenance hashing existed." };
  }
  if (typeof crypto === "undefined" || !crypto.subtle) {
    return { state: "unavailable", reason: "This browser cannot compute SHA-256 on this page (it needs HTTPS)." };
  }
  const actual = await sha256Hex(provenancePayload(chunk));
  return actual === chunk.provenance_hash
    ? { state: "verified", hash: actual }
    : { state: "mismatch", expected: chunk.provenance_hash, actual };
}
