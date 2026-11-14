import { describe, expect, it } from "vitest";
import { provenancePayload, sha256Hex, verifyChunk } from "./provenance";
import type { ChunkDetail } from "../types";

const base: ChunkDetail = {
  chunk_id: "c1",
  document_id: "d",
  source_filename: "refund-policy.md",
  section_path: "Refund Policy > Eligibility",
  page_number: null,
  char_start: 10,
  char_end: 52,
  text: "Customers may request a refund within 30 days.",
  provenance_hash: "",
  document_checksum_sha256: "abc",
  hash_recipe: "",
  classification: "PUBLIC",
};

describe("provenance", () => {
  it("uses exactly the recipe the server publishes", () => {
    expect(provenancePayload(base)).toBe("abc|10|52|Customers may request a refund within 30 days.");
  });

  it("matches a known SHA-256 vector", async () => {
    expect(await sha256Hex("abc")).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  });

  it("verifies an untampered chunk", async () => {
    const hash = await sha256Hex(provenancePayload(base));
    expect(await verifyChunk({ ...base, provenance_hash: hash })).toEqual({ state: "verified", hash });
  });

  it("reports a mismatch when the text was altered", async () => {
    const hash = await sha256Hex(provenancePayload(base));
    const result = await verifyChunk({ ...base, provenance_hash: hash, text: base.text.replace("30", "90") });
    expect(result.state).toBe("mismatch");
  });

  it("is honest about chunks with no hash", async () => {
    expect((await verifyChunk(base)).state).toBe("unavailable");
  });
});
