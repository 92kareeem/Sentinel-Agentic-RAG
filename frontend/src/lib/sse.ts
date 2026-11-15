// A minimal Server-Sent Events parser for a fetch() body.
//
// EventSource cannot POST and cannot send an API key header, so the stream is
// read from fetch() instead, and this turns its byte chunks into events. Kept
// pure and incremental — feed it text as it arrives, get back the events that
// are complete — so it is unit-tested without a network.

export interface SseEvent {
  event: string;
  data: string;
}

export class SseParser {
  private buffer = "";

  /** Add a chunk of text; return every event it completed. */
  push(chunk: string): SseEvent[] {
    // Normalise CRLF so a proxy that rewrites line endings cannot split the
    // stream into events that never terminate.
    this.buffer += chunk.replace(/\r\n/g, "\n");
    const out: SseEvent[] = [];
    let sep = this.buffer.indexOf("\n\n");
    while (sep !== -1) {
      const block = this.buffer.slice(0, sep);
      this.buffer = this.buffer.slice(sep + 2);
      const parsed = parseBlock(block);
      if (parsed) out.push(parsed);
      sep = this.buffer.indexOf("\n\n");
    }
    return out;
  }

  /** Whatever is left when the stream ends (a final event without a blank line). */
  flush(): SseEvent[] {
    const rest = this.buffer.trim();
    this.buffer = "";
    const parsed = rest ? parseBlock(rest) : null;
    return parsed ? [parsed] : [];
  }
}

function parseBlock(block: string): SseEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (!line || line.startsWith(":")) continue; // comment / keep-alive
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  return data.length ? { event, data: data.join("\n") } : null;
}
