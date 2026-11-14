import { describe, expect, it } from "vitest";
import { SseParser } from "./sse";

describe("SseParser", () => {
  it("parses complete events and holds back a partial one", () => {
    const p = new SseParser();
    expect(p.push('event: start\ndata: {"a":1}\n\nevent: stage\ndata: {"b"')).toEqual([
      { event: "start", data: '{"a":1}' },
    ]);
    expect(p.push(":2}\n\n")).toEqual([{ event: "stage", data: '{"b":2}' }]);
  });

  it("survives an event split across many tiny chunks", () => {
    const p = new SseParser();
    const raw = "event: result\ndata: hello\n\n";
    const out = [...raw].flatMap((ch) => p.push(ch));
    expect(out).toEqual([{ event: "result", data: "hello" }]);
  });

  it("normalises CRLF line endings", () => {
    const p = new SseParser();
    expect(p.push("event: x\r\ndata: y\r\n\r\n")).toEqual([{ event: "x", data: "y" }]);
  });

  it("ignores comments and defaults the event name", () => {
    const p = new SseParser();
    expect(p.push(": keep-alive\n\ndata: z\n\n")).toEqual([{ event: "message", data: "z" }]);
  });

  it("flushes a last event that has no trailing blank line", () => {
    const p = new SseParser();
    expect(p.push("event: result\ndata: done")).toEqual([]);
    expect(p.flush()).toEqual([{ event: "result", data: "done" }]);
  });
});
