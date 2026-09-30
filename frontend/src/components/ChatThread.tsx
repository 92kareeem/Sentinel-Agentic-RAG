import { useEffect, useLayoutEffect, useRef } from "react";
import { prefersReducedMotion } from "../lib/useDialog";
import type { ChatMessage, Citation, TraceRecord } from "../types";
import { EmptyState } from "./EmptyState";
import { MessageBubble } from "./MessageBubble";

interface Props {
  messages: ChatMessage[];
  traces: Record<string, TraceRecord>;
  onCiteClick: (citation: Citation) => void;
  hasDocuments: boolean;
  onExample: (q: string) => void;
  busy: boolean;
}

// How close to the bottom still counts as "following the conversation".
const STICK_PX = 120;

export function ChatThread({ messages, traces, onCiteClick, hasDocuments, onExample, busy }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const lastCount = useRef(messages.length);

  // Follow the conversation when it grows, but never drag someone back down
  // while they are reading an earlier answer.
  //
  // This used to fire only when the NUMBER of messages changed. The "…"
  // placeholder and the answer that replaces it are the same message, so the
  // count does not change when the answer arrives — and a long answer, or a
  // refusal card, ended up hidden behind the question box, cut off mid-title.
  // It now reacts to any change, and always scrolls when the user has just
  // asked something, since that is an explicit request to see the reply.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const asked = messages.length > lastCount.current;
    lastCount.current = messages.length;
    if (asked || stick.current) {
      el.scrollTo({ top: el.scrollHeight, behavior: prefersReducedMotion() ? "auto" : "smooth" });
    }
  }, [messages]);

  // A fresh conversation starts at the top of the welcome screen.
  useEffect(() => {
    if (messages.length === 0) scrollRef.current?.scrollTo({ top: 0 });
  }, [messages.length]);

  return (
    <div
      ref={scrollRef}
      className="chat-scroll"
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < STICK_PX;
      }}
    >
      <div className="chat-column">
        {messages.length === 0 ? (
          <EmptyState hasDocuments={hasDocuments} onExample={onExample} />
        ) : (
          // A log: screen readers announce each new answer as it arrives,
          // politely, without interrupting what is being read.
          <section className="thread" role="log" aria-live="polite" aria-busy={busy} aria-label="Conversation">
            {messages.map((m) => (
              <MessageBubble
                key={m.id}
                message={m}
                onCiteClick={onCiteClick}
                trace={m.role === "assistant" && m.kind === "answer" ? traces[m.traceId] ?? null : null}
              />
            ))}
          </section>
        )}
      </div>
    </div>
  );
}
