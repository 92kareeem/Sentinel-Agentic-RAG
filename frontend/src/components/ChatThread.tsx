import { useEffect, useRef } from "react";
import type { ChatMessage, Citation, TraceRecord } from "../types";
import { EmptyState } from "./EmptyState";
import { MessageBubble } from "./MessageBubble";

interface Props {
  messages: ChatMessage[];
  traces: Record<string, TraceRecord>;
  onCiteClick: (citation: Citation) => void;
  hasDocuments: boolean;
  onExample: (q: string) => void;
}

export function ChatThread({ messages, traces, onCiteClick, hasDocuments, onExample }: Props) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length]);

  if (messages.length === 0) {
    return <EmptyState hasDocuments={hasDocuments} onExample={onExample} />;
  }

  return (
    <div className="chat-thread">
      {messages.map((m) => (
        <MessageBubble
          key={m.id}
          message={m}
          onCiteClick={onCiteClick}
          trace={m.role === "assistant" && m.kind === "answer" ? traces[m.traceId] ?? null : null}
        />
      ))}
      <div ref={endRef} />
    </div>
  );
}
