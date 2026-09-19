import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, deleteDocument, getTrace, listDocuments, postQuery } from "./api";
import { ChatThread } from "./components/ChatThread";
import { Composer } from "./components/Composer";
import { Header } from "./components/Header";
import { InsightsModal } from "./components/InsightsModal";
import { Sidebar } from "./components/Sidebar";
import { SourceDrawer } from "./components/SourceDrawer";
import { UploadModal } from "./components/UploadModal";
import type {
  ChatMessage,
  Citation,
  ConversationTurn,
  DocumentSummary,
  QueryResult,
  TraceRecord,
} from "./types";
import { isRefusal } from "./types";

function uid(): string {
  return Math.random().toString(36).slice(2);
}

const CITATION_TAG = /\s*\[chunk:[\w-]+\]/g;

// What this turn should look like when it is sent back as conversation
// history on the NEXT question.
//
// The raw response is the wrong thing to echo. A refusal's `reason` is the
// literal control token "INSUFFICIENT_CONTEXT", and an answer is full of
// [chunk:<id>] tags — feeding either back means the next prompt contains an
// assistant turn that says something no human would say, spends tokens on
// ids that are meaningless out of context, and invites the model to imitate
// the tagging convention from history rather than from its instructions.
//
// History is background about what was discussed, never evidence, so a plain
// readable sentence is exactly what belongs here.
function historyTurnFor(r: QueryResult): string {
  if (isRefusal(r)) return "I couldn't answer that from the available documents.";
  return r.answer.replace(CITATION_TAG, "").trim();
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [traces, setTraces] = useState<Record<string, TraceRecord>>({});
  const [citation, setCitation] = useState<Citation | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [insightsOpen, setInsightsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const historyRef = useRef<ConversationTurn[]>([]);

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
    } catch {
      // Non-fatal: the chat still works even if the sidebar can't load.
    }
  }, []);

  useEffect(() => {
    refreshDocuments();
  }, [refreshDocuments]);

  // Poll while anything is still processing — the backend has no push
  // channel for lifecycle transitions, so this is the only way the sidebar
  // status can move from PROCESSING to INDEXED/FAILED on its own.
  useEffect(() => {
    const inFlight = documents.some((d) => d.status === "UPLOADING" || d.status === "UPLOADED" || d.status === "PROCESSING");
    if (!inFlight) return;
    const t = window.setInterval(refreshDocuments, 2000);
    return () => window.clearInterval(t);
  }, [documents, refreshDocuments]);

  const activeDoc = documents.find((d) => d.document_id === activeDocId) ?? null;
  const scopeLabel = activeDoc ? `Answering from ${activeDoc.filename}` : "Asking across all documents";

  const ask = async (query: string) => {
    const userMsg: ChatMessage = { id: uid(), role: "user", text: query };
    const pendingMsg: ChatMessage = { id: uid(), role: "assistant", kind: "pending" };
    setMessages((m) => [...m, userMsg, pendingMsg]);
    setLoading(true);

    const nextHistory = [...historyRef.current, { role: "user", content: query }];

    try {
      const r = await postQuery(query, activeDocId, nextHistory);
      historyRef.current = [...nextHistory, { role: "assistant", content: historyTurnFor(r) }];

      const finalMsg: ChatMessage = isRefusal(r)
        ? {
            id: pendingMsg.id,
            role: "assistant",
            kind: "refusal",
            reason: r.reason,
            reasonCode: r.reason_code ?? "INSUFFICIENT_EVIDENCE",
            traceId: r.trace_id,
          }
        : {
            id: pendingMsg.id,
            role: "assistant",
            kind: "answer",
            text: r.answer,
            citations: r.citations,
            critic: r.critic,
            repairCount: r.repair_count,
            model: r.model_used,
            latencyMs: r.latency_ms,
            traceId: r.trace_id,
          };
      setMessages((m) => m.map((msg) => (msg.id === pendingMsg.id ? finalMsg : msg)));

      getTrace(r.trace_id)
        .then((t) => setTraces((prev) => ({ ...prev, [r.trace_id]: t })))
        .catch(() => {});
    } catch (e) {
      let text = "Sentinel couldn't reach the server. Please try again.";
      if (e instanceof ApiError && e.status === 429) {
        text = `Daily demo quota reached — resets ${
          e.retryAfter ? `in ~${Math.ceil(Number(e.retryAfter) / 3600)}h` : "at midnight UTC"
        }. Thanks for trying Sentinel!`;
      } else if (e instanceof ApiError) {
        text = e.detail || text;
      }
      setMessages((m) =>
        m.map((msg) => (msg.id === pendingMsg.id ? { id: pendingMsg.id, role: "assistant", kind: "error", message: text } : msg)),
      );
    } finally {
      setLoading(false);
    }
  };

  const onDelete = async (docId: string) => {
    try {
      await deleteDocument(docId);
      if (activeDocId === docId) setActiveDocId(null);
      refreshDocuments();
    } catch {
      // Surfacing this inline would need its own toast primitive; the
      // sidebar simply keeps showing the document, which is an accurate
      // reflection of "deletion did not take effect".
    }
  };

  const newChat = () => {
    setMessages([]);
    setTraces({});
    historyRef.current = [];
  };

  return (
    <div className="shell">
      <Header
        onMenuClick={() => setSidebarOpen(true)}
        onNewChat={newChat}
        onInsightsClick={() => setInsightsOpen(true)}
        hasMessages={messages.length > 0}
      />

      <div className="body">
        <Sidebar
          documents={documents}
          activeDocId={activeDocId}
          onSelect={(id) => {
            setActiveDocId(id);
            setSidebarOpen(false);
          }}
          onUploadClick={() => setUploadOpen(true)}
          onDelete={onDelete}
          open={sidebarOpen}
          onCloseMobile={() => setSidebarOpen(false)}
        />

        <main className="chat-pane">
          <ChatThread
            messages={messages}
            traces={traces}
            onCiteClick={setCitation}
            hasDocuments={documents.some((d) => d.status !== "DELETED")}
            onExample={ask}
          />
          <Composer loading={loading} onSubmit={ask} scopeLabel={scopeLabel} />
        </main>
      </div>

      {uploadOpen && (
        <UploadModal
          onClose={() => setUploadOpen(false)}
          onUploaded={() => {
            setUploadOpen(false);
            refreshDocuments();
          }}
        />
      )}

      {insightsOpen && <InsightsModal onClose={() => setInsightsOpen(false)} />}

      <SourceDrawer citation={citation} onClose={() => setCitation(null)} />
    </div>
  );
}
