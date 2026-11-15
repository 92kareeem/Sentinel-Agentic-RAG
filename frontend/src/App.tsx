import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  apiKey,
  deleteDocument,
  getHealth,
  getTrace,
  listDocuments,
  retryDocument,
  setApiKey,
  streamQuery,
} from "./api";
import { ChaosDemo } from "./components/ChaosDemo";
import { ChatThread } from "./components/ChatThread";
import { COMPOSER_ID, Composer } from "./components/Composer";
import { type ControlTab, ControlPlane } from "./components/ControlPlane";
import { Header } from "./components/Header";
import { Sidebar } from "./components/Sidebar";
import { SourceDrawer } from "./components/SourceDrawer";
import { type UploadContext, UploadModal } from "./components/UploadModal";
import type {
  ChatMessage,
  Citation,
  ConversationTurn,
  DocumentSummary,
  Health,
  KnowledgeGap,
  QueryResult,
  StageEvent,
  TraceRecord,
} from "./types";
import { isRefusal } from "./types";

function uid(): string {
  return Math.random().toString(36).slice(2);
}

const CITATION_TAG = /\s*\[chunk:[\w-]+\]/g;

// What this turn should look like when it is sent back as conversation
// history on the NEXT question: a plain readable sentence, never the
// INSUFFICIENT_CONTEXT control token or [chunk:id] tags.
function historyTurnFor(r: QueryResult): string {
  if (isRefusal(r)) return "I couldn't answer that from the available documents.";
  return r.answer.replace(CITATION_TAG, "").trim();
}

export default function App() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [identity, setIdentity] = useState(apiKey());
  const [activeDocId, setActiveDocId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [traces, setTraces] = useState<Record<string, TraceRecord>>({});
  const [citation, setCitation] = useState<Citation | null>(null);
  const [loading, setLoading] = useState(false);
  const [upload, setUpload] = useState<{ context: UploadContext | null } | null>(null);
  const [controlTab, setControlTab] = useState<ControlTab | null>(null);
  const [chaosOpen, setChaosOpen] = useState(false);
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
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, [refreshDocuments, identity]);

  // Poll while anything is still processing — the backend has no push
  // channel for lifecycle transitions.
  useEffect(() => {
    const inFlight = documents.some((d) => ["UPLOADING", "UPLOADED", "PROCESSING"].includes(d.status));
    if (!inFlight) return;
    const t = window.setInterval(refreshDocuments, 2000);
    return () => window.clearInterval(t);
  }, [documents, refreshDocuments]);

  const activeDoc = documents.find((d) => d.document_id === activeDocId) ?? null;

  const ask = async (query: string) => {
    const userMsg: ChatMessage = { id: uid(), role: "user", text: query };
    const pendingId = uid();
    setMessages((m) => [...m, userMsg, { id: pendingId, role: "assistant", kind: "pending", stages: [] }]);
    setLoading(true);

    const nextHistory = [...historyRef.current, { role: "user", content: query }];
    const onStage = (stage: StageEvent) =>
      setMessages((m) =>
        m.map((msg) =>
          msg.id === pendingId && msg.role === "assistant" && msg.kind === "pending"
            ? { ...msg, stages: [...msg.stages, stage] }
            : msg,
        ),
      );

    try {
      const r = await streamQuery(query, activeDocId, historyRef.current, onStage);
      historyRef.current = [...nextHistory, { role: "assistant", content: historyTurnFor(r) }];

      const finalMsg: ChatMessage = isRefusal(r)
        ? {
            id: pendingId,
            role: "assistant",
            kind: "refusal",
            reason: r.reason,
            reasonCode: r.reason_code ?? "INSUFFICIENT_EVIDENCE",
            traceId: r.trace_id,
          }
        : {
            id: pendingId,
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
      setMessages((m) => m.map((msg) => (msg.id === pendingId ? finalMsg : msg)));

      getTrace(r.trace_id)
        .then((t) => setTraces((prev) => ({ ...prev, [r.trace_id]: t })))
        .catch(() => {});
    } catch (e) {
      let text = "Sentinel couldn't reach the server. Please try again.";
      if (e instanceof ApiError && e.status === 429) {
        text = `Daily demo quota reached — resets ${
          e.retryAfter ? `in ~${Math.ceil(Number(e.retryAfter) / 3600)}h` : "at midnight UTC"
        }.`;
      } else if (e instanceof ApiError && e.status === 422) {
        text = `Blocked by the input guardrails: ${e.detail}`;
      } else if (e instanceof ApiError) {
        text = e.detail || text;
      }
      setMessages((m) =>
        m.map((msg) => (msg.id === pendingId ? { id: pendingId, role: "assistant", kind: "error", message: text } : msg)),
      );
    } finally {
      setLoading(false);
    }
  };

  const onDelete = async (doc: DocumentSummary) => {
    // Deleting removes the document from every future answer and cannot be
    // undone from here, so it asks first.
    if (!window.confirm(`Delete ${doc.filename}? It will no longer be used to answer questions.`)) return;
    try {
      await deleteDocument(doc.document_id);
      if (activeDocId === doc.document_id) setActiveDocId(null);
    } finally {
      refreshDocuments();
    }
  };

  const onRetry = async (doc: DocumentSummary) => {
    try {
      await retryDocument(doc.document_id);
    } finally {
      refreshDocuments();
    }
  };

  const newChat = () => {
    setMessages([]);
    setTraces({});
    historyRef.current = [];
    document.getElementById(COMPOSER_ID)?.focus();
  };

  const switchIdentity = (key: string) => {
    // A different identity sees different documents, so the conversation and
    // any scope that pointed at the old identity's documents start fresh.
    setApiKey(key);
    setIdentity(key);
    setActiveDocId(null);
    newChat();
  };

  // "Upload the missing document & re-verify": close the control plane, take
  // the upload, then ask the gap's question again in the chat — the reader
  // watches the refusal turn into a cited answer.
  const fixGap = (gap: KnowledgeGap) => {
    setControlTab(null);
    setUpload({ context: { topic: gap.topic, questions: gap.example_questions.slice(0, 1) } });
  };

  const hasCorpus =
    documents.some((d) => d.status !== "DELETED") || (health !== null && health.index_version !== "absent");

  return (
    <div className="shell">
      <a className="skip-link" href={`#${COMPOSER_ID}`}>
        Skip to question
      </a>
      <Header
        onMenuClick={() => setSidebarOpen(true)}
        menuOpen={sidebarOpen}
        onNewChat={newChat}
        onControlPlane={() => setControlTab("telemetry")}
        onChaos={health?.demo_mode ? () => setChaosOpen(true) : null}
        hasMessages={messages.length > 0}
        identity={identity}
        onIdentityChange={switchIdentity}
      />

      {health && !health.llm_configured && (
        <p className="banner" role="status">
          No LLM key configured for <strong>{health.llm_provider}</strong>: answers will fail until you set one in{" "}
          <code>.env</code>. The chaos demo still runs, with simulated model calls.
        </p>
      )}

      <div className="layout">
        <Sidebar
          documents={documents}
          activeDocId={activeDocId}
          onSelect={(id) => {
            setActiveDocId(id);
            setSidebarOpen(false);
          }}
          onUploadClick={() => {
            // Close the phone sidebar first: two overlays stacked means two
            // focus traps fighting over the keyboard.
            setSidebarOpen(false);
            setUpload({ context: null });
          }}
          onDelete={onDelete}
          onRetry={onRetry}
          open={sidebarOpen}
          onCloseMobile={() => setSidebarOpen(false)}
        />

        <main className="chat-pane" id="main">
          <ChatThread
            messages={messages}
            traces={traces}
            onCiteClick={setCitation}
            hasDocuments={hasCorpus}
            onExample={ask}
            busy={loading}
          />
          <Composer
            loading={loading}
            onSubmit={ask}
            scopedTo={activeDoc?.filename ?? null}
            onClearScope={() => setActiveDocId(null)}
          />
        </main>
      </div>

      {upload && (
        <UploadModal
          context={upload.context}
          onClose={() => setUpload(null)}
          onUploaded={() => {
            const questions = upload.context?.questions ?? [];
            setUpload(null);
            refreshDocuments();
            setActiveDocId(null);
            // Re-verify: ask the question that used to be refused.
            if (questions[0]) void ask(questions[0]);
          }}
        />
      )}

      {controlTab && (
        <ControlPlane initialTab={controlTab} onClose={() => setControlTab(null)} onFixGap={fixGap} />
      )}

      {chaosOpen && <ChaosDemo onClose={() => setChaosOpen(false)} />}

      <SourceDrawer citation={citation} onClose={() => setCitation(null)} />
    </div>
  );
}
