import { useEffect, useState } from "react";
import { ApiError, getTrace, postQuery } from "./api";
import { AnswerPanel } from "./components/AnswerPanel";
import { CitationDrawer } from "./components/CitationDrawer";
import { QueryBox } from "./components/QueryBox";
import { ScoreBadge } from "./components/ScoreBadge";
import { TraceTimeline } from "./components/TraceTimeline";
import { UploadBar } from "./components/UploadBar";
import type { Citation, ConversationTurn, QueryResult, TraceRecord } from "./types";
import { isRefusal } from "./types";

export default function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [trace, setTrace] = useState<TraceRecord | null>(null);
  const [citation, setCitation] = useState<Citation | null>(null);
  const [activeDoc, setActiveDoc] = useState<{ docId: string; filename: string } | null>(null);
  const [conversationHistory, setConversationHistory] = useState<ConversationTurn[]>([]);
  const [renderedAnswer, setRenderedAnswer] = useState("");
  const [isRenderingAnswer, setIsRenderingAnswer] = useState(false);

  useEffect(() => {
    if (!result || isRefusal(result)) {
      setIsRenderingAnswer(false);
      return;
    }

    const target = result.answer;
    const chars = target.split("");
    let index = 0;
    setRenderedAnswer("");
    setIsRenderingAnswer(true);

    const timer = window.setInterval(() => {
      index += 1;
      setRenderedAnswer(chars.slice(0, index).join(""));
      if (index >= chars.length) {
        window.clearInterval(timer);
        setIsRenderingAnswer(false);
      }
    }, 16);

    return () => window.clearInterval(timer);
  }, [result]);

  const ask = async (query: string) => {
    const nextHistory = [...conversationHistory, { role: "user", content: query }];
    setLoading(true);
    setError(null);
    setResult(null);
    setTrace(null);
    setCitation(null);
    setRenderedAnswer("");
    setIsRenderingAnswer(false);
    try {
      const r = await postQuery(query, activeDoc?.docId, nextHistory);
      setResult(r);
      setConversationHistory([
        ...nextHistory,
        {
          role: "assistant",
          content: isRefusal(r) ? r.reason : r.answer,
        },
      ]);
      getTrace(r.trace_id).then(setTrace).catch(() => {});
    } catch (e) {
      if (e instanceof ApiError && e.status === 429) {
        setError(
          `Daily demo quota reached — resets ${
            e.retryAfter ? `in ~${Math.ceil(Number(e.retryAfter) / 3600)}h` : "at midnight UTC"
          }. Thanks for trying Sentinel!`,
        );
      } else if (e instanceof ApiError) {
        setError(`${e.status}: ${e.detail}`);
      } else {
        setError("Network error — is the API reachable?");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app">
      <header className="masthead">
        <h1>Sentinel</h1>
        <p>Self-healing document Q&A — every sentence cited, every answer verified.</p>
      </header>

      <QueryBox loading={loading} onSubmit={ask} error={error} />

      {conversationHistory.length > 0 && (
        <section className="conversation-history">
          <h3>Conversation context</h3>
          <ul>
            {conversationHistory.map((turn, idx) => (
              <li key={`${turn.role}-${idx}`}>
                <strong>{turn.role}:</strong> {turn.content}
              </li>
            ))}
          </ul>
        </section>
      )}

      <UploadBar
        onIndexed={(filename, _chunks, docId) => setActiveDoc({ docId, filename })}
      />
      {activeDoc && (
        <div className="scope">
          <span>
            Answering from <b>{activeDoc.filename}</b>
          </span>
          <button className="scope-clear" onClick={() => setActiveDoc(null)}>
            Ask across all documents
          </button>
        </div>
      )}

      {result && isRefusal(result) && (
        <section className="refusal">
          <h3>Sentinel declined to answer</h3>
          <p>{result.reason === "INSUFFICIENT_CONTEXT"
            ? "The indexed documents don't contain enough information for a grounded answer."
            : result.reason}</p>
          <span className="trace-id">trace: {result.trace_id}</span>
        </section>
      )}

      {result && !isRefusal(result) && (
        <>
          <AnswerPanel
            answer={renderedAnswer}
            citations={result.citations}
            onChipClick={setCitation}
            isRendering={isRenderingAnswer}
          />
          <ScoreBadge
            critic={result.critic}
            repairCount={result.repair_count}
            model={result.model_used}
            latencyMs={result.latency_ms}
          />
          {trace && <TraceTimeline steps={trace.steps} />}
          <span className="trace-id">trace: {result.trace_id}</span>
        </>
      )}

      <CitationDrawer citation={citation} onClose={() => setCitation(null)} />
    </main>
  );
}
