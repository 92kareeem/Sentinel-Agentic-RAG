import { useId, useRef, useState } from "react";
import { useDialog } from "../lib/useDialog";
import type { KnowledgeGap } from "../types";
import { Icon, type IconName } from "./Icon";
import { GapsPanel } from "./panels/GapsPanel";
import { RatchetPanel } from "./panels/RatchetPanel";
import { TelemetryPanel } from "./panels/TelemetryPanel";
import { TracesPanel } from "./panels/TracesPanel";

export type ControlTab = "telemetry" | "gaps" | "ratchet" | "traces";

const TABS: { id: ControlTab; label: string; icon: IconName; blurb: string }[] = [
  { id: "telemetry", label: "Telemetry", icon: "activity", blurb: "Latency, token burn, cost and refusal rate, live from traces." },
  { id: "gaps", label: "Corpus gaps", icon: "layers", blurb: "What your documents could not answer, ranked — with the fix." },
  { id: "ratchet", label: "Quality ratchet", icon: "lock", blurb: "Fixed mistakes become permanent tests that CI enforces." },
  { id: "traces", label: "Traces", icon: "branch", blurb: "Every request, node by node, including every repair." },
];

interface Props {
  initialTab?: ControlTab;
  onClose: () => void;
  onFixGap: (gap: KnowledgeGap) => void;
}

// The Sentinel control plane: one dialog, four tabs, for whoever operates the
// system rather than whoever asks it questions. WAI-ARIA tabs: arrow keys
// move between tabs, Tab moves into the panel.
export function ControlPlane({ initialTab = "telemetry", onClose, onFixGap }: Props) {
  const [tab, setTab] = useState<ControlTab>(initialTab);
  const ref = useDialog<HTMLDivElement>(true, onClose);
  const titleId = useId();
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const current = TABS.find((t) => t.id === tab)!;

  const onKeyDown = (e: React.KeyboardEvent) => {
    const i = TABS.findIndex((t) => t.id === tab);
    let next = i;
    if (e.key === "ArrowRight") next = (i + 1) % TABS.length;
    else if (e.key === "ArrowLeft") next = (i - 1 + TABS.length) % TABS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = TABS.length - 1;
    else return;
    e.preventDefault();
    setTab(TABS[next].id);
    tabRefs.current[TABS[next].id]?.focus();
  };

  return (
    <div className="scrim" onClick={onClose}>
      <div
        ref={ref}
        className="dialog dialog-wide dialog-tall"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="dialog-header">
          <div>
            <h2 id={titleId} className="dialog-title">
              Control plane
            </h2>
            <p className="dialog-sub">{current.blurb}</p>
          </div>
          <button type="button" className="btn btn-ghost btn-icon dialog-close" onClick={onClose} aria-label="Close control plane">
            <Icon name="close" size={18} />
          </button>
        </div>

        <div className="tabs" role="tablist" aria-label="Control plane sections" onKeyDown={onKeyDown}>
          {TABS.map((t) => (
            <button
              key={t.id}
              ref={(el) => {
                tabRefs.current[t.id] = el;
              }}
              type="button"
              role="tab"
              id={`tab-${t.id}`}
              aria-selected={tab === t.id}
              aria-controls={`panel-${t.id}`}
              tabIndex={tab === t.id ? 0 : -1}
              className="tab"
              onClick={() => setTab(t.id)}
              data-autofocus={tab === t.id ? true : undefined}
            >
              <Icon name={t.icon} size={14} />
              {t.label}
            </button>
          ))}
        </div>

        <div className="dialog-body" role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} tabIndex={0}>
          {tab === "telemetry" && <TelemetryPanel />}
          {tab === "gaps" && <GapsPanel onFix={onFixGap} />}
          {tab === "ratchet" && <RatchetPanel />}
          {tab === "traces" && <TracesPanel />}
        </div>
      </div>
    </div>
  );
}
