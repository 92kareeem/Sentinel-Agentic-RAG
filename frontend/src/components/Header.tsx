import { IS_LOCAL_API, LOCAL_IDENTITIES } from "../api";
import { Icon } from "./Icon";
import { ThemeToggle } from "./ThemeToggle";

interface Props {
  onMenuClick: () => void;
  menuOpen: boolean;
  onNewChat: () => void;
  onControlPlane: () => void;
  onChaos: (() => void) | null; // null when the server is not in demo mode
  hasMessages: boolean;
  identity: string;
  onIdentityChange: (key: string) => void;
}

export function Header({
  onMenuClick,
  menuOpen,
  onNewChat,
  onControlPlane,
  onChaos,
  hasMessages,
  identity,
  onIdentityChange,
}: Props) {
  return (
    <header className="topbar">
      <button
        type="button"
        className="btn btn-ghost btn-icon compact-only"
        onClick={onMenuClick}
        aria-label="Open documents"
        aria-expanded={menuOpen}
        aria-controls="documents-panel"
      >
        <Icon name="menu" size={20} />
      </button>

      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          S
        </span>
        {/* The page's one h1. */}
        <h1 className="brand-name">Sentinel</h1>
        <span className="brand-tag">Verified answers from your documents</span>
      </div>

      <div className="topbar-actions">
        {/* First in the group on purpose: the group is right-aligned, so an
            item appearing at its left edge moves nothing else. */}
        {hasMessages && (
          <button type="button" className="btn btn-secondary" onClick={onNewChat}>
            <Icon name="edit" />
            <span className="btn-label">New chat</span>
          </button>
        )}
        {/* Local fixture identities only: switching shows tenant isolation,
            role sharing and clearance from one browser. */}
        {IS_LOCAL_API && (
          <label className="identity">
            <Icon name="user" />
            <span className="sr-only">Acting as</span>
            <select className="select select-sm" value={identity} onChange={(e) => onIdentityChange(e.target.value)}>
              {LOCAL_IDENTITIES.map((id) => (
                <option key={id.key} value={id.key} title={id.detail}>
                  {id.label}
                </option>
              ))}
            </select>
          </label>
        )}
        {onChaos && (
          <button type="button" className="btn btn-secondary btn-chaos" onClick={onChaos}>
            <Icon name="bolt" />
            <span className="btn-label">Chaos demo</span>
          </button>
        )}
        <button type="button" className="btn btn-secondary" onClick={onControlPlane}>
          <Icon name="chart" />
          <span className="btn-label">Control plane</span>
        </button>
        <ThemeToggle />
      </div>
    </header>
  );
}
