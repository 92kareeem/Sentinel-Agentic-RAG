import { Icon } from "./Icon";
import { ThemeToggle } from "./ThemeToggle";

interface Props {
  onMenuClick: () => void;
  menuOpen: boolean;
  onNewChat: () => void;
  onInsightsClick: () => void;
  hasMessages: boolean;
}

export function Header({ onMenuClick, menuOpen, onNewChat, onInsightsClick, hasMessages }: Props) {
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
        {/* The page's one h1. Every other heading (the welcome title, dialog
            titles, the documents list) sits below it, so a screen reader's
            heading list reads as an outline rather than a pile of h3s. */}
        <h1 className="brand-name">Sentinel</h1>
        <span className="brand-tag">Verified answers from your documents</span>
      </div>

      <div className="topbar-actions">
        {/* First in the group on purpose: the group is right-aligned, so an
            item appearing at its left edge moves nothing else. It used to be
            second, and appearing pushed Coverage sideways under the cursor. */}
        {hasMessages && (
          <button type="button" className="btn btn-secondary" onClick={onNewChat}>
            <Icon name="edit" />
            <span className="btn-label">New chat</span>
          </button>
        )}
        {/* Always available: the report is about the workspace over time, not
            about the open conversation. */}
        <button type="button" className="btn btn-secondary" onClick={onInsightsClick}>
          <Icon name="chart" />
          <span className="btn-label">Coverage</span>
        </button>
        <ThemeToggle />
      </div>
    </header>
  );
}
