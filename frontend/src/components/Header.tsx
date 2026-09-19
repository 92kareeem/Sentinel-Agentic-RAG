import { ThemeToggle } from "./ThemeToggle";

interface Props {
  onMenuClick: () => void;
  onNewChat: () => void;
  hasMessages: boolean;
}

export function Header({ onMenuClick, onNewChat, hasMessages }: Props) {
  return (
    <header className="topbar">
      <button className="mobile-menu" onClick={onMenuClick} aria-label="Open documents">☰</button>
      <div className="brand">
        <span className="brand-mark" aria-hidden>S</span>
        <span className="brand-name">Sentinel</span>
        <span className="brand-tag">Verified answers from your documents</span>
      </div>
      <div className="topbar-actions">
        {hasMessages && (
          <button className="new-chat" onClick={onNewChat}>
            New chat
          </button>
        )}
        <ThemeToggle />
      </div>
    </header>
  );
}
