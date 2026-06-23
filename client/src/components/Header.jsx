/**
 * Header component — app branding, model info badge, and history toggle.
 */
export default function Header({ historyCount, showHistory, onToggleHistory }) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="header-logo" aria-hidden="true">🌐</div>
        <div>
          <h1 className="header-title">Neural Translate</h1>
          <p className="header-subtitle">Transformer-powered machine translation</p>
        </div>
      </div>

      <div className="header-right">
        <div className="header-badge">
          <span className="header-badge-dot" />
          Transformer · 256d · 3 layers
        </div>

        <button
          id="history-toggle"
          className="history-toggle-btn"
          onClick={onToggleHistory}
          aria-label={showHistory ? 'Hide history' : 'Show history'}
        >
          {showHistory ? '✕ Hide' : '📋 History'}
          {historyCount > 0 && (
            <span className="history-count">{historyCount}</span>
          )}
        </button>
      </div>
    </header>
  );
}
