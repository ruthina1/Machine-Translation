/**
 * TranslationHistory — sidebar listing recent translations.
 */
export default function TranslationHistory({
  history,
  onSelect,
  onClear,
}) {
  return (
    <aside className="history-sidebar" id="history-sidebar">
      <div className="history-header">
        <span className="history-title">Recent Translations</span>
        {history.length > 0 && (
          <button
            className="history-clear-btn"
            onClick={onClear}
            id="clear-history-btn"
          >
            Clear All
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="history-empty">
          <div className="history-empty-icon">📝</div>
          <p>No translations yet.<br />Start typing to translate!</p>
        </div>
      ) : (
        <div className="history-list">
          {history.map((item, idx) => (
            <div
              key={`${item.timestamp}-${idx}`}
              className="history-item"
              onClick={() => onSelect(item)}
              style={{ animationDelay: `${idx * 0.05}s` }}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === 'Enter' && onSelect(item)}
            >
              <div className="history-item-source">
                {truncate(item.source, 60)}
              </div>
              <div className="history-item-target">
                {truncate(item.translation, 60)}
              </div>
              <div className="history-item-time">
                {formatTime(item.timestamp)}
              </div>
            </div>
          ))}
        </div>
      )}
    </aside>
  );
}

/* ── Helpers ──────────────────────────────── */

function truncate(str, max) {
  if (!str) return '';
  return str.length > max ? str.slice(0, max) + '…' : str;
}

function formatTime(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}
