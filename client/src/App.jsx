import { useState, useCallback, useEffect } from 'react';
import Header from './components/Header';
import TranslationPanel from './components/TranslationPanel';
import TranslationHistory from './components/TranslationHistory';
import { translateText, getHistory, clearHistory } from './services/api';
import './App.css';

export default function App() {
  const [sourceText, setSourceText] = useState('');
  const [translation, setTranslation] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  // Fetch history on mount
  useEffect(() => {
    getHistory()
      .then((data) => setHistory(data.history || []))
      .catch(() => {}); // Silently fail – server may not be up yet
  }, []);

  // Translate handler
  const handleTranslate = useCallback(async () => {
    const trimmed = sourceText.trim();
    if (!trimmed) return;

    setLoading(true);
    setError('');
    setTranslation('');

    try {
      const result = await translateText(trimmed);
      setTranslation(result.translation);

      // Prepend to local history
      setHistory((prev) => {
        const next = [
          { source: result.source, translation: result.translation, timestamp: result.timestamp },
          ...prev,
        ];
        return next.slice(0, 50);
      });
    } catch (err) {
      setError(err.message || 'Translation failed. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [sourceText]);

  // Keyboard shortcut: Ctrl/Cmd + Enter to translate
  const handleKeyDown = useCallback(
    (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        handleTranslate();
      }
    },
    [handleTranslate],
  );

  // Select from history
  const handleHistorySelect = useCallback((item) => {
    setSourceText(item.source);
    setTranslation(item.translation);
    setError('');
  }, []);

  // Clear history
  const handleClearHistory = useCallback(async () => {
    try {
      await clearHistory();
      setHistory([]);
    } catch {
      // Silently fail
    }
  }, []);

  return (
    <div className="app" onKeyDown={handleKeyDown}>
      <Header
        historyCount={history.length}
        showHistory={showHistory}
        onToggleHistory={() => setShowHistory((v) => !v)}
      />

      {/* Language direction bar */}
      <div className="lang-direction">
        <span className="lang-label">
          <span className="lang-flag">🇫🇷</span> French
        </span>
        <span className="lang-arrow">
          <span className="lang-arrow-line" />
        </span>
        <span className="lang-label">
          <span className="lang-flag">🇬🇧</span> English
        </span>
      </div>

      <main className="app-main">
        <div className="translation-container">
          {/* Panels */}
          <div className="panels-wrapper">
            <TranslationPanel
              type="source"
              language="French"
              value={sourceText}
              onChange={setSourceText}
              placeholder="Type French text here… (e.g. bonjour, comment ça va ?)"
            />
            <TranslationPanel
              type="target"
              language="English"
              value={translation}
              loading={loading}
              placeholder="Translation will appear here…"
            />
          </div>

          {/* Error message */}
          {error && (
            <div className="error-message" role="alert">
              <span className="error-icon">⚠️</span>
              <span>{error}</span>
            </div>
          )}

          {/* Translate button */}
          <div className="translate-btn-wrapper">
            <button
              id="translate-btn"
              className="translate-btn"
              onClick={handleTranslate}
              disabled={loading || !sourceText.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Translating…
                </>
              ) : (
                <>⚡ Translate</>
              )}
            </button>
          </div>

          {/* Keyboard hint */}
          <p
            style={{
              textAlign: 'center',
              fontSize: '0.7rem',
              color: 'var(--text-muted)',
              marginTop: '-4px',
            }}
          >
            Press <kbd style={{ padding: '1px 5px', borderRadius: '3px', background: 'rgba(255,255,255,0.06)', fontSize: '0.65rem' }}>Ctrl</kbd> + <kbd style={{ padding: '1px 5px', borderRadius: '3px', background: 'rgba(255,255,255,0.06)', fontSize: '0.65rem' }}>Enter</kbd> to translate
          </p>
        </div>

        {/* History sidebar */}
        {showHistory && (
          <TranslationHistory
            history={history}
            onSelect={handleHistorySelect}
            onClear={handleClearHistory}
          />
        )}
      </main>
    </div>
  );
}
