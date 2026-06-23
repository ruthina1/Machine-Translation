import { useState, useCallback } from 'react';

/**
 * TranslationPanel — reusable card for source input or target output.
 *
 * @param {'source' | 'target'} type
 * @param {string}              language   – display name (e.g. "French")
 * @param {string}              value
 * @param {(v: string) => void} [onChange] – only for source panel
 * @param {boolean}             loading
 * @param {string}              placeholder
 */
export default function TranslationPanel({
  type,
  language,
  value,
  onChange,
  loading = false,
  placeholder = '',
}) {
  const [copied, setCopied] = useState(false);
  const isSource = type === 'source';

  const handleCopy = useCallback(() => {
    if (!value) return;
    navigator.clipboard.writeText(value).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [value]);

  const handleClear = useCallback(() => {
    if (onChange) onChange('');
  }, [onChange]);

  return (
    <div className="panel" id={`panel-${type}`}>
      {/* Panel header */}
      <div className="panel-header">
        <span className="panel-lang">
          <span className={`panel-lang-dot ${type}`} />
          {language}
        </span>

        <div className="panel-actions">
          {isSource && value && (
            <button
              className="panel-action-btn"
              onClick={handleClear}
              title="Clear"
              aria-label="Clear text"
            >
              ✕
            </button>
          )}
          {!isSource && (
            <button
              className={`panel-action-btn ${copied ? 'copied' : ''}`}
              onClick={handleCopy}
              title={copied ? 'Copied!' : 'Copy to clipboard'}
              aria-label="Copy translation"
            >
              {copied ? '✓' : '📋'}
            </button>
          )}
        </div>
      </div>

      {/* Panel body */}
      <div className="panel-body">
        {isSource ? (
          <textarea
            id="source-textarea"
            className="panel-textarea"
            value={value}
            onChange={(e) => onChange?.(e.target.value)}
            placeholder={placeholder}
            maxLength={500}
            spellCheck={false}
          />
        ) : loading ? (
          <div className="loading-skeleton">
            <div className="skeleton-line skeleton" />
            <div className="skeleton-line skeleton" />
            <div className="skeleton-line skeleton" />
          </div>
        ) : (
          <div
            id="target-output"
            className={`panel-output ${!value ? 'empty' : ''}`}
          >
            {value || placeholder}
          </div>
        )}
      </div>

      {/* Panel footer */}
      <div className="panel-footer">
        <span className="char-count">
          {isSource
            ? `${value.length} / 500`
            : value
            ? `${value.split(' ').length} words`
            : ''}
        </span>
      </div>
    </div>
  );
}
