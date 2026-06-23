/**
 * API service module.
 *
 * Handles all communication with the Node.js backend.
 * In development the Vite proxy rewrites /api → http://localhost:3001.
 */

const BASE_URL = '/api';

/**
 * Translate French text to English.
 * @param {string} text – French input text
 * @returns {Promise<{source: string, translation: string, timestamp: string}>}
 */
export async function translateText(text) {
  const res = await fetch(`${BASE_URL}/translate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.error || `Translation failed (${res.status})`);
  }

  return res.json();
}

/**
 * Fetch translation history.
 * @returns {Promise<{history: Array}>}
 */
export async function getHistory() {
  const res = await fetch(`${BASE_URL}/history`);
  if (!res.ok) throw new Error('Failed to fetch history');
  return res.json();
}

/**
 * Clear translation history.
 */
export async function clearHistory() {
  const res = await fetch(`${BASE_URL}/history`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to clear history');
  return res.json();
}
