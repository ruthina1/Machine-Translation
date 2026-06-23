/**
 * Node.js API Gateway for Machine Translation.
 *
 * Routes:
 *   POST /api/translate  – proxies to Python inference server
 *   GET  /api/history    – returns recent translations
 *   GET  /api/health     – health check
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();

const PORT = process.env.PORT || 3001;
const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:5000';

// ── Middleware ───────────────────────────────────────
app.use(cors());
app.use(express.json());

// ── In-memory translation history ───────────────────
const MAX_HISTORY = 50;
let translationHistory = [];

// ── Routes ──────────────────────────────────────────

// Health check
app.get('/api/health', async (_req, res) => {
  try {
    const pythonHealth = await axios.get(`${PYTHON_SERVICE_URL}/health`, {
      timeout: 3000,
    });
    res.json({
      status: 'ok',
      node: true,
      python: pythonHealth.data,
    });
  } catch {
    res.json({
      status: 'degraded',
      node: true,
      python: { status: 'unreachable' },
    });
  }
});

// Translate
app.post('/api/translate', async (req, res) => {
  const { text } = req.body;

  // Input validation
  if (!text || typeof text !== 'string') {
    return res.status(400).json({ error: 'Please provide a "text" field (string).' });
  }

  const trimmed = text.trim();
  if (trimmed.length === 0) {
    return res.status(400).json({ error: 'Text cannot be empty.' });
  }
  if (trimmed.length > 500) {
    return res.status(400).json({ error: 'Text too long. Maximum 500 characters.' });
  }

  try {
    const response = await axios.post(
      `${PYTHON_SERVICE_URL}/predict`,
      { text: trimmed },
      { timeout: 30000 }
    );

    const result = {
      source: trimmed,
      translation: response.data.translation,
      timestamp: new Date().toISOString(),
    };

    // Prepend to history (most recent first)
    translationHistory.unshift(result);
    if (translationHistory.length > MAX_HISTORY) {
      translationHistory = translationHistory.slice(0, MAX_HISTORY);
    }

    return res.json(result);
  } catch (err) {
    console.error('Translation error:', err.message);

    if (err.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: 'Translation service is not available. Please ensure the Python inference server is running.',
      });
    }

    const status = err.response?.status || 500;
    const message =
      err.response?.data?.error || 'An unexpected error occurred during translation.';
    return res.status(status).json({ error: message });
  }
});

// History
app.get('/api/history', (_req, res) => {
  res.json({ history: translationHistory });
});

// Clear history
app.delete('/api/history', (_req, res) => {
  translationHistory = [];
  res.json({ message: 'History cleared.' });
});

// ── Error handling middleware ────────────────────────
app.use((err, _req, res, _next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error.' });
});

// ── Start ───────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`🚀 Node.js API gateway running on http://localhost:${PORT}`);
  console.log(`   Proxying to Python service at ${PYTHON_SERVICE_URL}`);
});
