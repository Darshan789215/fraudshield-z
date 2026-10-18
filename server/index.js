import express from 'express';
import cors from 'cors';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MODEL_CARD, MODEL_VERSION, DETERMINISTIC_SEED, SCENARIOS } from '../shared/contracts.js';
import { configureAuditStore, listAudit, recordFeedback } from './store.js';
import { validateTransaction } from './engine.js';
import { decideTransaction } from './decision.js';
import { approveStagedTraining, exportModelArtifact, getStagedTraining, getTrainedModel, getTrainedModels, hasTrainedModel, importModelArtifact, MAX_ROWS, predictRow, previewDataset, replayDataset, trainDataset } from './ml.js';
import { persistActiveModel, restoreActiveModel } from './model-persistence.js';
import { authRole, createSession, expiredSessionCookie, requiredRole, revokeSession, roleForToken, sessionCookie, sessionRole, validateBindHost, validateProductionSecurity } from './security.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = Number(process.env.PORT || 3001);
const production = process.env.NODE_ENV === 'production';
const dataDir = process.env.FRAUDSHIELD_DATA_DIR || path.resolve(process.cwd(), 'data');
const auditPath = process.env.FRAUDSHIELD_AUDIT_PATH || (production ? path.join(dataDir, 'audit.jsonl') : null);
validateProductionSecurity();
configureAuditStore(auditPath);
restoreActiveModel();

const allowedOrigins = (process.env.FRAUDSHIELD_ALLOWED_ORIGINS || '').split(',').map((value) => value.trim()).filter(Boolean);
app.disable('x-powered-by');
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  if (production) res.setHeader('Cache-Control', 'no-store');
  next();
});
app.use(cors({ credentials: true, allowedHeaders: ['Content-Type', 'X-Requested-With', 'Authorization'], origin(origin, callback) {
  if (!origin) return callback(null, false);
  if (!production && /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)) return callback(null, true);
  return callback(null, allowedOrigins.includes(origin));
} }));
const hits = new Map();
const loginHits = new Map();
const authExempt = (req) => (req.method === 'GET' && req.path === '/auth/session') || (req.method === 'POST' && ['/auth/login', '/auth/logout'].includes(req.path));
const trustedBrowserOrigin = (req) => {
  const origin = req.get('origin');
  if (!origin) return false;
  return allowedOrigins.includes(origin) || origin === `https://${req.get('host')}`;
};
app.use('/api', (req, res, next) => {
  const now = Date.now(); const key = req.socket.remoteAddress || 'unknown';
  const current = hits.get(key);
  if (!current || now - current.start >= 60_000) hits.set(key, { start: now, count: 1 });
  else if (++current.count > 300) return res.status(429).json({ ok: false, error: 'Request rate limit exceeded.' });
  if (hits.size > 10_000) for (const [ip, value] of hits) if (now - value.start >= 60_000) hits.delete(ip);
  if (req.path === '/health' && req.method === 'GET') return next();
  if (req.path === '/auth/login' && req.method === 'POST') {
    const login = loginHits.get(key);
    if (!login || now - login.start >= 60_000) loginHits.set(key, { start: now, count: 1 });
    else if (++login.count > 10) return res.status(429).json({ ok: false, error: 'Too many login attempts. Wait one minute and try again.' });
  }
  if (production && req.path === '/auth/logout' && req.method === 'POST' && (req.get('x-requested-with') !== 'XMLHttpRequest' || !trustedBrowserOrigin(req))) {
    return res.status(403).json({ ok: false, error: 'Logout request failed browser-origin validation.' });
  }
  if (authExempt(req)) return next();
  if (!production) return next();
  const cookieRole = sessionRole(req);
  const bearerRole = authRole(req);
  const role = cookieRole || bearerRole;
  if (!role) return res.status(401).json({ ok: false, error: 'Authentication required.' });
  if (cookieRole && !bearerRole && !['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
    if (req.get('x-requested-with') !== 'XMLHttpRequest' || !trustedBrowserOrigin(req)) return res.status(403).json({ ok: false, error: 'Browser session request failed CSRF validation.' });
  }
  const required = requiredRole(req.method, req.originalUrl.split('?')[0]);
  if (required === 'admin' && role !== 'admin') return res.status(403).json({ ok: false, error: 'Administrator role required.' });
  req.authRole = role;
  next();
});
app.use(express.json({ limit: '10mb', strict: true }));

const persistIfReady = () => { try { persistActiveModel(); } catch (error) { console.error('Active-model persistence failed:', error.message); throw error; } };

app.get('/api/auth/session', (req, res) => {
  if (!production) return res.json({ ok: true, authenticated: true, role: 'admin', mode: 'local-development' });
  const role = sessionRole(req) || authRole(req);
  res.json({ ok: true, authenticated: Boolean(role), role: role || null, mode: 'production-session' });
});
app.post('/api/auth/login', (req, res) => {
  if (!production) return res.json({ ok: true, authenticated: true, role: 'admin', mode: 'local-development' });
  if (req.get('x-requested-with') !== 'XMLHttpRequest' || !trustedBrowserOrigin(req)) return res.status(403).json({ ok: false, error: 'Login request failed browser-origin validation.' });
  const role = roleForToken(req.body?.accessToken);
  if (!role) return res.status(401).json({ ok: false, error: 'Invalid access token.' });
  const session = createSession(role);
  res.setHeader('Set-Cookie', sessionCookie(session.id, true));
  res.json({ ok: true, authenticated: true, role: session.role, expires_at: session.expires_at, mode: 'production-session' });
});
app.post('/api/auth/logout', (req, res) => {
  revokeSession(req);
  res.setHeader('Set-Cookie', expiredSessionCookie(production));
  res.json({ ok: true, authenticated: false });
});

app.get('/api/health', (_req, res) => res.json({ ok: true, service: 'fraudshield-z-api', model_version: MODEL_VERSION, deterministic_seed: DETERMINISTIC_SEED, model_status: process.env.FRAUDSHIELD_MODEL_DISABLED === '1' ? 'unavailable' : hasTrainedModel() ? 'trained' : 'untrained', deployment_mode: production ? 'protected-reverse-proxy' : 'local-only' }));
app.get('/api/scenarios', (_req, res) => res.json({ ok: true, scenarios: SCENARIOS.map((scenario) => ({ ...scenario, score: decideTransaction(scenario.transaction) })) }));
app.get('/api/ml/model', (_req, res) => res.json({ ok: true, trained: hasTrainedModel(), max_rows: MAX_ROWS, model: getTrainedModel(), models: getTrainedModels() }));

app.post('/api/ml/preview', (req, res) => {
  try {
    if (!req.body?.content) return res.status(400).json({ ok: false, error: 'Dataset content is required.' });
    res.json({ ok: true, preview: previewDataset(req.body.content, req.body.labelColumn) });
  } catch (error) { res.status(400).json({ ok: false, error: error.message }); }
});
app.post('/api/ml/train', (req, res) => {
  try {
    if (!req.body?.content) return res.status(400).json({ ok: false, error: 'Dataset content is required.' });
    const result = trainDataset(req.body.content, { labelColumn: req.body.labelColumn, timeColumn: req.body.timeColumn, algorithms: req.body.algorithms, costMatrix: req.body.costMatrix, exploratoryTestComparison: Boolean(req.body.exploratoryTestComparison) });
    persistIfReady();
    res.json(result);
  } catch (error) { res.status(400).json({ ok: false, error: error.message }); }
});
app.post('/api/ml/predict', (req, res) => {
  try {
    const result = predictRow(req.body?.row ?? req.body, req.body?.modelId, { observe: Boolean(req.body?.observe) });
    if (req.body?.observe && result.ok && result.model_id === getTrainedModel()?.model_id) persistIfReady();
    res.json(result);
  }
  catch (error) { res.status(400).json({ ok: false, error: error.message }); }
});
app.post('/api/ml/replay', (req, res) => {
  try {
    if (!req.body?.content) return res.status(400).json({ ok: false, error: 'Replay dataset content is required.' });
    res.json(replayDataset(req.body.content, { modelId: req.body.modelId, labelColumn: req.body.labelColumn, timeColumn: req.body.timeColumn, limit: req.body.limit }));
  } catch (error) { res.status(400).json({ ok: false, error: error.message }); }
});
app.get('/api/ml/export', (req, res) => { try { res.json({ ok: true, artifact: exportModelArtifact(req.query.modelId) }); } catch (error) { res.status(400).json({ ok: false, error: error.message }); } });
app.post('/api/ml/import', (req, res) => { try { const result = importModelArtifact(req.body?.artifact ?? req.body); persistIfReady(); res.json(result); } catch (error) { res.status(400).json({ ok: false, error: error.message }); } });
app.post('/api/ml/retrain/candidate', (req, res) => {
  try {
    if (!req.body?.content) return res.status(400).json({ ok: false, error: 'Candidate dataset content is required.' });
    res.json({ ...trainDataset(req.body.content, { labelColumn: req.body.labelColumn, timeColumn: req.body.timeColumn, algorithms: req.body.algorithms, costMatrix: req.body.costMatrix, commit: false }), approval_required: true, replacement_policy: 'Candidate training never replaces the active model until /api/ml/retrain/approve is called.' });
  } catch (error) { res.status(400).json({ ok: false, error: error.message }); }
});
app.get('/api/ml/retrain/status', (_req, res) => res.json({ ok: true, staged: getStagedTraining(), approval_required: true }));
app.post('/api/ml/retrain/approve', (_req, res) => { const result = approveStagedTraining(); if (result.ok) persistIfReady(); res.status(result.ok ? 200 : 409).json(result); });

app.post('/api/ml/audit', (req, res) => {
  const { row, action, modelId } = req.body ?? {};
  if (!row || !['confirmed_fraud', 'marked_legitimate'].includes(action)) return res.status(400).json({ ok: false, error: 'row and a supported action are required.' });
  const score = decideTransaction(row, { modelId });
  const transaction = { ...row, transaction_id: row.transaction_id || `ml_txn_${Date.now()}` };
  res.status(201).json({ ok: true, entry: recordFeedback({ transaction, action, score }) });
});
app.post('/api/score', (req, res) => {
  const result = decideTransaction(req.body, { observe: Boolean(req.body?.observe) });
  if (req.body?.observe && result.model_id === getTrainedModel()?.model_id) persistIfReady();
  if (!result.ok) return res.status(400).json(result);
  res.json(result);
});
app.post('/api/what-if', (req, res) => { const result = decideTransaction(req.body?.transaction ?? req.body); res.json({ ...result, what_if: true }); });
app.post('/api/feedback', (req, res) => {
  const { transaction, action } = req.body ?? {};
  if (!transaction || !['confirmed_fraud', 'marked_legitimate'].includes(action)) return res.status(400).json({ ok: false, error: 'transaction and a supported action are required.' });
  const score = decideTransaction(transaction);
  const entry = recordFeedback({ transaction, action, score });
  res.status(201).json({ ok: true, entry });
});
app.get('/api/audit', (_req, res) => res.json({ ok: true, entries: listAudit() }));
app.get('/api/metrics', (_req, res) => {
  const model = getTrainedModel();
  res.json({
    ok: true, environment: 'prototype; no independent real-world benchmark configured',
    evaluation_status: model ? 'uploaded_dataset_evaluation_only' : 'not_evaluated',
    model_version: model?.model_version || MODEL_VERSION, deterministic_seed: DETERMINISTIC_SEED,
    metrics: model?.metrics ?? null, metrics_source: model?.metrics_source ?? null,
    model_card: MODEL_CARD,
    benchmark_note: 'No independent real-world fraud benchmark has been evaluated. Uploaded-dataset metrics apply only to that dataset and its documented split; they are not a real-world accuracy claim.'
  });
});
app.get('/api/segments', (_req, res) => res.json({ ok: true, environment: 'prototype', evaluation_status: 'segment_metrics_not_evaluated', segments: null, note: 'No independently validated segment-level fraud metrics are configured.' }));
app.get('/api/stream', (req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache, no-store', Connection: 'keep-alive', 'X-Accel-Buffering': 'no' });
  let index = 0;
  const send = () => {
    const scenario = SCENARIOS[index % SCENARIOS.length];
    const payload = { ...scenario, score: decideTransaction(scenario.transaction), stream_index: index, emitted_at: new Date().toISOString() };
    res.write(`id: stream_${index}\ndata: ${JSON.stringify(payload)}\n\n`);
    index += 1;
  };
  send();
  const timer = setInterval(send, 5200);
  req.on('close', () => clearInterval(timer));
});
app.get('/api/validate', (_req, res) => res.json({ ok: true, sample_errors: validateTransaction({}) }));

const distPath = path.resolve(__dirname, '../dist');
if (production) {
  app.use(express.static(distPath, { dotfiles: 'deny', index: false, maxAge: '1h' }));
  app.get('*', (_req, res) => res.sendFile(path.join(distPath, 'index.html')));
}
app.use((error, _req, res, _next) => {
  if (error?.type === 'entity.too.large') return res.status(413).json({ ok: false, error: 'Request body exceeds the 10 MB limit.' });
  console.error('Unhandled request error:', error?.message || error);
  res.status(500).json({ ok: false, error: 'Internal server error.' });
});

const host = validateBindHost(process.env.FRAUDSHIELD_BIND_HOST || '127.0.0.1');
app.listen(PORT, host, () => console.log(`FraudShield Z API listening on ${host}:${PORT}${production ? ' (reverse-proxy protected)' : ' (local-only)'}`));

export { app };
