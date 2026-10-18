import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

const auditEntries = [];
let auditPath = process.env.FRAUDSHIELD_AUDIT_PATH || null;
let previousHash = 'GENESIS';

function canonical(entry) { return JSON.stringify(entry); }
function entryHash(entry) { return createHash('sha256').update(`${entry.previous_hash}\n${canonical(Object.fromEntries(Object.entries(entry).filter(([key]) => key !== 'entry_hash')))}`).digest('hex'); }

export function configureAuditStore(filePath = auditPath) {
  auditPath = filePath;
  if (!auditPath || !fs.existsSync(auditPath)) return;
  const lines = fs.readFileSync(auditPath, 'utf8').split('\n').filter(Boolean);
  auditEntries.length = 0;
  previousHash = 'GENESIS';
  for (const line of lines) {
    const entry = JSON.parse(line);
    if (entry.previous_hash !== previousHash || entry.entry_hash !== entryHash(entry)) throw new Error('Audit log hash-chain verification failed; refuse to serve a modified or corrupt audit log.');
    auditEntries.unshift(entry);
    previousHash = entry.entry_hash;
  }
  if (fs.existsSync(auditPath)) fs.chmodSync(auditPath, 0o600);
}

export function recordFeedback({ transaction, action, score }) {
  const entry = {
    audit_id: `audit_${Date.now()}_${auditEntries.length + 1}`,
    transaction_id: transaction.transaction_id,
    action,
    recorded_at: new Date().toISOString(),
    original_decision: score.decision ?? score.prediction,
    risk_score: score.risk_score,
    hybrid_risk_score: score.hybrid_risk_score,
    model_version: score.model_version,
    decision_threshold: score.hybrid_threshold ?? score.threshold ?? null,
    explanation_summary: score.explanation || score.explanation_summary || 'No explanation was available.',
    explanation_method: score.explanation?.method || score.explanation_method || 'temporal/relationship feature signal plus classifier score',
    final_action: action,
    causality_note: 'Feature contributions describe model association and do not establish causation.',
    investigator_label: action === 'confirmed_fraud' ? 'Confirmed fraud' : 'Marked legitimate',
    previous_hash: previousHash,
  };
  entry.entry_hash = entryHash(entry);
  if (auditPath) {
    fs.mkdirSync(path.dirname(auditPath), { recursive: true, mode: 0o700 });
    fs.chmodSync(path.dirname(auditPath), 0o700);
    fs.appendFileSync(auditPath, `${JSON.stringify(entry)}\n`, { mode: 0o600 });
    fs.chmodSync(auditPath, 0o600);
  }
  previousHash = entry.entry_hash;
  auditEntries.unshift(entry);
  return entry;
}

export function listAudit() {
  return auditEntries.slice(0, 20);
}
