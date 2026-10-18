import { getModelScoringSchema, hasTrainedModel, predictRow } from './ml.js';

const aliases = {
  customer: ['customer_id', 'customer', 'user_id', 'account_id', 'card_id', 'client_id'],
  amount: ['amount', 'transaction_amount', 'value', 'spend'],
  device: ['device_id', 'device', 'browser_id', 'fingerprint'],
  merchant: ['merchant_category', 'merchant', 'seller', 'store', 'category'],
  time: ['timestamp', 'transaction_time', 'datetime', 'date', 'time'],
};
const pick = (source, names) => {
  for (const name of names) if (source[name] !== undefined) return source[name];
  const wanted = new Set(names.map((name) => name.toLowerCase()));
  const match = Object.keys(source).find((key) => wanted.has(key.toLowerCase()));
  return match === undefined ? undefined : source[match];
};

function mapTransactionToModel(transaction, schema) {
  const row = { ...transaction };
  const config = schema.feature_config;
  const values = {
    [config.customer_column]: pick(transaction, aliases.customer),
    [config.amount_column]: pick(transaction, aliases.amount),
    [config.device_column]: pick(transaction, aliases.device),
    [config.merchant_column]: pick(transaction, aliases.merchant),
    [config.time_column]: pick(transaction, aliases.time),
  };
  for (const [column, value] of Object.entries(values)) if (column && value !== undefined) row[column] = value;
  for (const column of schema.input_columns || []) {
    if (row[column] !== undefined) continue;
    const value = pick(transaction, [column]);
    if (value !== undefined) row[column] = value;
  }
  return row;
}

/**
 * Shared decision path for the workbench, legacy demo scoring and feedback.
 * The trained classifier is fused with five temporal/relationship signals using
 * the exact formula and validation-cost threshold stored at model training time.
 * No model or scoring failure is fail-closed as human review, never approval.
 */
export function decideTransaction(transaction, { modelId, observe = false } = {}) {
  if (process.env.FRAUDSHIELD_MODEL_DISABLED === '1' || !hasTrainedModel()) {
    return { ok: true, decision: 'review_required', decision_label: 'Review required', model_failure: true,
      risk_score: null, explanation: 'No validated trained model is active. Route this transaction to an investigator; do not approve automatically.',
      fallback: 'review_required / no approval', decision_service: 'fraudshield-hybrid-v1' };
  }
  const schema = getModelScoringSchema(modelId);
  if (!schema) return { ok: true, decision: 'review_required', decision_label: 'Review required', model_failure: true, fallback: 'review_required / no approval' };
  const row = mapTransactionToModel(transaction || {}, schema);
  const result = predictRow(row, schema.model_id, { explain: true, observe });
  if (!result.ok) return { ok: true, decision: 'review_required', decision_label: 'Review required', model_failure: true,
    risk_score: null, error: result.error, fallback: 'review_required / no approval', decision_service: 'fraudshield-hybrid-v1' };
  const decision = result.decision === 'approve' ? 'approve' : 'hold';
  return {
    ...result,
    ok: true,
    decision,
    decision_label: decision === 'approve' ? 'Approve' : 'Hold for investigation',
    risk_score: result.hybrid_risk_score,
    threshold: result.hybrid_threshold,
    classifier_score: result.risk_score,
    action: decision === 'approve' ? 'approve' : 'human_review',
    transaction_id: transaction?.transaction_id,
    selected_risk_signals: result.derived_features,
    decision_service: 'fraudshield-hybrid-v1',
    fallback: 'classifier or feature failure routes to review_required; no automatic approval',
  };
}
