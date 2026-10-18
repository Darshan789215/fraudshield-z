import { BASELINES, DETERMINISTIC_SEED, MODEL_CARD, MODEL_VERSION, SCENARIOS, THRESHOLDS } from '../shared/contracts.js';

const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));
const round = (value, digits = 3) => Number(value.toFixed(digits));

export function baselineFor(customerId) {
  return BASELINES[customerId] ?? {
    usual_amount: 100,
    known_devices: [],
    known_locations: [],
    common_categories: [],
    usual_velocity_10m: 1,
    customer_tenure_days: 0,
    is_new_customer: true
  };
}

export function validateTransaction(input = {}) {
  const errors = [];
  const required = ['transaction_id', 'customer_id', 'amount', 'device_id', 'location', 'merchant_category', 'timestamp'];
  for (const field of required) if (input[field] === undefined || input[field] === '') errors.push({ field, message: 'Required field is missing.' });
  if (input.amount !== undefined && (!Number.isFinite(Number(input.amount)) || Number(input.amount) < 0)) errors.push({ field: 'amount', message: 'Amount must be a non-negative number.' });
  if (input.velocity_10m !== undefined && (!Number.isFinite(Number(input.velocity_10m)) || Number(input.velocity_10m) < 0)) errors.push({ field: 'velocity_10m', message: 'Velocity must be a non-negative number.' });
  if (input.timestamp && Number.isNaN(Date.parse(input.timestamp))) errors.push({ field: 'timestamp', message: 'Timestamp must be ISO-8601 compatible.' });
  return errors;
}

function featuresFor(transaction) {
  const baseline = baselineFor(transaction.customer_id);
  const amount = Number(transaction.amount);
  const velocity = Number(transaction.velocity_10m ?? 1);
  const hour = new Date(transaction.timestamp).getUTCHours();
  return {
    amount_ratio: amount / Math.max(baseline.usual_amount, 1),
    device_new: baseline.known_devices.length === 0 || !baseline.known_devices.includes(transaction.device_id),
    location_new: baseline.known_locations.length === 0 || !baseline.known_locations.includes(transaction.location),
    category_new: baseline.common_categories.length === 0 || !baseline.common_categories.includes(transaction.merchant_category),
    velocity_10m: velocity,
    velocity_delta: Math.max(0, velocity - baseline.usual_velocity_10m),
    unusual_hour: hour < 5 || hour > 23,
    hour,
    baseline
  };
}

function factor(label, value, weight, tone, detail) {
  return { label, value: round(value), weight: round(weight), tone, detail };
}

export function scoreTransaction(transaction, options = {}) {
  const started = performance.now();
  const errors = validateTransaction(transaction);
  const modelAvailable = options.modelAvailable !== false && process.env.FRAUDSHIELD_MODEL_DISABLED !== '1';
  if (errors.length) return { ok: false, errors };
  if (!modelAvailable) {
    return {
      ok: true,
      model_failure: true,
      transaction_id: transaction.transaction_id,
      decision: 'review_required',
      decision_label: 'Review required',
      risk_score: null,
      explanation: 'Automated scoring is unavailable. Route this transaction to a human reviewer instead of silently approving it.',
      explanation_method: 'fallback state — no model explanation available',
      explanation_limitation: 'Feature contributions describe model association and do not establish causation.',
      top_factors: [],
      thresholds: THRESHOLDS,
      model_version: MODEL_VERSION,
      deterministic_seed: DETERMINISTIC_SEED,
      latency_ms: round(performance.now() - started, 2),
      decision_trace: [
        { key: 'event', label: 'Transaction received', status: 'complete' },
        { key: 'model', label: 'Model unavailable', status: 'blocked' },
        { key: 'action', label: 'Human review required', status: 'active' }
      ]
    };
  }

  const f = featuresFor(transaction);
  const amountSignal = clamp(Math.log10(Math.max(f.amount_ratio, 1)) / 1.45);
  const deviceSignal = f.device_new ? 0.25 : 0;
  const locationSignal = f.location_new ? 0.18 : 0;
  const categorySignal = f.category_new ? 0.07 : 0;
  const velocitySignal = clamp(f.velocity_delta / 5) * 0.22;
  const hourSignal = f.unusual_hour ? 0.06 : 0;
  const supervised = clamp(0.08 + amountSignal * 0.44 + deviceSignal + locationSignal + categorySignal + velocitySignal + hourSignal);
  const novelty = clamp(0.05 + amountSignal * 0.33 + (f.device_new ? 0.22 : 0) + (f.location_new ? 0.18 : 0) + (f.category_new ? 0.08 : 0) + clamp(f.velocity_delta / 6) * 0.19);
  const fused = clamp(supervised * 0.62 + novelty * 0.38);
  const decision = fused < THRESHOLDS.approveMax ? 'approve' : fused < THRESHOLDS.stepUpMax ? 'step_up' : 'hold';
  const decisionLabel = decision === 'approve' ? 'Approve' : decision === 'step_up' ? 'Step-up authentication' : 'Hold for investigation';
  const factors = [
    factor('Amount vs baseline', f.amount_ratio, amountSignal * 0.44, f.amount_ratio > 4 ? 'danger' : f.amount_ratio > 2 ? 'warning' : 'neutral', `The amount is approximately ${round(f.amount_ratio, 1)}× the customer’s usual amount.`),
    factor('Device familiarity', f.device_new ? 1 : 0, deviceSignal, f.device_new ? 'danger' : 'positive', f.device_new ? 'The device has not been seen in the customer’s recent history.' : 'The device is familiar to this customer.'),
    factor('Location familiarity', f.location_new ? 1 : 0, locationSignal, f.location_new ? 'warning' : 'positive', f.location_new ? 'The location is new for this customer.' : 'The location matches the customer baseline.'),
    factor('Velocity / 10 min', f.velocity_10m, velocitySignal, f.velocity_delta > 2 ? 'danger' : f.velocity_delta > 0 ? 'warning' : 'positive', `${f.velocity_10m} transaction${f.velocity_10m === 1 ? '' : 's'} occurred during the last 10 minutes.`),
    factor('Merchant category', f.category_new ? 1 : 0, categorySignal, f.category_new ? 'warning' : 'positive', f.category_new ? 'This merchant category is not common for the customer.' : 'This merchant category is familiar.'),
    factor('Anomaly evidence', novelty, novelty * 0.38, novelty > 0.65 ? 'danger' : novelty > 0.35 ? 'warning' : 'neutral', 'The combined behavior is unusual for this customer and the observed population.')
  ].sort((a, b) => b.weight - a.weight);

  const explanation = decision === 'approve'
    ? 'The transaction aligns with the customer’s usual behavior and remains below the friction threshold.'
    : decision === 'step_up'
      ? 'The evidence is mixed. Ask for additional authentication before completing the transaction.'
      : 'Several independent signals combine into a high-risk pattern. Hold the transaction for investigation.';

  return {
    ok: true,
    model_failure: false,
    transaction_id: transaction.transaction_id,
    customer_id: transaction.customer_id,
    decision,
    decision_label: decisionLabel,
    risk_score: round(fused),
    supervised_score: round(supervised),
    anomaly_score: round(novelty),
    fusion_method: 'Validation-selected weighted calibrated fusion (prototype)',
    thresholds: THRESHOLDS,
    threshold: decision === 'approve' ? THRESHOLDS.approveMax : decision === 'step_up' ? THRESHOLDS.stepUpMax : 1,
    top_factors: factors.slice(0, 5),
    customer_baseline: {
      usual_amount: f.baseline.usual_amount,
      known_devices: f.baseline.known_devices,
      known_locations: f.baseline.known_locations,
      usual_velocity_10m: f.baseline.usual_velocity_10m,
      customer_tenure_days: f.baseline.customer_tenure_days
    },
    model_card: MODEL_CARD,
    explanation,
    explanation_method: 'deterministic model factor attribution',
    explanation_limitation: 'Feature contributions describe model association and do not establish causation.',
    latency_ms: round(Math.max(0.4, performance.now() - started), 2),
    deterministic_seed: DETERMINISTIC_SEED,
    model_version: MODEL_VERSION,
    decision_trace: [
      { key: 'event', label: 'Transaction received', status: 'complete', value: transaction.merchant_category },
      { key: 'baseline', label: 'Customer baseline enriched', status: 'complete', value: `${round(f.amount_ratio, 1)}× amount ratio` },
      { key: 'supervised', label: 'Supervised evidence', status: 'complete', value: `${round(supervised * 100)}%` },
      { key: 'anomaly', label: 'Novelty evidence', status: novelty > 0.6 ? 'alert' : 'complete', value: `${round(novelty * 100)}% unusual` },
      { key: 'fusion', label: 'Risk fusion + thresholds', status: 'complete', value: `${round(fused * 100)} / 100` },
      { key: 'action', label: decisionLabel, status: decision === 'hold' ? 'alert' : decision === 'step_up' ? 'warning' : 'complete' }
    ]
  };
}

export function scenarios() {
  return SCENARIOS.map((scenario) => ({ ...scenario, score: scoreTransaction(scenario.transaction) }));
}
