import test from 'node:test';
import assert from 'node:assert/strict';
import { SCENARIOS, MODEL_VERSION, DETERMINISTIC_SEED } from '../shared/contracts.js';
import { scoreTransaction } from '../server/engine.js';
import { approveStagedTraining, exportModelArtifact, getStagedTraining, importModelArtifact, predictRow, previewDataset, replayDataset, trainDataset } from '../server/ml.js';
import { decideTransaction } from '../server/decision.js';
import { authRole, createSession, expiredSessionCookie, requiredRole, revokeSession, sessionCookie, sessionRole, SESSION_TTL_SECONDS, validateBindHost, validateProductionSecurity } from '../server/security.js';
import { createEncoder, countInvalidEncodings, encodeRow } from '../server/ml/encoder.js';
import { BEHAVIOR_FEATURES, buildFeatures, computeFeatures, createFeatureState, detectFeatureConfig, observe } from '../server/ml/features.js';
import { chooseThreshold, normalizeCostMatrix } from '../server/ml/metrics.js';
import { smoteNC } from '../server/ml/smote.js';
import { login as frontendLogin, logout as frontendLogout } from '../src/lib/api.js';

test('normal scenario approves deterministically', () => {
  const first = scoreTransaction(SCENARIOS[0].transaction);
  const second = scoreTransaction(SCENARIOS[0].transaction);
  assert.equal(first.decision, 'approve');
  assert.equal(first.risk_score, second.risk_score);
  assert.equal(first.model_version, MODEL_VERSION);
  assert.equal(first.deterministic_seed, DETERMINISTIC_SEED);
});

test('borderline scenario requests step-up authentication', () => {
  const result = scoreTransaction(SCENARIOS[1].transaction);
  assert.equal(result.decision, 'step_up');
  assert.ok(result.decision_trace.some((item) => item.key === 'fusion'));
});

test('suspicious scenario is held with human-readable evidence', () => {
  const result = scoreTransaction(SCENARIOS[2].transaction);
  assert.equal(result.decision, 'hold');
  assert.ok(result.top_factors.length >= 3);
  assert.match(result.top_factors[0].detail, /amount|device|location|transaction|behavior/i);
});

test('model failure returns review_required instead of approval', () => {
  const result = scoreTransaction(SCENARIOS[2].transaction, { modelAvailable: false });
  assert.equal(result.decision, 'review_required');
  assert.equal(result.model_failure, true);
  assert.ok(result.decision_trace.some((item) => item.status === 'blocked'));
});

test('production requires strong distinct role tokens and protects administrator routes', () => {
  assert.equal(validateBindHost('127.0.0.1', { NODE_ENV: 'development' }), '127.0.0.1');
  assert.throws(() => validateBindHost('0.0.0.0', { NODE_ENV: 'development' }), /local-only/);
  assert.equal(validateBindHost('0.0.0.0', { NODE_ENV: 'production' }), '0.0.0.0');
  assert.throws(() => validateProductionSecurity({ NODE_ENV: 'production', FRAUDSHIELD_TLS_TERMINATED: '1' }), /distinct/);
  assert.throws(() => validateProductionSecurity({ NODE_ENV: 'production', FRAUDSHIELD_ADMIN_TOKEN: 'a'.repeat(32), FRAUDSHIELD_INVESTIGATOR_TOKEN: 'b'.repeat(32) }), /TLS termination/);
  assert.doesNotThrow(() => validateProductionSecurity({ NODE_ENV: 'production', FRAUDSHIELD_TLS_TERMINATED: '1', FRAUDSHIELD_ADMIN_TOKEN: 'a'.repeat(32), FRAUDSHIELD_INVESTIGATOR_TOKEN: 'b'.repeat(32) }));
  assert.equal(requiredRole('POST', '/api/ml/train'), 'admin');
  assert.equal(requiredRole('POST', '/api/ml/import'), 'admin');
  assert.equal(requiredRole('POST', '/api/feedback'), 'investigator');
  const priorAdmin = process.env.FRAUDSHIELD_ADMIN_TOKEN;
  const priorInvestigator = process.env.FRAUDSHIELD_INVESTIGATOR_TOKEN;
  try {
    process.env.FRAUDSHIELD_ADMIN_TOKEN = 'admin-test-token';
    process.env.FRAUDSHIELD_INVESTIGATOR_TOKEN = 'investigator-test-token';
    assert.equal(authRole({ get: () => 'Bearer investigator-test-token' }), 'investigator');
    assert.equal(authRole({ get: () => 'Bearer admin-test-token' }), 'admin');
    assert.equal(authRole({ get: () => 'Bearer nope' }), null);
  } finally {
    if (priorAdmin === undefined) delete process.env.FRAUDSHIELD_ADMIN_TOKEN; else process.env.FRAUDSHIELD_ADMIN_TOKEN = priorAdmin;
    if (priorInvestigator === undefined) delete process.env.FRAUDSHIELD_INVESTIGATOR_TOKEN; else process.env.FRAUDSHIELD_INVESTIGATOR_TOKEN = priorInvestigator;
  }
});

test('cookie sessions are role-scoped, expire, revoke, and use HttpOnly/SameSite/Secure flags', () => {
  const issued = createSession('investigator', 1000);
  const request = { get: (name) => name === 'cookie' ? `fraudshield_session=${issued.id}` : '' };
  assert.equal(sessionRole(request, 1001), 'investigator');
  assert.equal(new Date(issued.expires_at).getTime(), 1000 + SESSION_TTL_SECONDS * 1000);
  assert.match(sessionCookie(issued.id, true), /HttpOnly/);
  assert.match(sessionCookie(issued.id, true), /SameSite=Strict/);
  assert.match(sessionCookie(issued.id, true), /; Secure/);
  assert.match(expiredSessionCookie(true), /Max-Age=0/);
  assert.equal(sessionRole(request, 1000 + SESSION_TTL_SECONDS * 1000 + 1), null);
  const active = createSession('admin', 2000);
  const activeRequest = { get: (name) => name === 'cookie' ? `fraudshield_session=${active.id}` : '' };
  revokeSession(activeRequest);
  assert.equal(sessionRole(activeRequest, 2001), null);
});

test('frontend auth requests use same-origin cookies and CSRF headers without bearer headers', async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return new Response(JSON.stringify({ ok: true }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
  try {
    await frontendLogin('one-time-access-token');
    await frontendLogout();
    assert.deepEqual(calls.map((call) => call.url), ['/api/auth/login', '/api/auth/logout']);
    for (const { options } of calls) {
      assert.equal(options.credentials, 'same-origin');
      assert.equal(options.headers['X-Requested-With'], 'XMLHttpRequest');
      assert.equal(options.headers.Authorization, undefined);
    }
    assert.deepEqual(JSON.parse(calls[0].options.body), { accessToken: 'one-time-access-token' });
  } finally { globalThis.fetch = originalFetch; }
});

test('invalid input returns labeled errors', () => {
  const result = scoreTransaction({});
  assert.equal(result.ok, false);
  assert.ok(result.errors.some((error) => error.field === 'amount'));
});


/* ---------------------------------------------------------------- ML helpers */
function syntheticDataset(count = 240, seed = 11) {
  let state = seed;
  const random = () => { state = (state * 1664525 + 1013904223) >>> 0; return state / 4294967296; };
  const merchants = ['grocery', 'fuel', 'travel', 'electronics', 'dining'];
  const rows = ['timestamp,customer_id,device_id,amount,merchant_category,is_fraud'];
  let time = Date.parse('2026-02-01T00:00:00Z');
  for (let i = 0; i < count; i += 1) {
    time += 60000 + Math.floor(random() * 240000);
    const customer = `c${Math.floor(random() * 12)}`;
    const fraud = random() < 0.12;
    rows.push([new Date(time).toISOString(), customer, fraud ? `d${Math.floor(random() * 3)}` : `dev_${customer}`, (fraud ? 220 + random() * 600 : 8 + random() * 120).toFixed(2), fraud ? merchants[2 + Math.floor(random() * 3)] : merchants[Math.floor(random() * 5)], fraud ? 1 : 0].join(','));
  }
  return { csv: rows.join('\n'), lastTime: time };
}
const { csv: DATA, lastTime: DATA_END } = syntheticDataset();

test('uploaded CSV is previewed, trained, and used for prediction', () => {
  const csv = `amount,velocity,merchant,is_fraud\n20,1,groceries,0\n30,1,groceries,0\n40,1,travel,0\n50,2,travel,0\n80,4,electronics,1\n120,5,electronics,1\n180,6,electronics,1\n250,8,electronics,1\n60,2,groceries,0\n90,5,electronics,1\n72,1,fuel,0\n310,7,electronics,1\n`;
  const preview = previewDataset(csv);
  assert.equal(preview.label_column, 'is_fraud');
  assert.equal(preview.row_count, 12);
  const trained = trainDataset(csv);
  assert.equal(trained.ok, true);
  assert.equal(trained.models.length, 5);
  assert.deepEqual(trained.models.map((model) => model.algorithm).sort(), ['decision_tree', 'gaussian_nb', 'gradient_boosted_stumps', 'logistic_regression', 'random_forest']);
  assert.equal(trained.model.evaluation_protocol.split, 'chronological 60% train / 20% validation / 20% test');
  assert.ok('pr_auc' in trained.model.metrics);
  assert.ok(trained.model.metrics.operational.prediction_latency_ms.p95 >= 0);
  const prediction = predictRow({ amount: '220', velocity: '7', merchant: 'electronics' });
  assert.equal(prediction.ok, true);
  assert.ok(prediction.risk_score >= 0 && prediction.risk_score <= 1);
});

/* Issue 1 ---------------------------------------------------------------- */
test('model selection uses validation results only; the test set is scored once for the winner', () => {
  const trained = trainDataset(DATA);
  const byValidation = [...trained.models].sort((a, b) => (b.selection_metrics.pr_auc ?? -1) - (a.selection_metrics.pr_auc ?? -1) || a.selection_metrics.cost.total_cost - b.selection_metrics.cost.total_cost || a.algorithm.localeCompare(b.algorithm));
  assert.equal(trained.selected_model_id, byValidation[0].model_id);
  assert.equal(trained.selection.data_used, 'validation only');
  const selected = trained.models.find((model) => model.model_id === trained.selected_model_id);
  assert.equal(selected.evaluation_role, 'selected_final_test_evaluation');
  assert.equal(selected.metrics_source, 'test');
  assert.equal(selected.test_metrics.evaluation, 'final_unbiased_test_evaluation');
  for (const other of trained.models.filter((model) => model !== selected)) {
    assert.equal(other.evaluation_role, 'not_selected_validation_only');
    assert.equal(other.metrics_source, 'validation');
    assert.equal(other.test_metrics, null);
    assert.equal(other.exploratory_test_metrics, null);
  }
});

test('exploratory test comparisons are labelled and never change the selection', () => {
  const plain = trainDataset(DATA);
  const exploratory = trainDataset(DATA, { exploratoryTestComparison: true });
  assert.equal(plain.model.algorithm, exploratory.model.algorithm);
  const others = exploratory.models.filter((model) => model.model_id !== exploratory.selected_model_id);
  assert.ok(others.every((model) => model.exploratory_test_metrics.evaluation.includes('do_not_use_for_selection')));
  assert.ok(others.every((model) => model.metrics_source === 'validation'));
});

/* Issue 2 ---------------------------------------------------------------- */
test('behavioural features are rebuilt from history at prediction time', () => {
  const trained = trainDataset(DATA, { algorithms: ['logistic_regression'] });
  const trainingRows = DATA.split('\n').slice(1, 1 + Math.floor(240 * 0.6));
  const known = trainingRows.at(-1).split(',');
  const customer = known[1];
  const soon = new Date(Date.parse(known[0]) + 60000).toISOString();
  const withHistory = predictRow({ timestamp: soon, customer_id: customer, device_id: 'brand_new_device', amount: 400, merchant_category: 'electronics' }, trained.model.model_id);
  assert.equal(withHistory.ok, true);
  assert.ok(withHistory.derived_features.behavior_txn_count_10m >= 1, 'customer history must feed the 10-minute count');
  assert.equal(withHistory.derived_features.behavior_new_device, 1);
  assert.ok(withHistory.derived_features.behavior_avg_spend_7d > 0);
  const stranger = predictRow({ timestamp: soon, customer_id: 'never_seen', device_id: 'x', amount: 400, merchant_category: 'electronics' }, trained.model.model_id);
  assert.equal(stranger.derived_features.behavior_txn_count_10m, 0);
  assert.notEqual(withHistory.risk_score, undefined);
});

test('caller-supplied derived values are ignored and scoring is read-only unless observe is set', () => {
  const trained = trainDataset(DATA, { algorithms: ['logistic_regression'] });
  const id = trained.model.model_id;
  const row = { timestamp: new Date(DATA_END + 1000).toISOString(), customer_id: 'c1', device_id: 'dev_c1', amount: 50, merchant_category: 'fuel' };
  const first = predictRow(row, id);
  const forged = predictRow({ ...row, behavior_txn_count_10m: 9999 }, id);
  assert.equal(first.risk_score, forged.risk_score);
  assert.deepEqual(first.derived_features, predictRow(row, id).derived_features);
  const before = predictRow(row, id).derived_features.graph_account_merchant_repeat;
  predictRow(row, id, { observe: true });
  assert.equal(predictRow(row, id).derived_features.graph_account_merchant_repeat, before + 1);
});

test('one pipeline: batch feature building equals row-by-row compute + observe', () => {
  const headers = ['timestamp', 'customer_id', 'device_id', 'amount', 'merchant_category', 'is_fraud'];
  const config = detectFeatureConfig(headers, { labelColumn: 'is_fraud', timeColumn: 'timestamp' });
  const lines = DATA.split('\n').slice(1, 60).map((line, index) => { const values = line.split(','); return { row: Object.fromEntries(headers.map((h, i) => [h, values[i]])), time: Date.parse(values[0]), index }; });
  const batch = buildFeatures(config, lines);
  const state = createFeatureState();
  lines.forEach((item, index) => {
    const features = computeFeatures(config, state, item.row, item.time);
    for (const name of BEHAVIOR_FEATURES) assert.equal(batch[index].row[name], features[name], `${name} differs at row ${index}`);
    observe(config, state, item.row, item.time);
  });
});

test('features never use future rows', () => {
  const config = detectFeatureConfig(['timestamp', 'customer_id', 'amount'], { labelColumn: 'y', timeColumn: 'timestamp' });
  const state = createFeatureState();
  observe(config, state, { customer_id: 'a', amount: '100' }, 1000);
  const earlier = computeFeatures(config, state, { customer_id: 'a', amount: '100' }, 500);
  assert.equal(earlier.behavior_txn_count_10m, 0);
});

test('rows without customer identifiers never share a synthetic customer or device-owner history', () => {
  const headers = ['timestamp', 'device_id', 'merchant_category', 'amount'];
  const config = detectFeatureConfig(headers, { timeColumn: 'timestamp' });
  const state = createFeatureState();
  observe(config, state, { device_id: 'shared', merchant_category: 'shop', amount: 100 }, 1000);
  const features = computeFeatures(config, state, { device_id: 'shared', merchant_category: 'shop', amount: 200 }, 1200);
  assert.equal(features.behavior_txn_count_10m, 0);
  assert.equal(features.behavior_amount_deviation, 0);
  assert.equal(features.relationship_shared_device_count, 0);
  assert.equal(features.graph_device_degree, 0);
  assert.equal(features.graph_merchant_degree, 0);
  assert.equal(features.graph_account_merchant_repeat, 0);
  assert.equal(state.customers.size, 0);
  assert.equal(state.deviceOwners.size, 0);

  const withMissingId = detectFeatureConfig(['timestamp', 'customer_id', 'device_id', 'amount'], { timeColumn: 'timestamp' });
  const missingIdState = createFeatureState();
  observe(withMissingId, missingIdState, { customer_id: '', device_id: 'device-x', amount: 10 }, 1000);
  assert.equal(missingIdState.customers.size, 0);
  assert.equal(missingIdState.deviceOwners.size, 0);
});

test('shared-device and merchant relationship features exclude future observations', () => {
  const config = detectFeatureConfig(['timestamp', 'customer_id', 'device_id', 'merchant_category', 'amount'], { timeColumn: 'timestamp' });
  const state = createFeatureState();
  observe(config, state, { customer_id: 'future-user', device_id: 'shared', merchant_category: 'shop', amount: 10 }, 2000);
  const earlier = computeFeatures(config, state, { customer_id: 'query-user', device_id: 'shared', merchant_category: 'shop', amount: 20 }, 1000);
  assert.equal(earlier.relationship_shared_device_count, 0);
  assert.equal(earlier.relationship_repeated_merchant_count, 0);
  assert.equal(earlier.graph_device_degree, 0);
  assert.equal(earlier.graph_merchant_degree, 0);
  assert.equal(earlier.graph_account_merchant_repeat, 0);
  const later = computeFeatures(config, state, { customer_id: 'query-user', device_id: 'shared', merchant_category: 'shop', amount: 20 }, 3000);
  assert.equal(later.relationship_shared_device_count, 1);
  assert.equal(later.relationship_repeated_merchant_count, 1);
});

test('candidate models own independent mutable feature histories and start at training cutoff', () => {
  const trained = trainDataset(DATA, { algorithms: ['logistic_regression', 'decision_tree'] });
  const [first, second] = trained.models;
  const firstBefore = exportModelArtifact(first.model_id).model.feature_state;
  const secondBefore = exportModelArtifact(second.model_id).model.feature_state;
  const probe = { timestamp: new Date(DATA_END + 60000).toISOString(), customer_id: 'isolated', device_id: 'only-first', amount: 777, merchant_category: 'isolated-shop' };
  predictRow(probe, first.model_id, { observe: true });
  assert.notDeepEqual(exportModelArtifact(first.model_id).model.feature_state, firstBefore);
  assert.deepEqual(exportModelArtifact(second.model_id).model.feature_state, secondBefore);
  const trainingCutoff = first.evaluation_protocol.train_period_end;
  assert.equal(second.evaluation_protocol.train_period_end, trainingCutoff);
  assert.equal(firstBefore.last_time, trainingCutoff);
  assert.equal(exportModelArtifact(second.model_id).model.feature_state.last_time, trainingCutoff);
});

test('shared decision service combines the trained score with calibrated temporal signals', () => {
  const trained = trainDataset(DATA, { algorithms: ['logistic_regression'] });
  const row = { timestamp: new Date(DATA_END + 60000).toISOString(), customer_id: 'c2', device_id: 'new-device', amount: 900, merchant_category: 'electronics' };
  const result = decideTransaction(row);
  assert.equal(result.ok, true);
  assert.equal(result.model_id, trained.model.model_id);
  assert.equal(result.decision_service, 'fraudshield-hybrid-v1');
  assert.equal(result.risk_score, result.hybrid_risk_score);
  assert.equal(result.threshold, trained.model.hybrid_threshold);
  assert.ok(result.selected_risk_signals.behavior_new_device >= 0);
  assert.ok(trained.model.hybrid_validation_metrics.cost.total_cost >= 0);
  assert.equal(trained.model.hybrid_test_metrics.evaluation, 'final_unbiased_test_evaluation');
});

/* Issue 3 ---------------------------------------------------------------- */
test('ablation is a controlled 2x2 design with identical split, learner and threshold procedure', () => {
  const { ablation } = trainDataset(DATA).experiments;
  assert.deepEqual(ablation.variants.map((variant) => variant.variant), ['original_baseline', 'original_smote', 'behavioural_baseline', 'behavioural_smote']);
  const [a, b, c, d] = ablation.variants;
  assert.equal(a.feature_count, b.feature_count);
  assert.equal(c.feature_count, d.feature_count);
  assert.ok(c.feature_count > a.feature_count);
  assert.deepEqual([a.imbalance_method, b.imbalance_method, c.imbalance_method, d.imbalance_method], ['none', 'SMOTE-NC', 'none', 'SMOTE-NC']);
  assert.ok(ablation.variants.every((variant) => variant.metrics.rows === a.metrics.rows), 'same test set everywhere');
  assert.ok(b.synthetic_rows > 0 && d.synthetic_rows > 0 && a.synthetic_rows === 0 && c.synthetic_rows === 0);
  assert.ok(ablation.variants.every((variant) => variant.resampling.invalid_encoded_rows === 0));
  assert.deepEqual(Object.keys(ablation.contrasts), ['behavioural_features_without_resampling', 'behavioural_features_with_smote', 'smote_on_original_features', 'smote_on_behavioural_features']);
});

test('benchmark mode can skip exploratory experiments without skipping candidate test evaluation', () => {
  const trained = trainDataset(DATA, { algorithms: ['logistic_regression'], skipExploratoryExperiments: true });
  assert.equal(trained.experiments.skipped, true);
  assert.equal(trained.model.evaluation_role, 'selected_final_test_evaluation');
  assert.ok(trained.model.hybrid_test_metrics.rows > 0);
  assert.ok(trained.model.test_metrics.pr_auc >= 0);
});

/* Issue 4 ---------------------------------------------------------------- */
test('explanations: exact for logistic, sampled Shapley elsewhere, faithfulness reported separately', () => {
  const trained = trainDataset(DATA);
  const row = { timestamp: new Date(DATA_END + 1000).toISOString(), customer_id: 'c2', device_id: 'dev_c2', amount: 520, merchant_category: 'electronics' };
  for (const model of trained.models) {
    const result = predictRow(row, model.model_id);
    assert.equal(result.ok, true, model.algorithm);
    const { explanation } = result;
    assert.ok(explanation.additivity_error < 1e-6, `${model.algorithm} additivity ${explanation.additivity_error}`);
    assert.equal(explanation.method, model.algorithm === 'logistic_regression' ? 'Exact linear SHAP (log-odds)' : 'Sampled Shapley values (interventional, permutation sampling)');
    assert.ok(explanation.faithfulness && 'deletion_ratio' in explanation.faithfulness && 'ablation_correlation' in explanation.faithfulness);
    assert.ok(!('fidelity_error' in explanation), 'reconstruction error is not reported as fidelity');
    assert.ok(explanation.limitation.includes('causation'));
    assert.deepEqual(explanation.contributions, predictRow(row, model.model_id).explanation.contributions, 'deterministic for identical input');
  }
  const validation = trained.experiments.explanation_validation;
  assert.equal(validation.data_split, 'validation');
  assert.equal(validation.exact_input_consistent, true);
});

test('feature importance is real for every model family and never a zero placeholder', () => {
  const trained = trainDataset(DATA);
  for (const model of trained.models) {
    assert.notEqual(model.importance_method, undefined);
    if (model.importance_method === 'unavailable') { assert.deepEqual(model.top_features, []); continue; }
    assert.ok(model.top_features.length > 0, `${model.algorithm} has no top features`);
    assert.ok(model.top_features.every((item) => item.importance > 0), `${model.algorithm} lists a zero importance`);
    const total = model.top_features.reduce((sum, item) => sum + item.importance, 0);
    assert.ok(total <= 1.0001);
  }
  const forest = trained.models.find((model) => model.algorithm === 'random_forest');
  assert.match(forest.importance_method, /impurity/);
});

/* Issue 5 ---------------------------------------------------------------- */
test('algorithm names are honest and the old xgboost id is an alias', () => {
  const trained = trainDataset(DATA, { algorithms: ['xgboost'] });
  assert.equal(trained.models.length, 1);
  assert.equal(trained.model.algorithm, 'gradient_boosted_stumps');
  assert.match(trained.model.model_type, /not XGBoost/);
  assert.match(trained.model.implementation_note, /NOT the native XGBoost/);
  assert.equal(trained.model.implementation, 'custom (no validated library)');
});

test('SMOTE-NC interpolates between nearest minority neighbours and keeps one-hot blocks valid', () => {
  const rows = [];
  const labels = [];
  const kinds = ['a', 'b', 'c'];
  for (let i = 0; i < 40; i += 1) { rows.push({ x: String(i), kind: kinds[i % 3] }); labels.push(0); }
  for (let i = 0; i < 6; i += 1) { rows.push({ x: String(100 + i * 2), kind: i < 4 ? 'a' : 'c' }); labels.push(1); }
  const encoder = createEncoder(rows, ['x', 'kind']);
  const X = rows.map((row) => encodeRow(row, encoder));
  const result = smoteNC(X, labels, encoder);
  assert.equal(result.synthetic_rows, 34);
  assert.equal(countInvalidEncodings(result.X, encoder), 0);
  const minority = X.filter((_, i) => labels[i] === 1);
  const xs = minority.map((row) => row[0]);
  const synthetic = result.X.slice(X.length);
  assert.ok(synthetic.every((row) => row[0] >= Math.min(...xs) - 1e-9 && row[0] <= Math.max(...xs) + 1e-9), 'synthetic points lie inside the minority range');
  const unseenKind = encodeRow({ x: '5', kind: 'zzz' }, encoder);
  assert.equal(countInvalidEncodings([unseenKind], encoder), 0, 'unseen categories fall into the "other" bucket');
  assert.deepEqual(result.X.slice(0, X.length), X, 'originals are untouched');
});

/* Issue 6 ---------------------------------------------------------------- */
test('threshold is chosen by cost on validation data and responds to the cost matrix', () => {
  const labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1];
  const scores = [0.05, 0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.55, 0.6, 0.9];
  const cheapMisses = chooseThreshold(labels, scores, { cost: normalizeCostMatrix({ false_negative_cost: 1, false_positive_cost: 50, review_cost: 1 }) });
  const costlyMisses = chooseThreshold(labels, scores, { cost: normalizeCostMatrix({ false_negative_cost: 500, false_positive_cost: 1, review_cost: 0.1 }) });
  assert.ok(costlyMisses < cheapMisses, `expensive misses should lower the threshold (${costlyMisses} vs ${cheapMisses})`);
  assert.ok(costlyMisses <= 0.4);
  assert.throws(() => normalizeCostMatrix({ false_negative_cost: -1 }), /non-negative/);
});

test('trained models report cost on validation and held-out test using the configured matrix', () => {
  const costMatrix = { false_negative_cost: 300, false_positive_cost: 20, review_cost: 3 };
  const trained = trainDataset(DATA, { costMatrix });
  assert.deepEqual(trained.cost_matrix, { ...costMatrix, false_negative_use_amount: false, amount_loss_fraction: 1 });
  const { metrics, selection_metrics: validation } = trained.model;
  const matrix = metrics.confusion_matrix;
  const expected = matrix.false_negative * 300 + matrix.false_positive * 20 + (matrix.true_positive + matrix.false_positive) * 3;
  assert.equal(metrics.cost.total_cost, expected);
  assert.ok(validation.cost.total_cost >= 0);
  assert.ok('savings_vs_approve_all' in metrics.cost);
  const amountAware = trainDataset(DATA, { costMatrix: { ...costMatrix, false_negative_use_amount: true } });
  assert.equal(amountAware.model.metrics.cost.amount_weighting_applied, true);
  assert.ok(metrics.thresholds.some((item) => item.selected && 'total_cost' in item));
});

/* Artifacts, replay, staging ---------------------------------------------- */
test('export/import (v3) reproduces predictions for every model family; older artifacts are rejected', () => {
  const trained = trainDataset(DATA);
  const row = { timestamp: new Date(DATA_END + 1000).toISOString(), customer_id: 'c3', device_id: 'dev_c3', amount: 310, merchant_category: 'travel' };
  // importing replaces the registry, so capture every artifact and prediction first
  const snapshots = trained.models.map((model) => ({ algorithm: model.algorithm, original: predictRow(row, model.model_id), artifact: JSON.parse(JSON.stringify(exportModelArtifact(model.model_id))) }));
  for (const { algorithm, original, artifact } of snapshots) {
    assert.equal(artifact.artifact_version, 3);
    const imported = importModelArtifact(artifact);
    const restored = predictRow(row, imported.model.model_id);
    assert.equal(restored.risk_score, original.risk_score, algorithm);
    assert.deepEqual(restored.derived_features, original.derived_features, algorithm);
  }
  assert.throws(() => importModelArtifact({ artifact_type: 'fraudshield-model', artifact_version: 2, model: {} }), /version 3/);
});

test('replay measures operations, export/import is versioned, and retraining requires approval', () => {
  const rows = ['timestamp,amount,customer_id,is_fraud'];
  for (let index = 0; index < 20; index += 1) rows.push(`2026-01-${String(index + 1).padStart(2, '0')},${index % 5 ? 40 + index : 600},c${index % 3},${index % 5 === 0 ? 1 : 0}`);
  const csv = rows.join('\n');
  const trained = trainDataset(csv, { algorithms: ['logistic_regression'] });
  const artifact = exportModelArtifact(trained.model.model_id);
  assert.equal(artifact.artifact_type, 'fraudshield-model');
  assert.equal(importModelArtifact(artifact).ok, true);
  const replay = replayDataset(csv, { labelColumn: 'is_fraud', timeColumn: 'timestamp' });
  assert.ok(replay.throughput_rows_per_second > 0);
  assert.ok(replay.latency_ms.p95 >= 0);
  assert.ok('drift' in replay);
  assert.equal(replay.scoring_failures, 0);
  assert.ok(replay.drift.features.some((feature) => BEHAVIOR_FEATURES.includes(feature.feature)), 'drift covers the derived features');
  assert.ok('cost' in replay.detection_quality.after_change);
  trainDataset(csv, { algorithms: ['logistic_regression'], commit: false });
  assert.ok(getStagedTraining());
  assert.equal(approveStagedTraining().ok, true);
  assert.equal(getStagedTraining(), null);
});
