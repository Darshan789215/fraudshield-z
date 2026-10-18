import { performance } from 'node:perf_hooks';
import { BEHAVIOR_FEATURES, GRAPH_FEATURES, buildFeatures, computeFeatures, createFeatureState, deserializeState, detectFeatureConfig, inferenceFallbackTime, observe, rowTime, serializeState } from './ml/features.js';
import { countInvalidEncodings, createEncoder, encodeRow, plainEncoder } from './ml/encoder.js';
import { chooseThreshold, costBreakdown, evaluate, normalizeCostMatrix, thresholdAnalysis } from './ml/metrics.js';
import { ALGORITHMS, anomalyScores, makePredictor, resolveAlgorithm, trainModel } from './ml/models.js';
import { buildTopFeatures, explainLocal, explanationValidation } from './ml/explain.js';
import { smoteNC } from './ml/smote.js';
import { DEFAULT_SEED, MAX_ROWS, clamp, detectLabelColumn, detectTimeColumn, mean, parseCSV, round, seededRandom, shuffled, toLabel } from './ml/util.js';

export { MAX_ROWS, parseCSV };

let trainedModels = new Map();
let selectedModelId = null;
let stagedTraining = null;

const BACKGROUND_ROWS = 32;
const amountsOf = (items, config) => (config.amount_column ? items.map((item) => { const value = Number(item.row[config.amount_column]); return Number.isFinite(value) ? value : 0; }) : null);

/* ------------------------------------------------------------------ preview */
export function previewDataset(text, requestedLabel) {
  const rows = parseCSV(text);
  const headers = Object.keys(rows[0]).filter((header) => header !== '__row_index');
  const labelColumn = detectLabelColumn(headers, requestedLabel);
  const timeColumn = detectTimeColumn(headers);
  const labels = rows.map((row) => toLabel(row[labelColumn])).filter((value) => value !== null);
  return {
    row_count: rows.length, columns: headers, label_column: labelColumn, time_column: timeColumn,
    positive_rate: labels.length ? round(labels.reduce((sum, label) => sum + label, 0) / labels.length) : null,
    preview: rows.slice(0, 5).map(({ __row_index, ...row }) => row),
  };
}

/* ------------------------------------------------------------- operational */
function operationalMetrics(model, X, labels, scores, context) {
  const sample = X.slice(0, 2000);
  const latencies = []; const repeated = []; let failures = 0;
  sample.forEach((features, index) => {
    try {
      const started = performance.now();
      const score = model.predictor(features);
      latencies.push(performance.now() - started);
      if (!Number.isFinite(score)) failures += 1;
      if (index < 5) repeated.push([score, model.predictor(features), model.predictor(features)]);
    } catch { failures += 1; }
  });
  latencies.sort((a, b) => a - b);
  const consistency = repeated.length ? repeated.reduce((sum, values) => sum + (Math.max(...values) - Math.min(...values) < 1e-12 ? 1 : 0), 0) / repeated.length : 1;
  const pick = (q) => latencies[Math.min(latencies.length - 1, Math.floor(latencies.length * q))] || 0;
  return {
    prediction_latency_ms: { median: round(pick(0.5), 4), p95: round(pick(0.95), 4), max: round(latencies.at(-1) || 0, 4) },
    false_positive_workload_rate: evaluate(labels, scores, model.threshold, context).workload_rate,
    decision_consistency_rate: round(consistency),
    failure_handling: { invalid_prediction_failures: failures, failure_rate: round(failures / Math.max(1, sample.length)), fallback: 'review_required / no approval' },
    latency_measured_on_rows: sample.length,
  };
}

function publicModel(model) {
  return {
    model_id: model.model_id, algorithm: model.algorithm, model_type: model.model_type, family: model.family, tradeoff: model.tradeoff,
    implementation: 'custom (no validated library)', implementation_note: model.implementation_note,
    label_column: model.label_column, time_column: model.time_column,
    input_columns: model.input_columns, derived_features: BEHAVIOR_FEATURES, feature_columns: model.feature_columns,
    model_version: model.model_version, trained_at: model.trained_at, row_count: model.row_count, positive_count: model.positive_count,
    threshold: model.threshold, hybrid_threshold: model.hybrid_threshold,
    hybrid_validation_metrics: model.hybrid_validation_metrics, hybrid_test_metrics: model.hybrid_test_metrics ?? null, cost_matrix: model.cost_matrix,
    evaluation_role: model.evaluation_role, metrics_source: model.metrics.source,
    metrics: model.metrics, selection_metrics: model.selection_metrics, test_metrics: model.test_metrics ?? null, exploratory_test_metrics: model.exploratory_test_metrics ?? null,
    top_features: model.top_features, importance_method: model.importance_method, importance_unit: model.importance_unit, importance_note: model.importance_note,
    evaluation_protocol: model.evaluation_protocol,
  };
}

function selectedRiskSignal(features) {
  const clamp01 = (value) => clamp(Number.isFinite(Number(value)) ? Number(value) : 0);
  return clamp(
    0.18 * clamp01(Number(features.behavior_amount_deviation) / 5) +
    0.18 * clamp01(features.behavior_new_device) +
    0.22 * clamp01(Number(features.behavior_txn_count_10m) / 5) +
    0.22 * clamp01(Number(features.relationship_shared_device_count) / 3) +
    0.20 * clamp01(Number(features.relationship_repeated_merchant_count) / 5)
  );
}
const hybridScore = (classifierScore, features) => clamp(0.75 * classifierScore + 0.25 * selectedRiskSignal(features));

/* ----------------------------------------------------------------- training */
export function trainDataset(text, options = {}) {
  const allRows = parseCSV(text);
  if (allRows.length > MAX_ROWS) throw new Error(`This dataset has ${allRows.length.toLocaleString()} rows. The maximum is ${MAX_ROWS.toLocaleString()} rows.`);
  const headers = Object.keys(allRows[0]).filter((header) => header !== '__row_index');
  const labelColumn = detectLabelColumn(headers, options.labelColumn);
  const timeColumn = detectTimeColumn(headers, options.timeColumn);
  const cost = normalizeCostMatrix(options.costMatrix);
  const config = detectFeatureConfig(headers, { labelColumn, timeColumn });

  let labeled = allRows
    .map((row) => ({ row, label: toLabel(row[labelColumn]), time: rowTime(config, row, row.__row_index) }))
    .filter((item) => item.label !== null)
    .sort((a, b) => a.time - b.time || a.row.__row_index - b.row.__row_index);
  if (labeled.length < 12) throw new Error('At least 12 rows with a recognizable binary label are required. Choose the target column if this is a bank CSV.');
  if (new Set(labeled.map((item) => item.label)).size < 2) throw new Error('The target column must contain both 0/legitimate and 1/fraud examples.');

  const trainEnd = Math.max(2, Math.floor(labeled.length * 0.6));
  const validationEnd = Math.max(trainEnd + 1, Math.floor(labeled.length * 0.8));
  // Features for the full chronology remain past-only. Deployment state is a
  // separate snapshot ending immediately before the validation period.
  const featureState = createFeatureState();
  labeled = buildFeatures(config, labeled, featureState);
  const trainingFeatureState = createFeatureState();
  buildFeatures(config, labeled.slice(0, trainEnd).map(({ row, time }) => ({ row, time })), trainingFeatureState);

  const rawColumns = headers.filter((header) => header !== labelColumn && header !== timeColumn);
  const featureColumns = [...rawColumns, ...BEHAVIOR_FEATURES];
  const trainRows = labeled.slice(0, trainEnd); const validationRows = labeled.slice(trainEnd, validationEnd); const testRows = labeled.slice(validationEnd);
  const trainY = trainRows.map((item) => item.label); const validationY = validationRows.map((item) => item.label); const testY = testRows.map((item) => item.label);
  if (new Set(trainY).size < 2) throw new Error('The training period (first 60% of rows in time order) contains only one class. Provide more data or a longer history.');
  const validationContext = { cost, amounts: amountsOf(validationRows, config) };
  const testContext = { cost, amounts: amountsOf(testRows, config) };

  const makeFeatureSet = (columns) => {
    const encoder = createEncoder(trainRows.map((item) => item.row), columns);
    const encode = (items) => items.map((item) => encodeRow(item.row, encoder));
    return { columns, encoder, trainX: encode(trainRows), validationX: encode(validationRows), testX: encode(testRows) };
  };
  const full = makeFeatureSet(featureColumns);
  const original = makeFeatureSet(rawColumns);
  const graph = makeFeatureSet([...rawColumns, ...GRAPH_FEATURES]);

  const requested = (Array.isArray(options.algorithms) && options.algorithms.length ? options.algorithms : Object.keys(ALGORITHMS)).map(resolveAlgorithm);
  const algorithms = [...new Set(requested.filter((algorithm) => ALGORITHMS[algorithm]))];
  if (!algorithms.length) throw new Error('Select at least one supported algorithm.');

  const trainedAt = new Date().toISOString();
  const background = shuffled([...full.trainX.keys()], seededRandom(DEFAULT_SEED + 11)).slice(0, BACKGROUND_ROWS).map((index) => full.trainX[index]);
  const protocol = {
    split: 'chronological 60% train / 20% validation / 20% test',
    train_period_end: trainRows.at(-1)?.time, validation_period_end: validationRows.at(-1)?.time, test_period_start: testRows[0]?.time,
    model_selection: 'validation PR-AUC (tie-break: lower validation cost, then algorithm name). Selection is frozen BEFORE the test set is scored.',
    threshold_selection: 'minimum total cost on validation data using the configured cost matrix',
    test_usage: 'The test set is scored once for the selected model. Other models have validation metrics only unless exploratoryTestComparison is requested; those are labelled exploratory and must not be used to re-select.',
    no_resampling: true, resampling_note: 'Production candidates are trained without resampling. SMOTE-NC appears only in the controlled ablation.',
    behavioural_features_past_only: true, behavioural_feature_names: BEHAVIOR_FEATURES,
    feature_pipeline: 'One shared past-only pipeline builds derived features; deployed state is snapshotted at the training cutoff. Validation/test history is excluded from deployed state.',
    deployment_history_cutoff: 'training_period_end; validation and test rows are not included in a newly trained model\'s initial inference history',
    hybrid_decision: '0.75 * classifier score + 0.25 * selected temporal/relationship signal; threshold minimizes configured cost on validation only',
  };

  // ---- 1. fit every candidate; compare on VALIDATION only
  const candidates = algorithms.map((algorithm) => {
    const implementation = ALGORITHMS[algorithm];
    const state = trainModel(algorithm, full.trainX, trainY);
    const predictor = makePredictor(state);
      const validationScores = full.validationX.map((features) => clamp(predictor(features)));
      const threshold = chooseThreshold(validationY, validationScores, validationContext);
      const selectionMetrics = evaluate(validationY, validationScores, threshold, validationContext);
      const validationHybridScores = validationScores.map((score, index) => hybridScore(score, validationRows[index].row));
      const hybridThreshold = chooseThreshold(validationY, validationHybridScores, validationContext);
      const hybridValidationMetrics = evaluate(validationY, validationHybridScores, hybridThreshold, validationContext);
    const model = {
      model_id: `${algorithm}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, algorithm, model_type: implementation.label, family: implementation.family,
      tradeoff: implementation.tradeoff, implementation_note: implementation.implementation_note,
      label_column: labelColumn, time_column: timeColumn || 'row order (no time column detected)', time_column_name: timeColumn || null,
      input_columns: [...rawColumns, ...(timeColumn ? [timeColumn] : [])], feature_columns: featureColumns,
      feature_config: config, feature_state: deserializeState(serializeState(trainingFeatureState)), encoder: full.encoder, background,
      predictor, predictor_state: state, cost_matrix: cost, threshold,
      hybrid_threshold: hybridThreshold, hybrid_validation_metrics: hybridValidationMetrics,
      model_version: `temporal-${algorithm}-${Date.now()}`, trained_at: trainedAt, row_count: labeled.length, positive_count: labeled.filter((item) => item.label === 1).length,
      selection_metrics: selectionMetrics, evaluation_protocol: protocol, validationScores,
    };
    Object.assign(model, (({ importance_method, top_features, importance_unit, note }) => ({ importance_method, top_features, importance_unit, importance_note: note }))(buildTopFeatures(model, full.validationX, validationY)));
    return model;
  });

  // ---- 2. choose the winner from validation results ONLY, then freeze it
  const ranked = [...candidates].sort((a, b) => (b.selection_metrics.pr_auc ?? -1) - (a.selection_metrics.pr_auc ?? -1) || a.selection_metrics.cost.total_cost - b.selection_metrics.cost.total_cost || a.algorithm.localeCompare(b.algorithm));
  const best = ranked[0];

  // ---- 3. evaluate the frozen choice on the test set exactly once
  const validationView = (model) => ({
    ...model.selection_metrics, threshold: model.threshold, source: 'validation', validation: model.selection_metrics,
    thresholds: thresholdAnalysis(validationY, model.validationScores, validationContext, model.threshold),
    operational: operationalMetrics(model, full.validationX, validationY, model.validationScores, validationContext),
  });
  const testScoresOf = (model) => full.testX.map((features) => clamp(model.predictor(features)));
  for (const model of candidates) {
    model.metrics = validationView(model);
    model.evaluation_role = 'not_selected_validation_only';
    if (options.exploratoryTestComparison && model !== best) {
      const scores = testScoresOf(model);
      model.exploratory_test_metrics = { ...evaluate(testY, scores, model.threshold, testContext), evaluation: 'exploratory_test_comparison_do_not_use_for_selection' };
    }
  }
  const finalScores = testScoresOf(best);
  const finalTest = evaluate(testY, finalScores, best.threshold, testContext);
  const finalHybridScores = finalScores.map((score, index) => hybridScore(score, testRows[index].row));
  best.hybrid_test_metrics = { ...evaluate(testY, finalHybridScores, best.hybrid_threshold, testContext), evaluation: 'final_unbiased_test_evaluation', threshold_source: 'validation' };
  best.test_metrics = { ...finalTest, evaluation: 'final_unbiased_test_evaluation' };
  best.evaluation_role = 'selected_final_test_evaluation';
  best.metrics = {
    ...finalTest, threshold: best.threshold, source: 'test', validation: best.selection_metrics,
    thresholds: thresholdAnalysis(testY, finalScores, testContext, best.threshold),
    natural_test_prevalence: finalTest.prevalence, evaluated_after_training_period: true,
    operational: operationalMetrics(best, full.testX, testY, finalScores, testContext),
  };
  for (const model of candidates) delete model.validationScores;

  // ---- 4. controlled experiments (exploratory: they never influence selection or thresholds)
  let experiments;
  if (options.skipExploratoryExperiments === true) {
    experiments = { skipped: true, note: 'Separate exploratory ablations and anomaly/graph comparisons were skipped by the caller; selected-candidate validation and held-out test evaluation still ran.' };
  } else {
  const runVariant = ({ id, set, imbalance, featuresLabel }) => {
    let X = set.trainX; let y = trainY; let resampling = { method: 'none', synthetic_rows: 0, invalid_encoded_rows: 0 };
    if (imbalance === 'smote') {
      const result = smoteNC(set.trainX, trainY, set.encoder);
      X = result.X; y = result.y;
      resampling = { method: result.method, synthetic_rows: result.synthetic_rows, k: result.k, invalid_encoded_rows: countInvalidEncodings(result.X, set.encoder), note: result.note };
    }
    const state = trainModel('logistic_regression', X, y, { classWeighted: imbalance === 'class_weighted' });
    const predictor = makePredictor(state);
    const validationScores = set.validationX.map((features) => clamp(predictor(features)));
    const threshold = chooseThreshold(validationY, validationScores, validationContext);
    const testScores = set.testX.map((features) => clamp(predictor(features)));
    return {
      variant: id, features: featuresLabel, feature_count: set.encoder.definitions.length, imbalance_method: imbalance === 'smote' ? 'SMOTE-NC' : imbalance === 'class_weighted' ? 'class weights' : 'none',
      training_rows: X.length, synthetic_rows: resampling.synthetic_rows, resampling, threshold,
      validation_metrics: evaluate(validationY, validationScores, threshold, validationContext),
      metrics: evaluate(testY, testScores, threshold, testContext), evaluation: 'exploratory_test',
    };
  };
  const ablationVariants = [
    runVariant({ id: 'original_baseline', set: original, imbalance: 'none', featuresLabel: 'original' }),
    runVariant({ id: 'original_smote', set: original, imbalance: 'smote', featuresLabel: 'original' }),
    runVariant({ id: 'behavioural_baseline', set: full, imbalance: 'none', featuresLabel: 'original + behavioural' }),
    runVariant({ id: 'behavioural_smote', set: full, imbalance: 'smote', featuresLabel: 'original + behavioural' }),
  ];
  const byId = Object.fromEntries(ablationVariants.map((variant) => [variant.variant, variant]));
  const delta = (a, b) => ({
    from: b.variant, to: a.variant,
    test_pr_auc_change: a.metrics.pr_auc === null || b.metrics.pr_auc === null ? null : round(a.metrics.pr_auc - b.metrics.pr_auc),
    test_cost_change: round(a.metrics.cost.total_cost - b.metrics.cost.total_cost, 2),
    validation_pr_auc_change: a.validation_metrics.pr_auc === null || b.validation_metrics.pr_auc === null ? null : round(a.validation_metrics.pr_auc - b.validation_metrics.pr_auc),
  });
  const testPositives = testY.reduce((sum, label) => sum + label, 0);
  const trainingVariants = [
    { ...byId.behavioural_baseline, variant: 'baseline' },
    { ...runVariant({ id: 'class_weighted', set: full, imbalance: 'class_weighted', featuresLabel: 'original + behavioural' }) },
    { ...byId.behavioural_smote, variant: 'smote' },
  ];
  const graphVariant = runVariant({ id: 'original_plus_graph', set: graph, imbalance: 'none', featuresLabel: 'original + graph' });
  const anomalyValidation = anomalyScores(full.trainX, full.validationX); const anomalyTest = anomalyScores(full.trainX, full.testX);
  const anomalyThreshold = chooseThreshold(validationY, anomalyValidation, validationContext);

  experiments = {
    note: 'Test-set numbers inside experiments are descriptive. They are never used to select the model, the threshold, or any setting.',
    training_variants: trainingVariants,
    ablation: {
      design: '2x2 controlled comparison: {original, original + behavioural features} x {no resampling, SMOTE-NC}. Same chronological split, same logistic-regression learner, same cost-based validation threshold procedure, same untouched test set.',
      variants: ablationVariants,
      contrasts: {
        behavioural_features_without_resampling: delta(byId.behavioural_baseline, byId.original_baseline),
        behavioural_features_with_smote: delta(byId.behavioural_smote, byId.original_smote),
        smote_on_original_features: delta(byId.original_smote, byId.original_baseline),
        smote_on_behavioural_features: delta(byId.behavioural_smote, byId.behavioural_baseline),
      },
      test_fraud_cases: testPositives,
      reliability_warning: testPositives < 30 ? `Only ${testPositives} fraud case(s) in the test period: differences between variants are within noise and should not be treated as evidence.` : null,
    },
    anomaly_detector: {
      algorithm: 'isolation_forest', implementation: 'custom simplified isolation forest (32 trees, 128-row subsamples)',
      note: 'Separate unsupervised experiment; it does not use fraud labels during fitting and is not blended into supervised probabilities.',
      threshold: anomalyThreshold, metrics: evaluate(testY, anomalyTest, anomalyThreshold, testContext), evaluation: 'exploratory_test',
    },
    graph_features: {
      feature_names: GRAPH_FEATURES, baseline_metrics: byId.original_baseline.metrics, graph_metrics: graphVariant.metrics,
      threshold: graphVariant.threshold, note: 'Original features vs original + relationship features; both without resampling.',
    },
    explanation_validation: explanationValidation(best, full.validationX, validationRows),
    cost_context: {
      cost_matrix: cost,
      selected_model_test_cost: best.metrics.cost, approve_everything_test_cost: best.metrics.cost.approve_all_cost,
      amount_weighting_applied: best.metrics.cost.amount_weighting_applied,
    },
  };
  }

  const candidateModels = new Map(candidates.map((model) => [model.model_id, model]));
  if (options.commit === false) stagedTraining = { selectedModelId: best.model_id, models: candidateModels, stagedAt: new Date().toISOString() };
  else { selectedModelId = best.model_id; trainedModels = candidateModels; }

  return {
    ok: true, selected_model_id: best.model_id, model: publicModel(best), models: candidates.map(publicModel), comparison: candidates.map(publicModel),
    selection: {
      criterion: 'validation PR-AUC, tie-break lower validation cost', data_used: 'validation only',
      ranking: ranked.map((model) => ({ model_id: model.model_id, algorithm: model.algorithm, validation_pr_auc: model.selection_metrics.pr_auc, validation_total_cost: model.selection_metrics.cost.total_cost })),
      final_test_reported_for: best.model_id,
    },
    cost_matrix: cost, experiments,
    literature_review_methodology: 'Search peer-reviewed and official benchmark sources using a documented query, record databases/date/eligibility criteria, deduplicate, extract dataset/time-split/metric/model details, and preserve the exact queries and screening decisions in a versioned evidence table.',
  };
}

/* --------------------------------------------------------------- prediction */
/** Shared scoring path for single predictions and replay. History comes from `state`. */
function scoreWithModel(model, state, row, { observeRow = false, time } = {}) {
  const config = model.feature_config;
  const rowTimeValue = time ?? rowTime(config, row, inferenceFallbackTime(state, row));
  const features = computeFeatures(config, state, row, rowTimeValue);
  const enriched = { ...row, ...features };
  const encoded = encodeRow(enriched, model.encoder);
  const risk = clamp(model.predictor(encoded));
  if (!Number.isFinite(risk)) throw new Error('Model returned a non-finite score.');
  const hybridRisk = hybridScore(risk, features);
  if (observeRow) observe(config, state, row, rowTimeValue);
  return { risk, hybridRisk, features, enriched, encoded, time: rowTimeValue };
}

/**
 * Score one transaction. Derived behavioural / relationship features are rebuilt from the model's
 * stored customer history, exactly as in training. By default scoring is read-only (what-if);
 * pass { observe: true } to append the transaction to the live history afterwards.
 */
export function predictRow(row, modelId = selectedModelId, options = {}) {
  if (process.env.FRAUDSHIELD_MODEL_DISABLED === '1') return { ok: true, decision: 'review_required', risk_score: null, model_failure: true, fallback: 'review_required / no approval' };
  const model = trainedModels.get(modelId) || trainedModels.get(selectedModelId);
  if (!model) throw new Error('Train a model before scoring a transaction.');
  try {
    const started = performance.now();
    const scored = scoreWithModel(model, model.feature_state, row, { observeRow: Boolean(options.observe) });
    const latency = performance.now() - started;
    const flagged = scored.hybridRisk >= model.hybrid_threshold;
    return {
      ok: true, risk_score: round(scored.hybridRisk), threshold: model.hybrid_threshold, prediction: flagged ? 'fraud' : 'legitimate',
      classifier_score: round(scored.risk), classifier_threshold: model.threshold,
      hybrid_risk_score: round(scored.hybridRisk), hybrid_threshold: model.hybrid_threshold,
      decision: flagged ? 'review_required' : 'approve', decision_basis: 'validation-calibrated classifier + temporal/relationship signals',
      action: flagged ? 'review_or_step_up' : 'approve', model_id: model.model_id, model_type: model.model_type, model_version: model.model_version,
      latency_ms: round(latency, 4), derived_features: Object.fromEntries(Object.entries(scored.features).map(([key, value]) => [key, round(value, 4)])),
      top_features: model.top_features, importance_method: model.importance_method,
      explanation: options.explain === false ? undefined : explainLocal(model, scored.encoded, scored.enriched),
      cost_matrix: model.cost_matrix, input: row, history_updated: Boolean(options.observe),
    };
  } catch (error) {
    return { ok: false, decision: 'review_required', model_failure: true, error: error.message, fallback: 'review_required / no approval' };
  }
}

/* ------------------------------------------------------------------- replay */
function driftSummary(before, after, columns) {
  const features = columns.map((column) => {
    const a = before.map((row) => Number(row[column])).filter(Number.isFinite); const b = after.map((row) => Number(row[column])).filter(Number.isFinite);
    if (!a.length || !b.length) return null;
    const baseline = mean(a); const current = mean(b);
    return { feature: column, baseline_mean: round(baseline), current_mean: round(current), relative_shift: round(Math.abs(current - baseline) / Math.max(1, Math.abs(baseline))) };
  }).filter(Boolean);
  const alerts = features.filter((feature) => feature.relative_shift >= 0.25);
  return { features, alerts, alert: alerts.length > 0, threshold: 0.25, message: alerts.length ? 'Feature distribution drift detected; validate a candidate before replacement.' : 'No configured feature-shift alert detected.' };
}

export function replayDataset(text, options = {}) {
  if (process.env.FRAUDSHIELD_MODEL_DISABLED === '1') throw new Error('Model scoring is disabled; replay cannot run.');
  const model = trainedModels.get(options.modelId || selectedModelId);
  if (!model) throw new Error('Train or import a model before replay.');
  const config = model.feature_config;
  const labelColumn = options.labelColumn || model.label_column;
  const rows = parseCSV(text).map((row) => ({ row, time: rowTime(config, row, row.__row_index) })).sort((a, b) => a.time - b.time || a.row.__row_index - b.row.__row_index);
  const selected = rows.slice(0, Math.min(rows.length, Number(options.limit) || rows.length));
  const cutoff = model.evaluation_protocol?.train_period_end;
  const replayStartsAfterTraining = cutoff !== undefined && selected.length > 0 && selected[0].time > cutoff;
  const state = replayStartsAfterTraining ? deserializeState(serializeState(model.feature_state)) : createFeatureState();
  const started = performance.now();
  let failures = 0;
  const observations = [];
  for (const item of selected) {
    try {
      const tick = performance.now();
      const scored = scoreWithModel(model, state, item.row, { observeRow: true, time: item.time });
      observations.push({ row: scored.enriched, risk_score: scored.hybridRisk, classifier_score: scored.risk, latency_ms: performance.now() - tick, label: toLabel(item.row[labelColumn]) });
    } catch { failures += 1; }
  }
  const elapsed = Math.max(0.001, performance.now() - started);
  const midpoint = Math.max(1, Math.floor(observations.length / 2));
  const before = observations.slice(0, midpoint); const after = observations.slice(midpoint);
  const quality = (window) => {
    const labelled = window.filter((item) => item.label !== null);
    if (!labelled.length) return null;
    const amounts = config.amount_column ? labelled.map((item) => Number(item.row[config.amount_column]) || 0) : null;
    return evaluate(labelled.map((item) => item.label), labelled.map((item) => item.risk_score), model.threshold, { cost: model.cost_matrix, amounts });
  };
  const latencies = observations.map((item) => item.latency_ms).sort((a, b) => a - b);
  const pick = (q) => latencies[Math.min(latencies.length - 1, Math.floor(latencies.length * q))] || 0;
  const driftColumns = model.encoder.definitions.filter((definition) => definition.type === 'numeric').map((definition) => definition.column);
  return {
    ok: true, model_version: model.model_version, model_id: model.model_id, rows_replayed: observations.length, scoring_failures: failures,
    throughput_rows_per_second: round(observations.length / (elapsed / 1000), 2),
    latency_ms: { median: round(pick(0.5), 4), p95: round(pick(0.95), 4), max: round(latencies.at(-1) || 0, 4) },
    detection_quality: { before_change: quality(before), after_change: quality(after) },
    drift: driftSummary(before.map((item) => item.row), after.map((item) => item.row), driftColumns),
    feature_pipeline: replayStartsAfterTraining
      ? 'Replay starts from saved training-cutoff history and advances transaction by transaction; validation/test history is never preloaded.'
      : 'Replay includes rows at or before the training cutoff, so it starts empty to avoid duplicate training history; each row advances history only after scoring.',
    in_sample_note: 'If the replayed rows overlap the training data, quality figures are in-sample and optimistic.',
    replacement_policy: 'Replay is observational. Drift alerts stage a candidate for validation; they never retrain or replace the active model automatically.',
  };
}

/* ------------------------------------------------------- staging / registry */
export function approveStagedTraining() {
  if (!stagedTraining) return { ok: false, error: 'No staged candidate is waiting for approval.' };
  trainedModels = stagedTraining.models; selectedModelId = stagedTraining.selectedModelId;
  const approvedAt = new Date().toISOString(); stagedTraining = null;
  return { ok: true, approved_at: approvedAt, model: getTrainedModel() };
}
export function getStagedTraining() { return stagedTraining ? { staged_at: stagedTraining.stagedAt, model: publicModel(stagedTraining.models.get(stagedTraining.selectedModelId)) } : null; }

export function exportModelArtifact(modelId = selectedModelId) {
  const model = trainedModels.get(modelId);
  if (!model) throw new Error('No trained model is available for export.');
  return {
    artifact_type: 'fraudshield-model', artifact_version: 3, exported_at: new Date().toISOString(),
    model: {
      ...publicModel(model), time_column_name: model.time_column_name, encoder: plainEncoder(model.encoder), predictor_state: model.predictor_state,
      feature_config: model.feature_config, feature_state: serializeState(model.feature_state), background: model.background,
    },
  };
}

export function importModelArtifact(artifact) {
  if (!artifact || artifact.artifact_type !== 'fraudshield-model') throw new Error('Unsupported FraudShield model artifact.');
  if (artifact.artifact_version !== 3) throw new Error('This artifact version does not contain timestamp-aware relationships and calibrated hybrid thresholds. Re-export from a retrained model (version 3).');
  const source = artifact.model;
  if (!source?.predictor_state || !source.encoder || !source.feature_config || !source.background) throw new Error('Artifact is missing predictor state, encoder, feature configuration or background sample.');
  if (!Number.isFinite(source.threshold) || source.threshold < 0 || source.threshold > 1 || !Number.isFinite(source.hybrid_threshold) || source.hybrid_threshold < 0 || source.hybrid_threshold > 1 || !source.feature_state) {
    throw new Error('Artifact is missing valid classifier/hybrid thresholds or its training-cutoff feature history.');
  }
  const model = {
    ...source, predictor: makePredictor(source.predictor_state), feature_state: deserializeState(source.feature_state),
    model_id: `${source.model_id}-imported-${Date.now()}`, model_version: `${source.model_version}-imported`,
  };
  trainedModels = new Map([[model.model_id, model]]); selectedModelId = model.model_id;
  return { ok: true, model: publicModel(model), imported_at: new Date().toISOString() };
}

export function getTrainedModel() { const model = trainedModels.get(selectedModelId); return model ? publicModel(model) : null; }
export function getTrainedModels() { return [...trainedModels.values()].map(publicModel); }
export function hasTrainedModel() { return trainedModels.size > 0; }
export function getModelScoringSchema(modelId = selectedModelId) {
  const model = trainedModels.get(modelId) || trainedModels.get(selectedModelId);
  return model ? { model_id: model.model_id, input_columns: model.input_columns, feature_config: model.feature_config } : null;
}
