import { encoderGroups } from './encoder.js';
import { logisticMargin, nativeImportance, NATIVE_IMPORTANCE_METHOD } from './models.js';
import { prAuc } from './metrics.js';
import { DEFAULT_SEED, round, seededRandom, shuffled } from './util.js';

/**
 * Local explanations.
 *
 * Players are INPUT COLUMNS (a numeric column, or a whole one-hot block), so swapping a player
 * with background values can never produce an invalid encoding.
 *
 *  - Logistic regression: exact linear SHAP in log-odds space, phi_g = sum_i w_i (x_i - mean_bg_i).
 *  - Every other model: permutation-sampling Shapley estimate (Strumbelj & Kononenko) against a
 *    background sample drawn from the training data (interventional / independent-features
 *    assumption), with antithetic permutations. It is an ESTIMATE with a reported standard error
 *    and is not TreeSHAP.
 *
 * Faithfulness is measured separately (see faithfulness()); additivity of the sampled values is a
 * mathematical property of the estimator, not evidence that the explanation is faithful.
 */

const hashVector = (vector) => { let h = 2166136261; for (const value of vector) { h ^= Math.round(value * 1e6) | 0; h = Math.imul(h, 16777619); } return h >>> 0; };
const blockOf = (vector, group) => vector.slice(group.start, group.end);
const withBlock = (target, source, group) => { const copy = target.slice(); for (let i = group.start; i < group.end; i += 1) copy[i] = source[i]; return copy; };

function outputFunction(model) {
  const linear = model.predictor_state.type === 'logistic';
  return { linear, unit: linear ? 'log-odds' : 'probability', fn: linear ? (x) => logisticMargin(model.predictor_state, x) : model.predictor };
}

export function explainLocal(model, encoded, rawRow, options = {}) {
  const { permutations = 32, seed = hashVector(encoded) + DEFAULT_SEED, faithfulnessCheck = true } = options;
  const groups = encoderGroups(model.encoder);
  const { linear, unit, fn } = outputFunction(model);
  const background = model.background;
  const output = fn(encoded);
  let phi; let baseValue; let stdErrors; let sampling;

  if (linear) {
    const weights = model.predictor_state.weights;
    const meanBackground = encoded.map((_, i) => background.reduce((sum, row) => sum + row[i], 0) / background.length);
    baseValue = model.predictor_state.bias + weights.reduce((sum, w, i) => sum + w * meanBackground[i], 0);
    phi = groups.map((group) => { let total = 0; for (let i = group.start; i < group.end; i += 1) total += weights[i] * (encoded[i] - meanBackground[i]); return total; });
    stdErrors = groups.map(() => 0);
    sampling = { method: 'closed form (exact)', permutations: 0, background_rows: background.length };
  } else {
    const random = seededRandom(seed);
    const sums = Array(groups.length).fill(0); const squares = Array(groups.length).fill(0);
    let baseSum = 0; let draws = 0;
    const walk = (order, start) => {
      let current = start.slice(); let previous = fn(current); baseSum += previous; draws += 1;
      for (const g of order) {
        current = withBlock(current, encoded, groups[g]);
        const value = fn(current); const delta = value - previous; previous = value;
        sums[g] += delta; squares[g] += delta * delta;
      }
    };
    for (let p = 0; p < permutations; p += 1) {
      const order = shuffled([...groups.keys()], random);
      const start = background[Math.floor(random() * background.length)];
      walk(order, start); walk([...order].reverse(), start); // antithetic pair shares the same background row
    }
    phi = sums.map((total) => total / draws);
    baseValue = baseSum / draws;
    stdErrors = squares.map((sq, g) => Math.sqrt(Math.max(0, sq / draws - phi[g] ** 2) / draws));
    sampling = { method: 'permutation sampling', permutations: permutations * 2, background_rows: background.length, max_std_error: round(Math.max(...stdErrors, 0), 5) };
  }

  const additivityError = Math.abs(baseValue + phi.reduce((sum, v) => sum + v, 0) - output);
  const rows = groups.map((group, g) => ({ feature: group.column, value: rawRow?.[group.column] ?? null, contribution: round(phi[g], 4), std_error: round(stdErrors[g], 4), direction: phi[g] >= 0 ? 'increases_risk' : 'decreases_risk' }));
  rows.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  const result = {
    method: linear ? 'Exact linear SHAP (log-odds)' : 'Sampled Shapley values (interventional, permutation sampling)',
    unit, base_value: round(baseValue, 4), output_value: round(output, 4),
    contributions: rows.slice(0, 6), all_contribution_count: rows.length,
    additivity_error: round(additivityError, 6),
    sampling,
    limitation: 'Contributions describe how this trained model responds to inputs (association) and do not establish causation. Features are treated as independent when sampling background values; correlated inputs can share credit. The sampled method is an estimate, not TreeSHAP.',
  };
  if (faithfulnessCheck) result.faithfulness = faithfulness(model, encoded, phi, groups, { seed: seed + 1 });
  Object.defineProperty(result, '_phi', { value: phi, enumerable: false });
  return result;
}

/**
 * Faithfulness is checked against the model itself, independently of how the attributions were produced:
 *  - deletion test: replace the k highest-|phi| columns by background values and measure the output change;
 *    compare with replacing k random columns. A faithful explanation has ratio > 1.
 *  - ablation correlation: Pearson correlation between phi_g and the effect of replacing ONLY column g.
 */
export function faithfulness(model, encoded, phi, groups, { seed }) {
  const { unit, fn } = outputFunction(model);
  const background = model.background; const random = seededRandom(seed);
  const output = fn(encoded);
  const k = Math.max(1, Math.min(3, Math.floor(groups.length / 2)));
  const meanReplacementOutput = (indexes) => background.reduce((sum, row) => { let current = encoded; for (const g of indexes) current = withBlock(current, row, groups[g]); return sum + fn(current); }, 0) / background.length;
  const effect = (indexes) => output - meanReplacementOutput(indexes);

  const ranking = groups.map((_, g) => g).sort((a, b) => Math.abs(phi[b]) - Math.abs(phi[a]));
  const topChange = Math.abs(effect(ranking.slice(0, k)));
  const randomTrials = 12; let randomChange = 0;
  for (let t = 0; t < randomTrials; t += 1) randomChange += Math.abs(effect(shuffled([...groups.keys()], random).slice(0, k)));
  randomChange /= randomTrials;

  const singles = groups.map((_, g) => effect([g]));
  const correlation = pearson(phi, singles);
  const ratio = randomChange > 1e-9 ? topChange / randomChange : null;
  return {
    output_unit: unit, deletion_k: k,
    deletion_top_k_change: round(topChange, 4), deletion_random_k_change: round(randomChange, 4), deletion_ratio: ratio === null ? null : round(ratio, 3),
    ablation_correlation: correlation === null ? null : round(correlation, 3),
    assessment: assessFaithfulness(topChange, randomChange, ratio, correlation),
    note: 'Assessment = deletion ratio >= 1 and ablation correlation >= 0.5 (not_assessable when the output does not respond to replacing columns). Empirical check on this input only.',
  };
}

function assessFaithfulness(topChange, randomChange, ratio, correlation) {
  if (topChange < 1e-9 && randomChange < 1e-9) return 'not_assessable_flat_output';
  const ratioOk = ratio === null ? topChange > 0 : ratio >= 1;
  return ratioOk && (correlation === null || correlation >= 0.5) ? 'consistent_with_model' : 'weak_or_inconclusive';
}

function pearson(a, b) {
  const n = a.length; if (n < 3) return null;
  const meanA = a.reduce((s, v) => s + v, 0) / n; const meanB = b.reduce((s, v) => s + v, 0) / n;
  let cov = 0; let varA = 0; let varB = 0;
  for (let i = 0; i < n; i += 1) { cov += (a[i] - meanA) * (b[i] - meanB); varA += (a[i] - meanA) ** 2; varB += (b[i] - meanB) ** 2; }
  return varA > 1e-12 && varB > 1e-12 ? cov / Math.sqrt(varA * varB) : null;
}

/** Empirical validation of the explanation method on held-out VALIDATION rows (never the test set). */
export function explanationValidation(model, validationX, validationRows, limit = 15) {
  const count = Math.min(limit, validationX.length);
  if (!count) return { rows_checked: 0, exact_input_consistent: true, prediction_reproducible: true, note: 'No validation rows available.' };
  const groups = encoderGroups(model.encoder);
  const topNames = (explanation, n = 5) => explanation.contributions.slice(0, n).map((item) => item.feature);
  const overlap = (a, b) => (a.length ? a.filter((name) => b.includes(name)).length / a.length : 1);
  let exact = true; let reproducible = true;
  const nearby = []; const seedStability = []; const ratios = []; const correlations = []; const standardErrors = []; let consistent = 0; let assessable = 0;
  for (let r = 0; r < count; r += 1) {
    const x = validationX[r]; const raw = validationRows[r]?.row;
    const first = explainLocal(model, x, raw, { faithfulnessCheck: true });
    const second = explainLocal(model, x, raw, { faithfulnessCheck: true });
    exact = exact && JSON.stringify(first) === JSON.stringify(second);
    reproducible = reproducible && model.predictor(x) === model.predictor(x);
    const perturbed = x.slice();
    for (const group of groups) if (group.type === 'numeric') perturbed[group.start] += 0.01; // categorical blocks stay valid one-hot
    nearby.push(overlap(topNames(first), topNames(explainLocal(model, perturbed, raw, { faithfulnessCheck: false }))));
    seedStability.push(overlap(topNames(first), topNames(explainLocal(model, x, raw, { faithfulnessCheck: false, seed: 987654 + r }))));
    if (first.faithfulness.deletion_ratio !== null) ratios.push(first.faithfulness.deletion_ratio);
    if (first.faithfulness.ablation_correlation !== null) correlations.push(first.faithfulness.ablation_correlation);
    if (first.faithfulness.assessment !== 'not_assessable_flat_output') assessable += 1;
    if (first.faithfulness.assessment === 'consistent_with_model') consistent += 1;
    standardErrors.push(first.sampling.max_std_error ?? 0);
  }
  const avg = (values) => (values.length ? round(values.reduce((s, v) => s + v, 0) / values.length, 3) : null);
  return {
    method: explainLocal(model, validationX[0], validationRows[0]?.row, { faithfulnessCheck: false }).method,
    rows_checked: count, data_split: 'validation',
    exact_input_consistent: exact, prediction_reproducible: reproducible,
    similar_input_top_feature_overlap: avg(nearby), sampling_seed_top_feature_overlap: avg(seedStability),
    mean_deletion_ratio: avg(ratios), mean_ablation_correlation: avg(correlations), rows_assessable: assessable, share_assessable_rows_faithful: assessable ? round(consistent / assessable, 3) : null,
    mean_max_std_error: avg(standardErrors),
    note: 'Faithfulness is measured against the trained model (deletion and single-column ablation), not inferred from additivity. Overlap numbers are empirical stability checks.',
  };
}

/** Model-agnostic permutation importance on validation data, per input column (drop in PR-AUC). */
export function permutationImportance(model, X, y, { repeats = 3, maxRows = 4000 } = {}) {
  if (!y.some((label) => label === 1) || !y.some((label) => label === 0)) return null;
  const random = seededRandom(DEFAULT_SEED + 5);
  const indexes = X.length > maxRows ? shuffled([...X.keys()], random).slice(0, maxRows) : [...X.keys()];
  const sample = indexes.map((i) => X[i]); const labels = indexes.map((i) => y[i]);
  const baseline = prAuc(labels, sample.map(model.predictor));
  if (baseline === null) return null;
  return encoderGroups(model.encoder).map((group) => {
    let drop = 0;
    for (let r = 0; r < repeats; r += 1) {
      const order = shuffled([...sample.keys()], random);
      const permuted = sample.map((row, i) => withBlock(row, sample[order[i]], group));
      drop += baseline - prAuc(labels, permuted.map(model.predictor));
    }
    return drop / repeats;
  });
}

/**
 * Global importance for display. Native importance (coefficients / impurity / gain) when the family
 * has one; otherwise validation permutation importance. Returns [] with an explicit method when no
 * real signal exists, so the UI never shows zero placeholders as if they were importances.
 */
export function buildTopFeatures(model, validationX, validationY, limit = 8) {
  const groups = encoderGroups(model.encoder);
  const native = nativeImportance(model.predictor_state, model.encoder.definitions.length);
  const permutation = permutationImportance(model, validationX, validationY);
  const groupSum = (values) => groups.map((group) => { let total = 0; for (let i = group.start; i < group.end; i += 1) total += values[i]; return total; });
  const normalise = (values) => { const clipped = values.map((v) => Math.max(0, v)); const total = clipped.reduce((s, v) => s + v, 0); return total > 0 ? clipped.map((v) => v / total) : null; };

  const nativeShare = native ? normalise(groupSum(native)) : null;
  const permutationShare = permutation ? normalise(permutation) : null;
  const primary = nativeShare || permutationShare;
  const method = nativeShare ? NATIVE_IMPORTANCE_METHOD[model.predictor_state.type] : permutationShare ? 'validation permutation importance (PR-AUC drop)' : 'unavailable';
  if (!primary) return { importance_method: method, top_features: [], note: 'No non-zero importance signal was found (for example a single-leaf tree or a validation set without both classes).' };

  const features = groups.map((group, g) => ({
    feature: group.column,
    importance: round(primary[g], 4),
    weight: model.predictor_state.type === 'logistic' && group.type === 'numeric' ? round(model.predictor_state.weights[group.start], 4) : null,
    permutation_importance: permutationShare ? round(permutationShare[g], 4) : null,
  })).filter((item) => item.importance > 0).sort((a, b) => b.importance - a.importance).slice(0, limit);
  return { importance_method: method, importance_unit: 'share of total importance (sums to 1 over all input columns)', top_features: features };
}
