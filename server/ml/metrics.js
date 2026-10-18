import { round } from './util.js';

/* ---------- Cost matrix (Issue 6) ---------- */

/**
 * Costs are expressed in one consistent currency unit.
 *  - false_negative_cost : fixed loss when a fraud is approved (chargeback fee, investigation, ...).
 *  - false_positive_cost : customer-friction cost when a legitimate transaction is blocked/challenged.
 *  - review_cost         : cost of the review / step-up authentication action. Paid for EVERY flagged
 *                          transaction (true or false positive) because the action happens either way.
 *  - false_negative_use_amount + amount_loss_fraction : optionally add amount * fraction to each missed
 *                          fraud (requires an amount column).
 */
export const DEFAULT_COST_MATRIX = Object.freeze({
  false_negative_cost: 100,
  false_positive_cost: 10,
  review_cost: 1,
  false_negative_use_amount: false,
  amount_loss_fraction: 1,
});

export function normalizeCostMatrix(input = {}) {
  const merged = { ...DEFAULT_COST_MATRIX, ...(input || {}) };
  for (const key of ['false_negative_cost', 'false_positive_cost', 'review_cost', 'amount_loss_fraction']) {
    const value = Number(merged[key]);
    if (!Number.isFinite(value) || value < 0) throw new Error(`Cost matrix field "${key}" must be a non-negative number.`);
    merged[key] = value;
  }
  merged.false_negative_use_amount = Boolean(merged.false_negative_use_amount);
  if (merged.false_negative_cost === 0 && merged.false_positive_cost === 0 && merged.review_cost === 0 && !merged.false_negative_use_amount) {
    throw new Error('Cost matrix cannot be all zero: the threshold would be undefined.');
  }
  return merged;
}

/** Candidate operating thresholds. 1.01 means "flag nothing". */
export const THRESHOLD_GRID = [...Array.from({ length: 99 }, (_, index) => round((index + 1) / 100, 2)), 1.01];

export function costBreakdown(labels, scores, threshold, context) {
  const { cost, amounts } = context;
  const useAmount = cost.false_negative_use_amount && Array.isArray(amounts);
  let fnTotal = 0; let fpTotal = 0; let reviewTotal = 0; let approveAll = 0; let flagged = 0;
  for (let i = 0; i < labels.length; i += 1) {
    const missedFraudCost = cost.false_negative_cost + (useAmount ? Math.max(0, amounts[i] || 0) * cost.amount_loss_fraction : 0);
    if (labels[i] === 1) approveAll += missedFraudCost;
    if (scores[i] >= threshold) {
      flagged += 1;
      reviewTotal += cost.review_cost;
      if (labels[i] === 0) fpTotal += cost.false_positive_cost;
    } else if (labels[i] === 1) {
      fnTotal += missedFraudCost;
    }
  }
  const total = fnTotal + fpTotal + reviewTotal;
  return {
    total_cost: round(total, 2),
    cost_per_1000_transactions: round(total / Math.max(1, labels.length) * 1000, 2),
    missed_fraud_cost: round(fnTotal, 2),
    false_positive_cost: round(fpTotal, 2),
    review_action_cost: round(reviewTotal, 2),
    approve_all_cost: round(approveAll, 2),
    savings_vs_approve_all: round(approveAll - total, 2),
    amount_weighting_applied: useAmount,
    flagged_transactions: flagged,
  };
}

/**
 * Pick the operating threshold that minimises total cost on VALIDATION data.
 * Ties go to the higher threshold (fewer customer interventions); "flag nothing" is only
 * chosen when it is strictly cheaper than every real threshold.
 */
export function chooseThreshold(labels, scores, context) {
  let best = null;
  for (const threshold of THRESHOLD_GRID) {
    const summary = costBreakdown(labels, scores, threshold, context);
    const better = !best || (threshold > 1 ? summary.total_cost < best.summary.total_cost : summary.total_cost <= best.summary.total_cost);
    if (better) best = { threshold, summary };
  }
  return best.threshold;
}

/* ---------- Classification metrics ---------- */

export function rocAuc(labels, scores) {
  const positives = labels.filter(Boolean).length;
  const negatives = labels.length - positives;
  if (!positives || !negatives) return null;
  const ranked = scores.map((score, index) => ({ score, label: labels[index] })).sort((a, b) => a.score - b.score);
  let rankSum = 0;
  for (let i = 0; i < ranked.length;) { // average ranks over tied scores
    let j = i;
    while (j + 1 < ranked.length && ranked[j + 1].score === ranked[i].score) j += 1;
    const averageRank = (i + j + 2) / 2;
    for (let k = i; k <= j; k += 1) if (ranked[k].label) rankSum += averageRank;
    i = j + 1;
  }
  return round((rankSum - positives * (positives + 1) / 2) / (positives * negatives));
}

/** Average precision (step-wise PR area); tied scores are evaluated together. */
export function prAuc(labels, scores) {
  const positives = labels.reduce((sum, label) => sum + label, 0);
  if (!positives) return null;
  const ranked = scores.map((score, index) => ({ score, label: labels[index] })).sort((a, b) => b.score - a.score);
  let tp = 0; let fp = 0; let previousRecall = 0; let area = 0;
  for (let i = 0; i < ranked.length;) {
    let j = i;
    while (j < ranked.length && ranked[j].score === ranked[i].score) { if (ranked[j].label) tp += 1; else fp += 1; j += 1; }
    const recall = tp / positives;
    area += (recall - previousRecall) * (tp / Math.max(1, tp + fp));
    previousRecall = recall;
    i = j;
  }
  return round(area);
}

export function confusion(labels, scores, threshold) {
  let tp = 0; let tn = 0; let fp = 0; let fn = 0;
  for (let i = 0; i < labels.length; i += 1) {
    const flagged = scores[i] >= threshold;
    if (flagged && labels[i] === 1) tp += 1;
    else if (!flagged && labels[i] === 0) tn += 1;
    else if (flagged) fp += 1;
    else fn += 1;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  return {
    threshold,
    accuracy: round((tp + tn) / Math.max(1, labels.length)),
    precision: round(precision),
    recall: round(recall),
    f1: round(precision && recall ? 2 * precision * recall / (precision + recall) : 0),
    false_positive_rate: round(fp / Math.max(1, fp + tn)),
    false_negative_rate: round(fn / Math.max(1, fn + tp)),
    false_positives: fp,
    false_negatives: fn,
    flagged_transactions: tp + fp,
    workload_rate: round((tp + fp) / Math.max(1, labels.length)),
    confusion_matrix: { true_positive: tp, true_negative: tn, false_positive: fp, false_negative: fn },
  };
}

/** `context` = { cost, amounts }. The cost block is included whenever a context is supplied. */
export function evaluate(labels, scores, threshold, context) {
  return {
    ...confusion(labels, scores, threshold),
    roc_auc: rocAuc(labels, scores),
    pr_auc: prAuc(labels, scores),
    prevalence: round(labels.reduce((sum, label) => sum + label, 0) / Math.max(1, labels.length)),
    rows: labels.length,
    ...(context ? { cost: costBreakdown(labels, scores, threshold, context) } : {}),
  };
}

export const REPORT_THRESHOLDS = [0.25, 0.35, 0.45, 0.5, 0.6, 0.7, 0.8];
export function thresholdAnalysis(labels, scores, context, selected) {
  const thresholds = [...new Set([...REPORT_THRESHOLDS, selected].filter((value) => value !== undefined))].sort((a, b) => a - b);
  return thresholds.map((threshold) => ({ ...confusion(labels, scores, threshold), total_cost: costBreakdown(labels, scores, threshold, context).total_cost, selected: threshold === selected }));
}
