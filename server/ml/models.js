import { DEFAULT_SEED, clamp, seededRandom, shuffled, sigmoid } from './util.js';

/**
 * Lightweight, dependency-free classifiers. EVERY model here is a custom implementation:
 * none of them is a validated library (no scikit-learn / native XGBoost). The labels and
 * `implementation_note` fields say so explicitly. Each trainer returns a plain-JSON state,
 * and makePredictor(state) rebuilds the scoring function, so trained and imported models
 * run through exactly the same code path.
 */

/* ---------- Logistic regression (batch gradient descent) ---------- */
function trainLogistic(X, y, { classWeighted = false } = {}) {
  const dimension = X[0].length;
  const weights = Array(dimension).fill(0);
  let bias = 0;
  const positives = y.reduce((sum, label) => sum + label, 0);
  const positiveWeight = classWeighted && positives ? clamp((y.length - positives) / positives, 1, 20, ) : 1;
  const sampleWeights = y.map((label) => (label ? positiveWeight : 1));
  const totalWeight = sampleWeights.reduce((sum, value) => sum + value, 0);
  for (let epoch = 0; epoch < 240; epoch += 1) {
    const gradient = Array(dimension).fill(0);
    let biasGradient = 0;
    for (let row = 0; row < X.length; row += 1) {
      const features = X[row];
      let margin = bias;
      for (let f = 0; f < dimension; f += 1) margin += features[f] * weights[f];
      const error = (sigmoid(margin) - y[row]) * sampleWeights[row];
      for (let f = 0; f < dimension; f += 1) gradient[f] += error * features[f];
      biasGradient += error;
    }
    for (let f = 0; f < dimension; f += 1) weights[f] -= 0.08 * gradient[f] / totalWeight;
    bias -= 0.08 * biasGradient / totalWeight;
  }
  return { type: 'logistic', weights, bias, class_weight_positive: positiveWeight };
}

/* ---------- Gaussian naive Bayes ---------- */
function trainNaiveBayes(X, y) {
  const stats = [0, 1].map((label) => {
    const rows = X.filter((_, index) => y[index] === label);
    const meanVector = X[0].map((_, f) => rows.reduce((sum, row) => sum + row[f], 0) / Math.max(1, rows.length));
    const variance = X[0].map((_, f) => Math.max(0.0001, rows.reduce((sum, row) => sum + (row[f] - meanVector[f]) ** 2, 0) / Math.max(1, rows.length)));
    return { mean: meanVector, variance, prior: rows.length / X.length };
  });
  return { type: 'naive_bayes', stats };
}

/* ---------- CART decision tree / random forest ---------- */
const giniFromCounts = (positives, total) => { if (!total) return 0; const p = positives / total; return 1 - p * p - (1 - p) * (1 - p); };

function buildTree(X, y, indices, depth, maxDepth, random, maxFeatures) {
  const n = indices.length;
  let positives = 0;
  for (const index of indices) positives += y[index];
  const probability = n ? positives / n : 0;
  if (depth >= maxDepth || n < 8 || positives === 0 || positives === n) return { leaf: true, probability, n };
  const parentImpurity = giniFromCounts(positives, n);
  const featureIndexes = shuffled([...Array(X[0].length).keys()], random).slice(0, maxFeatures);
  let best = null;
  for (const feature of featureIndexes) {
    const sorted = [...indices].sort((a, b) => X[a][feature] - X[b][feature]);
    const boundaries = []; const cumulativePositives = []; let running = 0;
    for (let i = 0; i < n; i += 1) {
      running += y[sorted[i]]; cumulativePositives.push(running);
      if (i < n - 1 && X[sorted[i]][feature] !== X[sorted[i + 1]][feature]) boundaries.push(i);
    }
    const step = Math.max(1, Math.ceil(boundaries.length / 32)); // evaluate at most ~32 candidate splits per feature
    for (let b = 0; b < boundaries.length; b += step) {
      const i = boundaries[b]; const leftCount = i + 1; const leftPositives = cumulativePositives[i];
      const gain = parentImpurity - (leftCount * giniFromCounts(leftPositives, leftCount) + (n - leftCount) * giniFromCounts(positives - leftPositives, n - leftCount)) / n;
      if (!best || gain > best.gain) best = { feature, gain, threshold: (X[sorted[i]][feature] + X[sorted[i + 1]][feature]) / 2 };
    }
  }
  if (!best || best.gain < 0.001) return { leaf: true, probability, n };
  const left = indices.filter((index) => X[index][best.feature] <= best.threshold);
  const right = indices.filter((index) => X[index][best.feature] > best.threshold);
  if (!left.length || !right.length) return { leaf: true, probability, n };
  return {
    feature: best.feature, threshold: best.threshold, gain: best.gain, n,
    left: buildTree(X, y, left, depth + 1, maxDepth, random, maxFeatures),
    right: buildTree(X, y, right, depth + 1, maxDepth, random, maxFeatures),
  };
}

function trainTree(X, y) {
  const random = seededRandom(DEFAULT_SEED + 3);
  return { type: 'tree', tree: buildTree(X, y, [...X.keys()], 0, 4, random, X[0].length) };
}

function trainForest(X, y) {
  const random = seededRandom(DEFAULT_SEED);
  const maxFeatures = Math.max(1, Math.ceil(Math.sqrt(X[0].length)));
  const trees = [];
  for (let t = 0; t < 17; t += 1) {
    const bootstrap = Array.from({ length: X.length }, () => Math.floor(random() * X.length));
    trees.push(buildTree(X, y, bootstrap, 0, 5, random, maxFeatures));
  }
  return { type: 'forest', trees };
}

const predictTree = (node, features) => { let current = node; while (!current.leaf) current = features[current.feature] <= current.threshold ? current.left : current.right; return current.probability; };

/* ---------- Gradient-boosted stumps (second-order, logistic loss) ---------- */
// Custom approximation of gradient boosting: depth-1 trees, Newton leaf values with L2 shrinkage.
// It is NOT the XGBoost library: no column/row subsampling, no depth > 1, no regularised tree growth, no sparsity handling.
function trainBoostedStumps(X, y) {
  const rounds = 40; const learningRate = 0.2; const lambda = 1;
  const positives = y.reduce((sum, label) => sum + label, 0);
  const base = Math.log((positives + 0.5) / (y.length - positives + 0.5));
  const margin = Array(y.length).fill(base);
  const dimension = X[0].length;
  const order = Array.from({ length: dimension }, (_, f) => [...X.keys()].sort((a, b) => X[a][f] - X[b][f])); // sort once per feature
  const stumps = [];
  for (let round = 0; round < rounds; round += 1) {
    const gradient = margin.map((value, i) => y[i] - sigmoid(value)); // -dLoss/dMargin
    const hessian = margin.map((value) => { const p = sigmoid(value); return Math.max(1e-6, p * (1 - p)); });
    const totalG = gradient.reduce((sum, v) => sum + v, 0); const totalH = hessian.reduce((sum, v) => sum + v, 0);
    let best = null;
    for (let f = 0; f < dimension; f += 1) {
      const sorted = order[f]; let leftG = 0; let leftH = 0;
      for (let i = 0; i < sorted.length - 1; i += 1) {
        leftG += gradient[sorted[i]]; leftH += hessian[sorted[i]];
        if (X[sorted[i]][f] === X[sorted[i + 1]][f]) continue;
        const rightG = totalG - leftG; const rightH = totalH - leftH;
        const gain = 0.5 * (leftG ** 2 / (leftH + lambda) + rightG ** 2 / (rightH + lambda) - totalG ** 2 / (totalH + lambda));
        if (!best || gain > best.gain) best = { feature: f, threshold: (X[sorted[i]][f] + X[sorted[i + 1]][f]) / 2, gain, leftValue: leftG / (leftH + lambda), rightValue: rightG / (rightH + lambda) };
      }
    }
    if (!best || best.gain <= 1e-9) break;
    stumps.push(best);
    for (let i = 0; i < margin.length; i += 1) margin[i] += learningRate * (X[i][best.feature] <= best.threshold ? best.leftValue : best.rightValue);
  }
  return { type: 'boosted_stumps', base, learning_rate: learningRate, stumps };
}

/* ---------- Registry ---------- */
export const ALGORITHMS = {
  logistic_regression: {
    label: 'Logistic regression (custom)', family: 'baseline', train: trainLogistic,
    implementation_note: 'Custom batch gradient-descent logistic regression (240 epochs, lr 0.08) on standardised / one-hot features.',
    tradeoff: 'Fast and interpretable; linear decision boundary.',
  },
  gaussian_nb: {
    label: 'Gaussian naive Bayes (custom)', family: 'baseline', train: trainNaiveBayes,
    implementation_note: 'Custom Gaussian naive Bayes. One-hot columns are treated as Gaussian, which is a rough approximation.',
    tradeoff: 'Very fast; assumes conditionally independent features.',
  },
  decision_tree: {
    label: 'Decision tree (custom CART)', family: 'baseline', train: trainTree,
    implementation_note: 'Custom CART with Gini impurity, depth 4, at most ~32 candidate splits per feature.',
    tradeoff: 'Readable nonlinear rules; can overfit without constraints.',
  },
  random_forest: {
    label: 'Random forest (custom, 17 trees)', family: 'ensemble', train: trainForest,
    implementation_note: 'Custom bagged forest: 17 depth-5 CART trees, bootstrap rows, sqrt(features) per split.',
    tradeoff: 'Strong nonlinear ensemble; higher training cost and less transparent than a single tree.',
  },
  gradient_boosted_stumps: {
    label: 'Gradient-boosted stumps (custom, not XGBoost)', family: 'ensemble', train: trainBoostedStumps,
    implementation_note: 'Custom second-order gradient boosting of depth-1 trees (40 rounds, lr 0.2, L2 lambda 1). This is a simplified approximation and NOT the native XGBoost algorithm; do not compare its numbers to XGBoost results.',
    tradeoff: 'Boosted stumps capture additive effects; no deep interactions; less transparent than a linear model.',
  },
};
/** Old API name kept so existing clients keep working. */
export const ALGORITHM_ALIASES = { xgboost: 'gradient_boosted_stumps' };
export const resolveAlgorithm = (name) => ALGORITHM_ALIASES[name] || name;

export function trainModel(algorithm, X, y, options) {
  const implementation = ALGORITHMS[algorithm];
  if (!implementation) throw new Error(`Unsupported algorithm: ${algorithm}`);
  return implementation.train(X, y, options);
}

/** Rebuild the scoring function from a serialisable state. */
export function makePredictor(state) {
  switch (state.type) {
    case 'logistic': return (features) => sigmoid(logisticMargin(state, features));
    case 'naive_bayes': return (features) => {
      const logProbability = (label) => state.stats[label].mean.reduce((sum, mean, i) => sum - 0.5 * Math.log(2 * Math.PI * state.stats[label].variance[i]) - ((features[i] - mean) ** 2) / (2 * state.stats[label].variance[i]), Math.log(state.stats[label].prior || 0.0001));
      return sigmoid(logProbability(1) - logProbability(0));
    };
    case 'tree': return (features) => predictTree(state.tree, features);
    case 'forest': return (features) => state.trees.reduce((sum, tree) => sum + predictTree(tree, features), 0) / state.trees.length;
    case 'boosted_stumps': return (features) => sigmoid(state.stumps.reduce((value, stump) => value + state.learning_rate * (features[stump.feature] <= stump.threshold ? stump.leftValue : stump.rightValue), state.base));
    default: throw new Error(`Unknown predictor state: ${state.type}`);
  }
}

export function logisticMargin(state, features) {
  let margin = state.bias;
  for (let i = 0; i < features.length; i += 1) margin += features[i] * state.weights[i];
  return margin;
}

/**
 * Native (training-time) importance per ENCODED feature index, or null when the model
 * family has no native importance (naive Bayes). Trees/stumps use accumulated impurity /
 * loss reduction; logistic regression uses |standardised coefficient|.
 */
export function nativeImportance(state, dimension) {
  const totals = Array(dimension).fill(0);
  const walk = (node) => { if (node.leaf) return; totals[node.feature] += node.gain * node.n; walk(node.left); walk(node.right); };
  switch (state.type) {
    case 'logistic': return state.weights.map(Math.abs);
    case 'tree': walk(state.tree); return totals;
    case 'forest': state.trees.forEach(walk); return totals.map((value) => value / state.trees.length);
    case 'boosted_stumps': state.stumps.forEach((stump) => { totals[stump.feature] += stump.gain; }); return totals;
    default: return null;
  }
}

export const NATIVE_IMPORTANCE_METHOD = {
  logistic: 'absolute standardised coefficient',
  tree: 'total Gini impurity decrease',
  forest: 'mean total Gini impurity decrease across trees',
  boosted_stumps: 'total second-order loss reduction (gain)',
};

/* ---------- Isolation forest (unsupervised anomaly experiment) ---------- */
// Custom simplified isolation forest: 32 trees, 128-row subsamples. Does not use labels.
function isolationTree(rows, depth, maxDepth, random) {
  if (rows.length <= 1 || depth >= maxDepth) return { leaf: true, size: rows.length };
  const feature = Math.floor(random() * rows[0].length);
  let min = Infinity; let max = -Infinity;
  for (const row of rows) { if (row[feature] < min) min = row[feature]; if (row[feature] > max) max = row[feature]; }
  if (min === max) return { leaf: true, size: rows.length };
  const split = min + random() * (max - min);
  const left = rows.filter((row) => row[feature] < split);
  const right = rows.filter((row) => row[feature] >= split);
  if (!left.length || !right.length) return { leaf: true, size: rows.length };
  return { feature, split, left: isolationTree(left, depth + 1, maxDepth, random), right: isolationTree(right, depth + 1, maxDepth, random) };
}
const averagePathLength = (n) => (n > 1 ? 2 * (Math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n : 0);
function pathLength(node, row, depth = 0) {
  if (node.leaf) return depth + averagePathLength(node.size);
  return pathLength(row[node.feature] < node.split ? node.left : node.right, row, depth + 1);
}
export function anomalyScores(trainX, targetX) {
  const random = seededRandom(DEFAULT_SEED + 99);
  const sampleSize = Math.min(128, trainX.length);
  const maxDepth = Math.ceil(Math.log2(Math.max(2, sampleSize)));
  const trees = Array.from({ length: 32 }, () => isolationTree(Array.from({ length: sampleSize }, () => trainX[Math.floor(random() * trainX.length)]), 0, maxDepth, random));
  const c = averagePathLength(sampleSize) || 1;
  return targetX.map((row) => clamp(2 ** (-(trees.reduce((sum, tree) => sum + pathLength(tree, row), 0) / trees.length) / c)));
}
