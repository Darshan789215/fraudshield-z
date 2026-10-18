import { encoderGroups } from './encoder.js';
import { DEFAULT_SEED, seededRandom, shuffled } from './util.js';

/**
 * SMOTE-NC (Chawla et al. 2002, "SMOTE-NC" variant for mixed numeric / categorical data),
 * implemented in plain JavaScript on the encoded feature space.
 *
 *  1. For each minority "seed" row find its k nearest minority neighbours. Distance =
 *     Euclidean over standardised numeric columns + a penalty (median numeric std of the
 *     minority class) for every categorical column whose category differs.
 *  2. Pick one of the k neighbours at random and place the synthetic row on the line
 *     segment between seed and neighbour for numeric columns (seed + gap * (nb - seed)).
 *  3. Each categorical column takes the MOST COMMON category among the k neighbours
 *     (the seed breaks ties), so a one-hot block is never interpolated and every synthetic
 *     row is a valid encoding.
 *
 * This is the textbook algorithm, but it is a local implementation (not imbalanced-learn).
 * Count-like numeric features (e.g. transactions in 10 minutes) can receive fractional values.
 * For very large minority classes the neighbour search runs over a random pool of at most
 * `maxSearchPool` minority rows to bound the O(m^2) cost.
 */
export function smoteNC(X, y, encoder, options = {}) {
  const { k = 5, targetRatio = 1, seed = DEFAULT_SEED + 17, maxSearchPool = 2000 } = options;
  const positives = []; let negatives = 0;
  y.forEach((label, index) => { if (label === 1) positives.push(index); else negatives += 1; });
  const base = { X, y, synthetic_rows: 0, method: 'SMOTE-NC (custom JS implementation)', k: 0, target_ratio: targetRatio };
  const needed = Math.floor(negatives * targetRatio) - positives.length;
  if (positives.length < 2 || needed <= 0) return { ...base, note: positives.length < 2 ? 'Fewer than 2 minority rows; SMOTE skipped.' : 'Minority class already at the target ratio; SMOTE skipped.' };

  const random = seededRandom(seed);
  const groups = encoderGroups(encoder);
  const numericIndexes = groups.filter((group) => group.type === 'numeric').map((group) => group.start);
  const categoricalGroups = groups.filter((group) => group.type === 'category');
  const minority = positives.map((index) => X[index]);

  // SMOTE-NC penalty: median of the standard deviations of the numeric features in the minority class.
  const stds = numericIndexes.map((feature) => {
    const values = minority.map((row) => row[feature]);
    const average = values.reduce((sum, value) => sum + value, 0) / values.length;
    return Math.sqrt(values.reduce((sum, value) => sum + (value - average) ** 2, 0) / values.length);
  }).sort((a, b) => a - b);
  const penalty = stds.length ? stds[Math.floor(stds.length / 2)] || 1 : 1;

  const categoryOf = (row, group) => { for (let i = group.start; i < group.end; i += 1) if (row[i] === 1) return i - group.start; return -1; };
  const distance = (a, b) => {
    let sum = 0;
    for (const feature of numericIndexes) sum += (a[feature] - b[feature]) ** 2;
    for (const group of categoricalGroups) if (categoryOf(a, group) !== categoryOf(b, group)) sum += penalty ** 2;
    return sum;
  };

  const neighbourK = Math.min(k, minority.length - 1);
  const pool = minority.length > maxSearchPool ? shuffled([...minority.keys()], random).slice(0, maxSearchPool) : [...minority.keys()];
  const neighbourCache = new Map();
  const neighboursOf = (seedIndex) => {
    if (neighbourCache.has(seedIndex)) return neighbourCache.get(seedIndex);
    const ranked = pool.filter((candidate) => candidate !== seedIndex).map((candidate) => ({ candidate, d: distance(minority[seedIndex], minority[candidate]) })).sort((a, b) => a.d - b.d || a.candidate - b.candidate).slice(0, neighbourK).map((item) => item.candidate);
    neighbourCache.set(seedIndex, ranked);
    return ranked;
  };

  const seedOrder = shuffled([...minority.keys()], random);
  const syntheticX = [];
  for (let n = 0; n < needed; n += 1) {
    const seedIndex = seedOrder[n % seedOrder.length];
    const neighbours = neighboursOf(seedIndex);
    const partner = neighbours[Math.floor(random() * neighbours.length)];
    const a = minority[seedIndex]; const b = minority[partner];
    const gap = random();
    const row = a.slice();
    for (const feature of numericIndexes) row[feature] = a[feature] + gap * (b[feature] - a[feature]);
    for (const group of categoricalGroups) {
      const votes = new Map();
      for (const neighbour of neighbours) { const category = categoryOf(minority[neighbour], group); votes.set(category, (votes.get(category) || 0) + 1); }
      let winner = categoryOf(a, group); let winnerVotes = votes.get(winner) || 0;
      for (const [category, count] of votes) if (count > winnerVotes) { winner = category; winnerVotes = count; }
      for (let i = group.start; i < group.end; i += 1) row[i] = i - group.start === winner ? 1 : 0;
    }
    syntheticX.push(row);
  }
  return { ...base, X: [...X, ...syntheticX], y: [...y, ...syntheticX.map(() => 1)], synthetic_rows: syntheticX.length, k: neighbourK, search_pool: pool.length, note: 'Synthetic rows are created from the training partition only.' };
}
