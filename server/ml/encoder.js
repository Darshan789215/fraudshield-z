import { MAX_CATEGORIES } from './util.js';

export function createEncoder(rows, featureColumns) {
  const definitions = [];
  const means = {};
  const stds = {};
  for (const column of featureColumns) {
    const values = rows.map((row) => row[column]).filter((value) => value !== '' && value !== null && value !== undefined);
    const numericValues = values.map(Number);
    const numeric = values.length > 0 && numericValues.every(Number.isFinite);
    if (numeric) {
      const mean = numericValues.reduce((sum, value) => sum + value, 0) / numericValues.length;
      const std = Math.sqrt(numericValues.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, numericValues.length - 1)) || 1;
      means[column] = mean;
      stds[column] = std;
      definitions.push({ column, type: 'numeric', name: column });
    } else {
      const categories = [...new Set(values.map((value) => String(value).trim()).filter(Boolean))].slice(0, MAX_CATEGORIES);
      categories.forEach((category) => definitions.push({ column, type: 'category', category, name: `${column}=${category}` }));
      definitions.push({ column, type: 'category', category: '__other__', name: `${column}=other` });
    }
  }
  return { definitions, means, stds };
}

/** Exactly one indicator is 1 per categorical column: unseen or missing values go to __other__. */
export function encodeRow(row, encoder) {
  const known = encoder.knownCategories || (encoder.knownCategories = knownCategoryMap(encoder));
  return encoder.definitions.map((definition) => {
    const raw = row[definition.column];
    if (definition.type === 'numeric') {
      const value = Number(raw);
      const safe = raw !== '' && raw !== null && raw !== undefined && Number.isFinite(value) ? value : encoder.means[definition.column];
      return (safe - encoder.means[definition.column]) / encoder.stds[definition.column];
    }
    const text = String(raw ?? '').trim();
    const resolved = text && known.get(definition.column).has(text) ? text : '__other__';
    return resolved === definition.category ? 1 : 0;
  });
}

function knownCategoryMap(encoder) {
  const map = new Map();
  for (const definition of encoder.definitions) {
    if (definition.type !== 'category') continue;
    if (!map.has(definition.column)) map.set(definition.column, new Set());
    if (definition.category !== '__other__') map.get(definition.column).add(definition.category);
  }
  return map;
}

/** Contiguous index ranges per input column (a numeric column = 1 index, a categorical column = its one-hot block). */
export function encoderGroups(encoder) {
  if (encoder.groups) return encoder.groups;
  const groups = [];
  encoder.definitions.forEach((definition, index) => {
    const last = groups.at(-1);
    if (last && last.column === definition.column) last.end = index + 1;
    else groups.push({ column: definition.column, type: definition.type, start: index, end: index + 1 });
  });
  encoder.groups = groups;
  return groups;
}

/** Strip lazily-built caches so the encoder serialises cleanly. */
export function plainEncoder(encoder) {
  return { definitions: encoder.definitions, means: encoder.means, stds: encoder.stds };
}

/** Count encoded vectors that are not a valid encoding (categorical blocks must be exactly one-hot). */
export function countInvalidEncodings(X, encoder) {
  const groups = encoderGroups(encoder).filter((group) => group.type === 'category');
  let invalid = 0;
  for (const vector of X) {
    const bad = groups.some((group) => {
      let sum = 0;
      for (let i = group.start; i < group.end; i += 1) {
        if (vector[i] !== 0 && vector[i] !== 1) return true;
        sum += vector[i];
      }
      return sum !== 1;
    });
    if (bad) invalid += 1;
  }
  return invalid;
}
