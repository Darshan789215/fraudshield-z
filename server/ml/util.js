export const MAX_ROWS = 300000;
export const MAX_CATEGORIES = 64;
export const DEFAULT_SEED = 731942;

export const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));
export const round = (value, digits = 3) => Number((Number(value) || 0).toFixed(digits));
export const sigmoid = (value) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, value))));
export const mean = (values) => (values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0);

/** Small deterministic LCG so every run with the same seed is reproducible. */
export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/** Unbiased Fisher-Yates shuffle (returns a new array). */
export function shuffled(items, random) {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

export function parseCSV(text) {
  const rows = [];
  const firstLine = String(text).split(/\r?\n/, 1)[0] || '';
  const delimiter = [',', ';', '\t', '|'].sort((a, b) => firstLine.split(b).length - firstLine.split(a).length)[0];
  let row = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && quoted && next === '"') { cell += '"'; i += 1; continue; }
    if (char === '"') { quoted = !quoted; continue; }
    if (char === delimiter && !quoted) { row.push(cell.trim()); cell = ''; continue; }
    if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && next === '\n') i += 1;
      row.push(cell.trim());
      cell = '';
      if (row.some((value) => value !== '')) rows.push(row);
      row = [];
      continue;
    }
    cell += char;
  }
  if (cell || row.length) { row.push(cell.trim()); if (row.some((value) => value !== '')) rows.push(row); }
  if (rows.length < 2) throw new Error('Upload a CSV with a header row and at least one data row.');
  const headers = rows.shift().map((header, index) => header || `column_${index + 1}`);
  return rows.map((values, index) => ({ ...Object.fromEntries(headers.map((header, column) => [header, values[column] ?? ''])), __row_index: index }));
}

export function toLabel(value) {
  const normalized = String(value).trim().toLowerCase();
  if (['1', 'true', 'yes', 'fraud', 'fraudulent', 'positive', 'bad', 'anomaly', 'charged back', 'chargeback'].includes(normalized)) return 1;
  if (['0', 'false', 'no', 'legitimate', 'legit', 'negative', 'good', 'normal', 'not fraud'].includes(normalized)) return 0;
  const numeric = Number(normalized);
  return Number.isFinite(numeric) ? (numeric > 0 ? 1 : 0) : null;
}

export function detectLabelColumn(headers, requested) {
  if (requested && headers.includes(requested)) return requested;
  const exact = headers.find((header) => /^(label|target|class|fraud|is_fraud|fraud_flag|outcome|y|response|default|bad_flag|churn)$/i.test(header));
  if (exact) return exact;
  const semantic = headers.find((header) => /(fraud|label|target|class|outcome|default|chargeback|response|churn)/i.test(header));
  return semantic || headers.find((header) => header !== '__row_index') || headers[headers.length - 1];
}

export function detectTimeColumn(headers, requested) {
  if (requested && headers.includes(requested)) return requested;
  return headers.find((header) => /(timestamp|datetime|date|time|created|occurred|transaction_time|event_time)/i.test(header)) || null;
}

export function temporalValue(value, fallback) {
  const numeric = Number(value);
  if (String(value ?? '').trim() !== '' && Number.isFinite(numeric)) return numeric;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}
