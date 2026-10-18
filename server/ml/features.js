import { mean, temporalValue } from './util.js';

/**
 * Stateful, past-only feature pipeline shared by training, replay and inference.
 * Every relationship fact stores its observation time so backdated queries cannot
 * see owners, merchants or account/merchant activity observed later.
 */
export const BEHAVIOR_FEATURES = [
  'behavior_txn_count_10m', 'behavior_avg_spend_7d', 'behavior_amount_deviation', 'behavior_new_device',
  'relationship_shared_device_count', 'relationship_repeated_merchant_count',
  'graph_device_degree', 'graph_merchant_degree', 'graph_account_merchant_repeat',
];
export const GRAPH_FEATURES = [
  'relationship_shared_device_count', 'relationship_repeated_merchant_count',
  'graph_device_degree', 'graph_merchant_degree', 'graph_account_merchant_repeat',
];

const MS_10_MIN = 10 * 60 * 1000;
const MS_7_DAYS = 7 * 24 * 60 * 60 * 1000;
const detect = (headers, patterns) => headers.find((header) => patterns.some((pattern) => pattern.test(header))) || null;

export function detectFeatureConfig(headers, { labelColumn, timeColumn } = {}) {
  const candidates = headers.filter((header) => header !== '__row_index' && header !== labelColumn && header !== timeColumn);
  return {
    customer_column: detect(candidates, [/customer/i, /user/i, /account/i, /card/i, /client/i]),
    amount_column: detect(candidates, [/amount/i, /value/i, /price/i, /spend/i, /transaction_value/i]),
    device_column: detect(candidates, [/device/i, /browser/i, /fingerprint/i]),
    merchant_column: detect(candidates, [/merchant/i, /seller/i, /store/i, /category/i]),
    time_column: timeColumn || null,
    window_10m: timeColumn ? MS_10_MIN : 10,
    window_7d: timeColumn ? MS_7_DAYS : 7 * 24 * 60 * 60,
  };
}

export function createFeatureState() {
  return { sequence: 0, last_time: null, customers: new Map(), deviceOwners: new Map(), merchantCounts: new Map(), merchantCustomers: new Map(), accountMerchant: new Map() };
}

export function rowTime(config, row, fallback) {
  if (config.time_column) return temporalValue(row[config.time_column], fallback);
  return fallback;
}

export function inferenceFallbackTime(state, row) {
  const index = Number(row?.__row_index);
  if (Number.isFinite(index)) return index;
  return state.last_time ?? state.sequence;
}

function keys(config, row) {
  // Unknown identity is not one shared customer: pooling unrelated rows creates false history and O(n²) scans.
  const customerValue = config.customer_column ? row[config.customer_column] : null;
  const customer = customerValue == null || String(customerValue).trim() === '' ? '' : String(customerValue);
  const rawAmount = config.amount_column ? Number(row[config.amount_column]) : NaN;
  const amount = Number.isFinite(rawAmount) ? rawAmount : 0;
  const device = config.device_column ? String(row[config.device_column] || '') : '';
  const merchant = config.merchant_column ? String(row[config.merchant_column] || '') : '';
  return { customer, amount, device, merchant };
}

const countAtOrBefore = (sortedTimes, time) => {
  if (!sortedTimes) return 0;
  let low = 0; let high = sortedTimes.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (sortedTimes[mid] <= time) low = mid + 1;
    else high = mid;
  }
  return low;
};
const countOwnersAtOrBefore = (owners, time) => owners ? [...owners.values()].filter((firstSeen) => firstSeen <= time).length : 0;
const firstSeen = (map, key, time) => { if (!map.has(key) || time < map.get(key)) map.set(key, time); };
const insertTime = (map, key, time) => {
  const values = map.get(key) || [];
  let low = 0; let high = values.length;
  while (low < high) { const mid = (low + high) >>> 1; if (values[mid] <= time) low = mid + 1; else high = mid; }
  values.splice(low, 0, time);
  map.set(key, values);
};

export function computeFeatures(config, state, row, time) {
  const { customer, amount, device, merchant } = keys(config, row);
  const profile = customer ? state.customers.get(customer) : null;
  const records = (profile?.records || []).filter((record) => record.timestamp <= time);
  let recent10m = 0;
  const recentAmounts = [];
  let knownDevice = false;
  for (const record of records) {
    const age = time - record.timestamp;
    if (age <= config.window_10m) recent10m += 1;
    if (age <= config.window_7d) recentAmounts.push(record.amount);
    if (device && record.device === device) knownDevice = true;
  }
  const average = recentAmounts.length ? mean(recentAmounts) : amount;
  const owners = device && customer ? state.deviceOwners.get(device) : null;
  const ownerCount = countOwnersAtOrBefore(owners, time);
  const ownersExcludingSelf = ownerCount - (owners?.has(customer) && owners.get(customer) <= time ? 1 : 0);
  const merchantOwners = merchant && customer ? state.merchantCustomers.get(merchant) : null;
  return {
    behavior_txn_count_10m: recent10m,
    behavior_avg_spend_7d: average,
    behavior_amount_deviation: average ? Math.abs(amount - average) / Math.max(1, average) : 0,
    behavior_new_device: records.length ? (knownDevice ? 0 : 1) : 0,
    relationship_shared_device_count: Math.max(0, ownersExcludingSelf),
    relationship_repeated_merchant_count: merchant ? countAtOrBefore(state.merchantCounts.get(merchant), time) : 0,
    graph_device_degree: ownerCount,
    graph_merchant_degree: countOwnersAtOrBefore(merchantOwners, time),
    graph_account_merchant_repeat: merchant && customer ? countAtOrBefore(state.accountMerchant.get(`${customer}::${merchant}`), time) : 0,
  };
}

export function enrichRow(config, state, row, time) {
  return { ...row, ...computeFeatures(config, state, row, time) };
}

export function observe(config, state, row, time) {
  const { customer, amount, device, merchant } = keys(config, row);
  if (customer) {
    let profile = state.customers.get(customer);
    if (!profile) { profile = { records: [] }; state.customers.set(customer, profile); }
    const record = { timestamp: time, amount, device };
    let low = 0; let high = profile.records.length;
    while (low < high) { const mid = (low + high) >>> 1; if (profile.records[mid].timestamp <= time) low = mid + 1; else high = mid; }
    profile.records.splice(low, 0, record);
    if (device) {
      if (!state.deviceOwners.has(device)) state.deviceOwners.set(device, new Map());
      firstSeen(state.deviceOwners.get(device), customer, time);
    }
  }
  if (merchant) {
    insertTime(state.merchantCounts, merchant, time);
    if (customer) {
      if (!state.merchantCustomers.has(merchant)) state.merchantCustomers.set(merchant, new Map());
      firstSeen(state.merchantCustomers.get(merchant), customer, time);
      insertTime(state.accountMerchant, `${customer}::${merchant}`, time);
    }
  }
  state.sequence += 1;
  state.last_time = state.last_time === null ? time : Math.max(state.last_time, time);
}

export function serializeState(state) {
  return {
    sequence: state.sequence, last_time: state.last_time,
    customers: [...state.customers].map(([key, value]) => [key, { records: value.records }]),
    deviceOwners: [...state.deviceOwners].map(([key, value]) => [key, [...value]]),
    merchantCounts: [...state.merchantCounts],
    merchantCustomers: [...state.merchantCustomers].map(([key, value]) => [key, [...value]]),
    accountMerchant: [...state.accountMerchant],
  };
}

export function deserializeState(data) {
  const state = createFeatureState();
  if (!data) return state;
  state.sequence = data.sequence || 0;
  state.last_time = data.last_time ?? null;
  for (const [key, value] of data.customers || []) state.customers.set(key, { records: value.records || [] });
  for (const [key, value] of data.deviceOwners || []) state.deviceOwners.set(key, new Map(value));
  for (const [key, value] of data.merchantCounts || []) state.merchantCounts.set(key, Array.isArray(value) ? value : []);
  for (const [key, value] of data.merchantCustomers || []) state.merchantCustomers.set(key, new Map(value));
  for (const [key, value] of data.accountMerchant || []) state.accountMerchant.set(key, Array.isArray(value) ? value : []);
  return state;
}

export function buildFeatures(config, items, state = createFeatureState()) {
  return items.map((item) => {
    const enriched = enrichRow(config, state, item.row, item.time);
    observe(config, state, item.row, item.time);
    return { ...item, row: enriched };
  });
}
