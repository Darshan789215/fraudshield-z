export const MODEL_VERSION = 'fz-prototype-1.0.0';
export const DETERMINISTIC_SEED = 731942;
export const DECISIONS = ['approve', 'step_up', 'hold', 'review_required'];

export const THRESHOLDS = {
  approveMax: 0.39,
  stepUpMax: 0.69,
  holdMin: 0.70
};

export const SCENARIOS = [
  {
    id: 'normal_001',
    label: 'Familiar purchase',
    kind: 'normal',
    description: 'Known device, familiar location, ordinary amount and calm velocity.',
    transaction: {
      transaction_id: 'txn_normal_001', customer_id: 'cust_8842', amount: 84.2, currency: 'USD',
      device_id: 'device_known_17', location: 'Austin, US', merchant_category: 'groceries',
      timestamp: '2026-10-17T12:02:00Z', velocity_10m: 1
    }
  },
  {
    id: 'borderline_001',
    label: 'Needs a second signal',
    kind: 'borderline',
    description: 'A known device meets a new location and a moderately unusual amount.',
    transaction: {
      transaction_id: 'txn_borderline_001', customer_id: 'cust_8842', amount: 286.4, currency: 'USD',
      device_id: 'device_known_17', location: 'Denver, US', merchant_category: 'travel',
      timestamp: '2026-10-17T12:04:00Z', velocity_10m: 2
    }
  },
  {
    id: 'suspicious_001',
    label: 'Burst + new context',
    kind: 'suspicious',
    description: 'New device, unusual location, high amount and burst velocity.',
    transaction: {
      transaction_id: 'txn_suspicious_001', customer_id: 'cust_8842', amount: 1548.8, currency: 'USD',
      device_id: 'device_new_42', location: 'Sofia, BG', merchant_category: 'electronics',
      timestamp: '2026-10-17T12:06:00Z', velocity_10m: 5
    }
  }
];

export const BASELINES = {
  cust_8842: {
    usual_amount: 84.2,
    known_devices: ['device_known_17'],
    known_locations: ['Austin, US'],
    common_categories: ['groceries', 'restaurants', 'fuel'],
    usual_velocity_10m: 1,
    customer_tenure_days: 812
  },
  cust_1098: {
    usual_amount: 142.5,
    known_devices: ['device_known_04', 'device_known_09'],
    known_locations: ['London, UK'],
    common_categories: ['travel', 'restaurants'],
    usual_velocity_10m: 1,
    customer_tenure_days: 441
  }
};

export const MODEL_CARD = {
  model_name: 'FraudShield Z prototype fusion engine',
  model_version: MODEL_VERSION,
  data_source: 'Deterministic synthetic transaction generator',
  deterministic_seed: DETERMINISTIC_SEED,
  feature_scope: 'Amount ratio, device familiarity, location familiarity, category familiarity, velocity and transaction hour',
  intended_use: 'Interactive demonstration of explainable transaction decisioning',
  limitations: 'Synthetic behavior is not production performance. Thresholds and metrics are demo artifacts, not bank policy.',
  supervised_component: 'Calibrated supervised fraud-probability proxy for the prototype',
  anomaly_component: 'Isolation-Forest-style novelty evidence proxy over customer and population context',
  explanation_boundary: 'Factors are model evidence or contribution, not causal proof.'
};
