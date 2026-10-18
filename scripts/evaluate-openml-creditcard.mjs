import { createHash } from 'node:crypto';
import { createWriteStream, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { pipeline } from 'node:stream/promises';
import { Readable } from 'node:stream';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { trainDataset } from '../server/ml.js';
import { DEFAULT_SEED } from '../server/ml/util.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DATA_DIR = path.join(ROOT, 'data', 'benchmark');
const REPORT_DIR = path.join(ROOT, 'reports');
const ARFF_PATH = process.env.BENCHMARK_ARFF || path.join(DATA_DIR, 'CreditCardFraudDetection.arff');
const OPENML_URL = 'https://www.openml.org/data/download/21829702/dataset';
const EXPECTED_MD5 = 'f9544642717dc57d30c6505a422da240';
const EXPECTED_ROWS = 284807;
const EXPECTED_FRAUD = 492;
const COST_MATRIX = { false_negative_cost: 100, false_positive_cost: 10, review_cost: 1, false_negative_use_amount: false };

function digestFile(file, algorithm) {
  const hash = createHash(algorithm);
  hash.update(readFileSync(file));
  return hash.digest('hex');
}

async function ensureDataset() {
  if (existsSync(ARFF_PATH)) return;
  mkdirSync(path.dirname(ARFF_PATH), { recursive: true });
  console.log(`Downloading the public OpenML ARFF dataset to ${ARFF_PATH}`);
  const response = await fetch(OPENML_URL, { redirect: 'follow' });
  if (!response.ok || !response.body) throw new Error(`Dataset download failed: HTTP ${response.status}`);
  await pipeline(Readable.fromWeb(response.body), createWriteStream(ARFF_PATH));
}

function readArffAsCsv(file) {
  console.log('Reading and validating dataset metadata/checksum…');
  const md5 = digestFile(file, 'md5');
  if (md5 !== EXPECTED_MD5) throw new Error(`OpenML checksum mismatch: expected ${EXPECTED_MD5}, received ${md5}`);
  const sha256 = digestFile(file, 'sha256');
  const text = readFileSync(file, 'utf8').replace(/^\uFEFF/, '');
  const lines = text.split(/\r?\n/);
  const columns = [];
  let dataStarted = false;
  const dataRows = [];
  for (const raw of lines) {
    const line = raw.trim();
    if (!line || line.startsWith('%')) continue;
    if (!dataStarted) {
      if (/^@attribute\s/i.test(line)) {
        const match = /^@attribute\s+(?:"([^"]+)"|'([^']+)'|(\S+))\s+/i.exec(line);
        if (!match) throw new Error(`Could not parse ARFF attribute declaration: ${line}`);
        columns.push(match[1] || match[2] || match[3]);
      } else if (/^@data\s*$/i.test(line)) dataStarted = true;
      continue;
    }
    dataRows.push(line);
  }
  if (!dataStarted || !columns.length) throw new Error('The downloaded file is not a readable ARFF dataset.');
  const labelIndex = columns.indexOf('Class');
  if (labelIndex < 0 || !columns.includes('Time')) throw new Error('Expected the documented Time and Class fields are missing.');
  let fraudCount = 0;
  for (let index = 0; index < dataRows.length; index += 1) {
    const row = dataRows[index].split(',');
    if (row.length !== columns.length) throw new Error(`ARFF row ${index + 1} has ${row.length} fields; expected ${columns.length}.`);
    const label = row[labelIndex].trim().toLowerCase();
    if (['1', '1.0', 'true', 'yes'].includes(label)) fraudCount += 1;
    else if (!['0', '0.0', 'false', 'no'].includes(label)) throw new Error(`Unrecognized target value at ARFF row ${index + 1}: ${row[labelIndex]}`);
  }
  if (dataRows.length !== EXPECTED_ROWS || fraudCount !== EXPECTED_FRAUD) {
    throw new Error(`Unexpected dataset dimensions or labels: rows=${dataRows.length}, fraud=${fraudCount}; expected ${EXPECTED_ROWS} and ${EXPECTED_FRAUD}.`);
  }
  console.log(`Verified ${dataRows.length.toLocaleString()} rows, ${fraudCount} fraud labels, ${columns.length} columns; SHA-256 ${sha256}`);
  return { csv: `${columns.join(',')}\n${dataRows.join('\n')}`, columns, rows: dataRows.length, fraudCount, md5, sha256 };
}

function metricSummary(value) {
  if (!value) return null;
  return {
    threshold: value.threshold,
    rows: value.rows,
    prevalence: value.prevalence,
    pr_auc: value.pr_auc,
    roc_auc: value.roc_auc,
    precision: value.precision,
    fraud_recall: value.recall,
    f1: value.f1,
    false_positives: value.false_positives,
    false_negatives: value.false_negatives,
    confusion_matrix: value.confusion_matrix,
    cost: value.cost,
  };
}

function makeMarkdown(report) {
  return `# OpenML Credit Card Fraud Evaluation\n\n` +
    `Evaluation date: ${report.evaluated_at}\n\n` +
    `## Dataset and scope\n\n` +
    `This run uses the public OpenML CreditCardFraudDetection dataset (ID ${report.dataset.openml_id}), described as anonymized European card transactions from September 2013. It covers two days and is extremely imbalanced. It is one public benchmark, not a representative modern bank portfolio. Features V1–V28 are PCA transformed; only Time and Amount retain their original meanings [1] [2].\n\n` +
    `Rows: ${report.dataset.rows}; fraud rows: ${report.dataset.fraud_rows}; columns: ${report.dataset.columns}; dataset MD5: \`${report.dataset.md5}\`; SHA-256: \`${report.dataset.sha256}\`. The OpenML ARFF declares Class as True/False; the runner normalizes these labels to binary values. The model receives Time as its chronological ordering field and Class as its target. Customer, device and merchant identifiers are absent, so account-specific behavior and relationship features cannot be validated. The data are not redistributed in the source archive.\n\n` +
    `Dataset license: the Kaggle data card lists Open Database, Contents Database Contents License 1.0; check that source page and license before reuse or redistribution [2].\n\n` +
    `Software: Node ${report.environment.node_version}, ${report.environment.platform}/${report.environment.architecture}, seed ${report.environment.seed}; package-lock SHA-256 ${report.environment.package_lock_sha256}.\n\n` +
    `## Protocol\n\n` +
    `The project pipeline sorts rows by Time and uses the fixed chronological 60% train / 20% validation / 20% test split. Logistic regression is the only requested candidate for this run; the validation period is used for model threshold selection and operating-threshold calibration. The final test period is evaluated once after selection. The primary operating metrics below are for the integrated hybrid decision score; the classifier-only test metrics are included as a comparator. No resampling is used for the selected classifier. Separate ablation/anomaly/graph experiments are skipped for this single-candidate benchmark. Cost weights are illustrative, not calibrated from a bank: missed fraud = 100 units, false positive = 10 units, review/step-up = 1 unit; amount is not added to missed-fraud cost.\n\n` +
    `## Held-out test results\n\n` +
    `The hybrid decision was evaluated on ${report.test.hybrid.rows.toLocaleString()} chronological test rows, including ${report.test.hybrid.confusion_matrix.true_positive + report.test.hybrid.false_negatives} fraud cases. Its step-wise PR-AUC (average precision) was ${report.test.hybrid.pr_auc}; fraud recall was ${report.test.hybrid.fraud_recall}; precision was ${report.test.hybrid.precision}; false positives were ${report.test.hybrid.false_positives}; false negatives were ${report.test.hybrid.false_negatives}. The confusion counts were ${report.test.hybrid.confusion_matrix.true_positive} true positives and ${report.test.hybrid.confusion_matrix.true_negative} true negatives. The validation-selected hybrid threshold was ${report.test.hybrid.threshold}. Under the illustrative cost matrix, total cost was ${report.test.hybrid.cost.total_cost} units, or ${report.test.hybrid.cost.cost_per_1000_transactions} units per 1,000 transactions; missed-fraud, false-positive and review costs were ${report.test.hybrid.cost.missed_fraud_cost}, ${report.test.hybrid.cost.false_positive_cost}, and ${report.test.hybrid.cost.review_action_cost} units, respectively.\n\n` +
    `The classifier-only comparator had PR-AUC ${report.test.classifier.pr_auc}, recall ${report.test.classifier.fraud_recall}, ${report.test.classifier.false_positives} false positives, ${report.test.classifier.false_negatives} false negatives, and total cost ${report.test.classifier.cost.total_cost} units at threshold ${report.test.classifier.threshold}. Its results match the hybrid decision on this dataset because customer/device/merchant identifiers are absent and no entity-level relationship signals can be computed. These are point estimates; no confidence interval was calculated.\n\n` +
    `The chronological boundaries are training period end ${report.protocol.train_period_end}, validation period end ${report.protocol.validation_period_end}, and test period start ${report.protocol.test_period_start}, measured in the dataset's elapsed-seconds Time field.\n\n` +
    `## Interpretation\n\n` +
    `These are held-out results on one historical, anonymized public dataset, under the specified split and illustrative cost matrix. They do not establish current, cross-bank, or real-world deployment performance. The two-day time horizon, anonymized PCA features, and missing customer/device/merchant identifiers limit operational interpretation. No LinuxONE deployment or independent security assessment is implied by this evaluation.\n\n` +
    `## References\n\n` +
    `[1]: https://www.openml.org/d/42397 "OpenML CreditCardFraudDetection dataset"\n\n` +
    `[2]: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "Kaggle ULB Credit Card Fraud dataset data card"\n`;
}

await ensureDataset();
const dataset = readArffAsCsv(ARFF_PATH);
console.log('Starting chronological single-candidate training/evaluation; optional ablation/anomaly/graph experiments are skipped.');
const output = trainDataset(dataset.csv, {
  labelColumn: 'Class', timeColumn: 'Time', algorithms: ['logistic_regression'], costMatrix: COST_MATRIX, skipExploratoryExperiments: true,
});
const model = output.model;
const report = {
  schema_version: 'openml-creditcard-eval-v1',
  evaluated_at: new Date().toISOString(),
  environment: { node_version: process.version, platform: process.platform, architecture: process.arch, seed: DEFAULT_SEED, package_lock_sha256: digestFile(path.join(ROOT, 'package-lock.json'), 'sha256') },
  dataset: {
    name: 'CreditCardFraudDetection', openml_id: 42397, rows: dataset.rows, fraud_rows: dataset.fraudCount,
    columns: dataset.columns.length, label_column: 'Class', label_encoding: 'OpenML ARFF True/False normalized to binary 1/0', time_column: 'Time', md5: dataset.md5, sha256: dataset.sha256,
    source_url: 'https://www.openml.org/d/42397', download_url: OPENML_URL,
    origin_url: 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud',
    license_note: 'Kaggle data card lists Open Database, Contents Database Contents License 1.0; review the source license before reuse or redistribution.',
    scope_note: 'Anonymized European card transactions from September 2013 over two days; no customer/device/merchant identifiers; not a current or representative portfolio.',
  },
  protocol: {
    split: model.evaluation_protocol.split,
    ordering: 'Ascending Time, ties stable by source row order.',
    selection: model.evaluation_protocol.model_selection,
    threshold_selection: model.evaluation_protocol.threshold_selection,
    test_usage: model.evaluation_protocol.test_usage,
    algorithm: 'logistic_regression (only candidate requested)',
    exploratory_experiments: 'Skipped; no effect on selected-candidate evaluation.',
    cost_matrix: COST_MATRIX,
    cost_units_note: 'Illustrative relative cost units; not calibrated to a bank or a particular currency.',
    train_period_end: model.evaluation_protocol.train_period_end,
    validation_period_end: model.evaluation_protocol.validation_period_end,
    test_period_start: model.evaluation_protocol.test_period_start,
    training_rows: Math.floor(dataset.rows * 0.6),
    validation_rows: Math.floor(dataset.rows * 0.8) - Math.floor(dataset.rows * 0.6),
    test_rows: dataset.rows - Math.floor(dataset.rows * 0.8),
  },
  selected_model: { algorithm: model.algorithm, threshold: model.threshold, hybrid_threshold: model.hybrid_threshold, positive_count: model.positive_count },
  test: { hybrid: metricSummary(model.hybrid_test_metrics), classifier: metricSummary(model.test_metrics) },
  validation: { hybrid: metricSummary(model.hybrid_validation_metrics), classifier: metricSummary(model.selection_metrics) },
  interpretation: 'Public historical benchmark only. No claim of current production accuracy, transfer to other portfolios, validation of entity-level relationship features, IBM Z/LinuxONE deployment, or security certification.',
};
mkdirSync(REPORT_DIR, { recursive: true });
const jsonPath = path.join(REPORT_DIR, 'openml-creditcard-evaluation.json');
const markdownPath = path.join(REPORT_DIR, 'openml-creditcard-evaluation.md');
writeFileSync(jsonPath, `${JSON.stringify(report, null, 2)}\n`);
writeFileSync(markdownPath, makeMarkdown(report));
console.log(`Evaluation saved to ${markdownPath} and ${jsonPath}`);
console.log(JSON.stringify({ test: report.test, protocol: report.protocol, selected_model: report.selected_model }, null, 2));
