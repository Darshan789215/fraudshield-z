# FraudShield Z ML Workbench

FraudShield Z is a simple, explainable fraud-ML workbench with a live fraud-monitoring companion. Upload a labelled CSV, preview the data, train and compare five benchmark classifiers, inspect future-period evaluation metrics and learned signals, then score a new transaction row with the selected model. Below the training flow, explore the original deterministic simulations and a live server-sent-event rail so the project works as both a training studio and an interactive fraud demo.

The interface follows the supplied video reference: a dark glow-morphism workspace, oversized kinetic headline, sticky glass navigation, a calm three-step progression, and focused cards that hide complexity until it is needed. The current brand palette is intentionally precise: `#000814` near-black depth, `#001d3d` navy panels, `#003566` blue surfaces, `#ffc300` gold structure, and `#ffd60a` bright gold actions and risk emphasis.

## Quick start

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. The Vite frontend proxies API calls to the Express backend on port `3001`.

`npm run dev` is a local development mode only: both Vite and the API bind to loopback. Do not change the bind address or expose the development server to a public or shared network. Production starts with `npm start`, fails closed without production security settings, and presents a login screen.

Production-style run:

```bash
npm run build
PORT=3000 npm start
```

## Upload format

Upload almost any comma-separated bank, card, payments, or transaction file with one row per record and a binary target column. FraudShield accepts comma, semicolon, tab, and pipe-delimited exports, arbitrary feature headers, quoted values, numeric values, categorical values, missing cells, and common bank labels. It detects these label names automatically when present: `is_fraud`, `fraud`, `fraud_flag`, `label`, `target`, `class`, `outcome`, `default`, `chargeback`, `response`, and `churn`. You can also select the target column manually in the training step when the bank export uses an unfamiliar name.

Example:

```csv
amount,device_age_days,velocity_10m,merchant_category,location_new,is_fraud
42,930,1,groceries,0,0
220,120,3,electronics,1,1
```

Numeric columns are standardized and categorical columns are one-hot encoded (unseen categories fall into an "other" bucket, so every categorical column is always exactly one-hot). The workbench sorts labeled rows by the detected timestamp/date/time column, or by original row order when no time field exists, then uses a chronological 60% train / 20% validation / 20% test split. See **Evaluation protocol** below for exactly what each split is used for.

**Algorithms.** The benchmark set is Logistic Regression, Gaussian Naive Bayes, a depth-limited CART decision tree, a 17-tree random forest, and a gradient-boosted-stump ensemble. **All five are custom, dependency-free implementations** and the UI/API label them that way. In particular, `gradient_boosted_stumps` (formerly `xgboost`, still accepted as an alias) is a simplified second-order boosting of depth-1 trees; it is **not** the native XGBoost algorithm and its numbers must not be compared with XGBoost results. The Isolation Forest experiment is likewise a simplified custom version. For a production claim, replace these with validated libraries and re-run the same protocol.

**Controlled ablation.** Each run reports a 2×2 experiment — {original features, original + behavioural features} × {no resampling, SMOTE-NC} — using the same chronological split, the same logistic-regression learner, the same cost-based validation threshold procedure and the same untouched test set, so feature effects and resampling effects can be told apart (`experiments.ablation.contrasts`). A warning is attached when the test period has fewer than 30 fraud cases. Oversampling is a SMOTE-NC implementation: k-nearest-neighbour search inside the minority class (numeric distance plus the SMOTE-NC categorical penalty), interpolation for numeric columns, and majority vote among neighbours for categorical columns, so synthetic rows are always valid one-hot encodings (checked and reported as `invalid_encoded_rows`). It is a local implementation, not imbalanced-learn, and count-like numeric features can receive fractional values. SMOTE is applied to the training partition only, and the production candidates are trained without resampling. A class-weighted variant and a separate unsupervised Isolation Forest experiment are also reported; the anomaly score is not blended into the supervised probabilities.

**One past-only feature pipeline.** Behavioural and relationship features (transactions in the last 10 minutes, 7-day average spend, amount deviation, new-device flag, shared-device count, merchant/device degree, account–merchant repeats) are produced by `server/ml/features.js` for training, validation, test, replay and single predictions. Customer rolling history and device/merchant relationship facts are timestamp-filtered, so backdated scoring never counts future records. A newly trained model stores a cloned feature state at the end of the training period; validation/test transactions are never preloaded into deployment history. Each candidate owns an independent mutable copy. Replay uses the saved training state only when its first row is after the training cutoff; otherwise it starts empty to avoid duplicate training rows. Scoring is read-only by default; `"observe": true` appends the scored row to that model's own history.

**Unified hybrid decision service.** The workbench prediction endpoint and the legacy `/api/score`, `/api/what-if`, `/api/feedback`, scenario and stream routes use `server/decision.js`. Its score is `0.75 × classifier score + 0.25 × selected temporal/relationship signal` (amount deviation, new device, 10-minute activity, shared-device owners, repeated merchant activity). The hybrid threshold minimizes the configured decision cost on validation data, and the selected model's hybrid metrics are reported on the held-out test period without using the test set for selection. This is validation-threshold selection, not probability calibration; scores must not be interpreted as calibrated real-world fraud probabilities. If no model is trained, the model is disabled, or scoring fails, the shared service fails closed to `review_required` and never silently approves.

**Explanations.** Local explanations treat each input column (a numeric column or a whole one-hot block) as one player. Logistic regression gets exact linear SHAP in log-odds. All other models get permutation-sampling Shapley estimates against a training-data background sample (interventional, independent-features assumption) with a reported sampling standard error; this is an estimate, not TreeSHAP. Additivity of the values is a property of the estimator and is **not** presented as fidelity. Faithfulness is measured separately against the model itself — a deletion test (top-k columns versus random columns) and the correlation with single-column ablation — and training reports the same checks, plus stability across inputs and sampling seeds, on validation rows. Global importance is native where it exists (absolute standardised coefficient; Gini impurity decrease; boosting gain) and validation permutation importance otherwise; if no real signal exists the list is empty rather than filled with zeros. Contributions describe model association, not causation.

Relationship features (shared-device counts, merchant/device degree, account–merchant repeats) are reported as a separate experiment against the original-feature baseline. This is a small relationship-feature experiment, not a graph neural network, and requires reliable relationship columns before the comparison is interpreted.

Uploads are limited to 10 MB per request and 300,000 CSV rows. For larger production files, use a queued or columnar ingestion pipeline rather than an in-memory browser upload.

## Evaluation protocol

1. **Train (60%)** — fits every candidate model and the encoder.
2. **Validation (20%)** — the only data used for decisions: models are ranked by validation PR-AUC (tie-break: lower validation cost), and each model's operating threshold is the cost-minimising threshold on validation data. The winner is frozen at this point.
3. **Test (20%)** — scored **once, for the frozen winner**; this is the unbiased estimate (`evaluation_role: selected_final_test_evaluation`, `metrics_source: test`). Other models carry validation figures only (`metrics_source: validation`). Pass `"exploratoryTestComparison": true` to also get test figures for the others; they are labelled `exploratory_test_comparison_do_not_use_for_selection` and never affect selection. Experiment test numbers are descriptive and are never used for tuning.

## Cost-aware decisions

Supply an optional `costMatrix` to train/candidate requests (defaults shown):

```json
{ "false_negative_cost": 100, "false_positive_cost": 10, "review_cost": 1,
  "false_negative_use_amount": false, "amount_loss_fraction": 1 }
```

`review_cost` is paid for every flagged transaction (the review or step-up happens either way), `false_positive_cost` is the extra customer-friction cost of flagging a legitimate transaction, and `false_negative_cost` is the loss from approving a fraud. With `false_negative_use_amount` the transaction amount × `amount_loss_fraction` is added to each missed fraud (needs an amount column). The threshold is the validation-cost minimiser (grid 0.01–0.99 plus "flag nothing"). Validation and test metrics include `cost.total_cost`, the missed-fraud / false-positive / review components, cost per 1,000 transactions, and savings versus approving everything. The threshold table marks the selected threshold and shows cost at each point.

## Public benchmark evaluation

Run `npm run benchmark:openml` to download the publicly accessible [OpenML CreditCardFraudDetection dataset](https://www.openml.org/d/42397) ARFF, verify its published MD5 and expected 284,807 rows / 492 fraud labels, then evaluate the project logistic-regression candidate with the same chronological 60/20/20 pipeline. The script fixes `Class` as the label, `Time` as the order, uses validation-only threshold selection, skips optional exploratory ablations for this single-candidate run, and reports the held-out hybrid decision and classifier-only PR-AUC, fraud recall, false positives, false negatives, and illustrative costs. It saves `reports/openml-creditcard-evaluation.md` and `.json`; the raw data stay in ignored `data/benchmark/` and are not included in the source ZIP. It may take several minutes and requires about 157 MB of download space plus additional memory for in-memory processing. The original [Kaggle dataset data card](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) states the source/collaboration context and DBCL 1.0 database-contents license.

The dataset describes anonymized European card transactions from September 2013 over two days. Its fields are largely PCA-transformed and it has no customer, device, or merchant identifier columns; the run therefore cannot validate account-specific behavior or relationship features. Rows with no customer ID are not merged into a fabricated shared customer history. The fixed costs are illustrative rather than calibrated to a real institution. This provides a reproducible historical public-benchmark result, not evidence of current or cross-bank fraud accuracy. See the Kaggle data card for the stated database-content license before reusing or redistributing the data.


## ML API

- `GET /api/auth/session` reports the browser session state. `POST /api/auth/login` accepts `{ "accessToken": "..." }` once and issues the protected session cookie; `POST /api/auth/logout` revokes it. The frontend login form calls these routes, keeps the token only for the duration of the form submission, and uses same-origin cookies plus a CSRF marker on later API requests.
- `POST /api/ml/preview` with `{ "content": "...csv...", "labelColumn": "optional_column" }` returns row count, columns, detected label, detected time column, positive rate, and sample rows.
- `POST /api/ml/train` with `{ "content": "...csv...", "labelColumn": "optional", "timeColumn": "optional", "algorithms": ["logistic_regression", "gaussian_nb", "decision_tree", "random_forest", "gradient_boosted_stumps"], "costMatrix": { ... }, "exploratoryTestComparison": false }` trains the requested models, selects the winner on validation data, scores the test period once for it, and returns the cost-aware metrics, the 2×2 ablation and the explanation validation. `"xgboost"` is accepted as a deprecated alias of `gradient_boosted_stumps`.
- `GET /api/ml/model` returns whether models exist, the selected model, all trained model metadata, and the maximum row limit.
- `POST /api/ml/predict` with `{ "modelId": "optional_model_id", "observe": false, "row": { "timestamp": "...", "customer_id": "c1", "amount": 220, "merchant_category": "electronics" } }` rebuilds the derived features from that model's training-cutoff history and returns the hybrid risk score, validation-selected cost threshold, decision, derived feature values and local explanation. Scoring never changes which model is active.
- `POST /api/ml/audit` with `{ "modelId": "optional_model_id", "row": { ... }, "action": "confirmed_fraud" | "marked_legitimate" }` records the model version, risk score, operating threshold, explanation summary/method, causality limitation, and final investigator action.
- `POST /api/ml/replay` with `{ "content": "...csv...", "modelId": "optional_model_id", "labelColumn": "optional_column", "timeColumn": "optional_timestamp_column" }` feeds rows through the real prediction endpoint and returns measured throughput, median/p95/max latency, before/after detection quality, feature-shift drift alerts, and an explicit no-auto-replacement policy.
- `GET /api/ml/export?modelId=...` exports a version-3 artifact for **any** model type: predictor state, encoder, feature configuration, timestamp-aware training-cutoff history, explanation background sample, cost matrix and metrics. `POST /api/ml/import` imports it as the active model and reproduces predictions exactly. Older artifacts are rejected because they lack timestamp-aware relationships and hybrid threshold metadata.
- `POST /api/ml/retrain/candidate` trains a candidate without replacing the active model; `GET /api/ml/retrain/status` checks the staged candidate; `POST /api/ml/retrain/approve` is the explicit replacement step. There is no automatic retraining.

The legacy demo endpoints remain, but their decisions now use the same hybrid decision service rather than a separate heuristic scoring path. `/api/metrics` and `/api/segments` are explicitly synthetic demo summaries.

## What the workbench shows

The first step is upload. The second step previews the dataset, lets the user confirm the label column, choose algorithms, set the decision costs, and starts chronological comparison training. The third step shows side-by-side algorithm cards (validation figures, except the selected model which shows the single final test evaluation), cost, false-positive workload, latency, threshold outcomes with cost, real feature importance, the 2×2 ablation, and a prediction form built from the raw input columns. Clicking a comparison card changes the model used for scoring.

After scoring, the workbench shows the local explanation method and leading contributions, with buttons to record confirmed fraud or marked legitimate. In development, audit entries are in memory. Production appends JSONL audit entries to a protected file with a SHA-256 hash chain; the active model artifact is atomically persisted with restrictive file permissions. Back up and monitor these files using your organization's controls.

The analysis step also includes a manual replay-and-drift dashboard. It measures the selected model over the uploaded transaction sequence, compares labelled quality before and after the midpoint distribution change, reports throughput and latency, and raises a feature-shift alert when the relative numeric mean shift reaches 25%. A drift alert only requests validation; it cannot retrain or replace the active model.

## Reproducible literature-review methodology

Record the protocol before searching. Search at least two scholarly indexes plus official benchmark or regulator sources using combinations of `(fraud OR payment OR card) AND (detection OR classification) AND (temporal OR concept drift OR time split)`, with separate queries for `precision recall PR-AUC`, `Random Forest`, `XGBoost`, `calibration`, and `false positive workload`. Save exact query strings, database names, access dates, result counts, and exported citations in a versioned evidence folder.

Screen titles and abstracts with predefined criteria: include peer-reviewed or official technical reports evaluating transaction/payment fraud with dataset construction, temporal ordering, class prevalence, or reproducible metrics; exclude marketing pages, duplicates, non-fraud anomaly tasks without a separable fraud label, and studies reporting only random cross-validation without leakage discussion. Deduplicate by DOI/title, record inclusion/exclusion reasons, and extract dataset period, label definition, prevalence, split protocol, feature availability time, model, hyperparameters, PR-AUC/ROC-AUC, precision, recall, false positives, false negatives, latency, and compute cost. Do not pool scores across incompatible datasets; synthesize by dataset and protocol. Store the screening table, extraction form, exact queries, and analysis script with a commit hash.

The model card displayed in the app is intentionally honest: this is an educational and prototyping workflow, not a production fraud model. Development mode binds the API and Vite to loopback only. Production refuses to start unless distinct 32-character-or-longer investigator and administrator access tokens are configured and TLS termination by a trusted reverse proxy is explicitly enabled. The browser login screen exchanges the entered token for a short-lived (8-hour) server-side session in an `HttpOnly`, `SameSite=Strict`, `Secure` production cookie; the token is not stored in local storage or sent with subsequent API requests. Sign-out revokes the session, and session expiry returns the UI to login. Session-authenticated state-changing requests require a same-origin/allowlisted `Origin` and a CSRF marker. Administrator sessions may upload/preview data, train, replay, import/export models and approve retraining; investigator sessions may score and record/read audit outcomes but cannot use those administrator routes. Trusted service clients may still use role-scoped bearer tokens. Configure `FRAUDSHIELD_ALLOWED_ORIGINS` narrowly if frontend and API origins differ; serve the browser and API same-origin or at least same-site because cookies use `SameSite=Strict`. JSON requests are capped at 10 MB, uploads are bounded by the CSV row limit, and common security headers and per-IP/login-attempt limits are applied. The API binds to loopback by default; a container may bind internally to all interfaces only on a private network behind a trusted TLS reverse proxy. Never publish the container port directly. Production model artifacts and audit logs are durable local files but still require encrypted disks, backups, access monitoring and external audit retention. Use only synthetic/public data for the hackathon; do not use this prototype for real payment decisions without independent security review, identity-provider integration, jurisdictional/privacy review, robust monitoring and revalidation on representative data.

Production setup example (use a secrets manager rather than committing values):

```bash
export NODE_ENV=production
export FRAUDSHIELD_TLS_TERMINATED=1
export FRAUDSHIELD_ADMIN_TOKEN="$(openssl rand -hex 32)"
export FRAUDSHIELD_INVESTIGATOR_TOKEN="$(openssl rand -hex 32)"
export FRAUDSHIELD_ALLOWED_ORIGINS="https://fraud.example.org"
export FRAUDSHIELD_DATA_DIR="/var/lib/fraudshield"
npm start
```

Place a TLS-enabled reverse proxy on the same host or private container network, preserve the external `Host` header (or configure the exact trusted web origin), and prevent direct public access to the API port. Users sign in on the FraudShield page using the relevant configured role token; the server exchanges it for a session cookie. Never put access tokens in browser bundles, URLs, logs or source control. The investigator token permits normal scoring/audit access; administrator workflows require signing in with the distinct administrator token. Sessions are in memory and become invalid on server restart, requiring sign-in again.

## Verification status and claims

The source suite and frontend production build were run in this Sandbox after `npm ci`; see `VALIDATION-STATUS.md` for the exact scope. That check ran on `x86_64`/Node `x64`, not IBM Z `s390x`, and Docker was unavailable, so neither container deployment nor IBM LinuxONE operation is verified. The project has not been evaluated on an independent real-world fraud benchmark or representative bank data. `/api/metrics` now returns no metrics until a model has been trained on an uploaded dataset; afterward it reports only that model's dataset-specific evaluation. Segment-level metrics are reported as not evaluated. These results are not evidence of general real-world accuracy. Do not claim real-world detection performance or LinuxONE deployment until the corresponding benchmark and target-hardware tests have been completed and documented.

## Tests

```bash
npm test
npm run check
```

The test suite covers deterministic legacy helper behavior, past-only temporal and relationship features, candidate state isolation, training-cutoff snapshots, the shared hybrid service and its held-out metrics, protected artifact versioning, replay/retraining behavior, and production browser authentication over HTTP. The end-to-end auth test runs with `npm test` and expects Express to be installed by `npm ci`.

## Project structure

- `server/ml.js` — training orchestration (validation-only selection, single test evaluation, ablation), prediction, replay/drift, artifact export/import. Supporting modules in `server/ml/`: `features.js` (shared stateful feature pipeline), `encoder.js`, `metrics.js` (PR/ROC-AUC, confusion, cost matrix and cost-based thresholds), `models.js` (custom classifiers and importances), `smote.js` (SMOTE-NC), `explain.js` (linear SHAP, sampled Shapley, faithfulness, global importance), `util.js`.
- `server/ml.js` — CSV parsing, chronological training, five benchmark algorithms, imbalance experiments, past-only behavioural/relationship features, graph-feature comparison, local model explanations, and prediction.
- `server/index.js` — Express API, upload/train/predict routes, legacy fraud routes, and production static serving.
- `server/index.js` — Express API, role enforcement, browser session login/logout, CSRF validation, bounded uploads, replay/drift monitoring, protected artifact routes, and approval-gated retraining.
- `server/decision.js` — shared validation-thresholded hybrid decision service used by both workbench and legacy monitoring routes.
- `server/security.js`, `server/model-persistence.js`, `server/store.js` — production role checks, atomic active-model persistence, and durable hash-chained audit storage.
- `src/App.jsx` — simple three-step ML workbench UI.
- `src/styles.css` — glow-morphism visual system inspired by the supplied video.
- `shared/contracts.js` — deterministic legacy demo contracts.
- `tests/smoke.test.js` — backend and ML smoke tests.
- `tests/auth.integration.test.js` — production-configured HTTP session, CSRF and RBAC test.
- `scripts/evaluate-openml-creditcard.mjs` — reproducible public benchmark runner; outputs are saved in `reports/`.
- `docs/DEPLOYMENT-READINESS.md` — evidence gates for IBM Z/LinuxONE and production-security review.

## Limitations

Uploaded datasets are processed locally and in memory; this prototype does not retain the raw training CSV or send it to an external model provider. In development, active models and audits are in memory; in production, the active artifact and append-only audit trail are persisted locally as described above. Browser sessions are held in the API process and expire on restart; run one API instance unless a shared, durable session store is added. Replay drift alerts remain observational and use a simple numeric mean-shift rule; production fraud monitoring should add robust categorical/continuous drift tests, alert calibration, window governance, and labelled delayed-outcome handling. Production now has cookie-session and bearer-token roles and requires trusted TLS termination, but it is still a controlled prototype—not a public payment-decision product or a substitute for enterprise identity governance. Random Forest and XGBoost-style are dependency-free benchmark implementations; real deployment should replace them with vetted libraries, pin versions, and validate numerical parity. Feature contributions are not causal evidence. Federated learning is intentionally deferred as a future extension only if cross-bank collaboration becomes part of the research objective. A graph neural network is also deferred: first validate imbalance handling, explanations, simple graph-derived features, replay quality, and approval governance with reliable relationship data; only then consider a GNN.
