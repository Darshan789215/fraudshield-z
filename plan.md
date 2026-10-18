# FraudShield Z — ML Workbench Plan

## Product direction

FraudShield Z is a simple three-step ML workbench for fraud datasets: **Upload → Train → Analyze**, paired with a live simulation companion. The user can upload a labelled CSV, preview the data, choose the target column, train a real model, inspect held-out evaluation metrics and learned signals, score a new transaction row, and then explore the original deterministic scenarios alongside an SSE-powered live event rail.

The visual language follows the supplied reference video: dark charcoal canvas, muted sage actions, oversized kinetic typography, sticky glass navigation, generous whitespace, glow-morphism cards, and a horizontal three-step progression. The product avoids a dense operations-console layout until the user has a trained model, then adds a calm simulation area rather than switching to a visually noisy dashboard.

## ML architecture

- `server/ml.js` parses CSV rows in memory, detects a binary label, encodes numeric and categorical features, performs a deterministic 80/20 split, trains logistic regression with gradient descent, calculates accuracy/precision/recall/F1/ROC-AUC/confusion matrix, stores the trained model in memory, and scores new rows.
- `server/index.js` exposes preview, train, model-status, and predict endpoints.
- The legacy deterministic scoring API remains available for the original demo workflow.
- The frontend reads local file content and sends it over JSON; no raw dataset is persisted.
- Model artifacts are intentionally in-memory for the prototype and disappear on process restart.

## User journey

1. **Upload data:** drag/drop or browse for CSV; the user may use a small sample dataset.
2. **Configure & train:** inspect sample rows, select the label column, and start training.
3. **Analyze predictions:** review metrics, top feature weights, train/test split, enter a new row, and run prediction.

## Design direction

Chosen movement: minimal glow-morphism ML studio. The ownable color is neon cyber-green for action and confidence. Red is reserved for fraud predictions. Large high-contrast headings create the same kinetic editorial feel as the reference video. Cards use quiet borders, soft green glows, and compact monospace metadata.

Animation is restrained: hover lift, progress fill, and short state transitions. The interface should feel easy before it feels powerful.

## Guardrails

- Accept CSV up to 12MB and 50,000 rows.
- Require at least 12 labelled rows and both classes.
- Never imply that synthetic demo metrics or tiny sample datasets establish production performance.
- Clearly label the in-memory artifact and production limitations.
- Keep public-dataset validation optional; the upload workflow is the primary feature.
