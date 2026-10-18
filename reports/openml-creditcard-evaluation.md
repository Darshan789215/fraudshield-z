# OpenML Credit Card Fraud Evaluation

Evaluated 2026-10-17. This is a reproducible historical benchmark result, not a production-accuracy claim.

## Dataset and scope

OpenML dataset 42397 describes 284,807 anonymized European card transactions from September 2013 over two days, with 492 frauds. The data card identifies `Time` as elapsed seconds, `Amount` as the transaction amount, and V1–V28 as PCA-transformed features [1] [2]. The OpenML ARFF target is encoded `{True, False}` despite the description's 1/0 wording; the runner normalizes these values and verifies the published MD5 before training.

Dataset MD5: `f9544642717dc57d30c6505a422da240`. SHA-256: `9cda3beed911b711edb234f3ff9fb26d782796cce5add725e8f43f94a067eee7`. The Kaggle data card lists Open Database, Contents Database Contents License 1.0; review the source license before reuse or redistribution [2]. The raw data are excluded from the project archive.

This dataset has no customer, device, or merchant identifier columns. It therefore cannot validate account-specific behavior or relationship features. Rows without a customer ID are not grouped under a fabricated shared customer. Its age, two-day horizon, anonymized features and missing entity identifiers limit operational relevance.

## Protocol

The project pipeline sorts by `Time` and uses a chronological 60/20/20 split: 170,884 training rows, 56,961 validation rows, and 56,962 test rows. Logistic regression was the only candidate requested. Selection and threshold choice used validation data; the selected model was evaluated on the test period once. The base classifier threshold was 0.08 and the hybrid threshold was 0.06. Separate ablation, anomaly and graph experiments were skipped because this run evaluates a single candidate.

The configured cost matrix is illustrative, not calibrated to a bank or currency: missed fraud costs 100 units, a false positive 10 units, and a review/step-up action 1 unit. Transaction amount was not added to missed-fraud cost.

## Held-out test results

The hybrid decision was evaluated on 56,962 chronological test rows, including 75 frauds. Its step-wise PR-AUC (average precision) was **0.796**; fraud recall was **0.760**; precision was **0.851**; false positives were **10**; and false negatives were **18**. The confusion counts were 57 true positives, 56,877 true negatives, 10 false positives, and 18 false negatives. Under the illustrative cost matrix, total cost was **1,967 units**, or **34.53 units per 1,000 transactions**. Approving every transaction would have cost 7,500 units under the same matrix.

The classifier-only comparator had the same PR-AUC (0.796), recall (0.760), 10 false positives, 18 false negatives, and total cost (1,967 units). It matches the hybrid result here because the dataset does not contain customer/device/merchant identifiers from which the selected relationship signals could be computed. The results are point estimates; no confidence interval was calculated, and only 75 fraud cases appear in the test period.

## Interpretation

These metrics describe one historical, anonymized public dataset under this specific split, feature set and illustrative cost matrix. They do not establish current, cross-bank or real-world deployment performance. They do not validate entity-level relationship signals. No IBM Z/LinuxONE deployment or independent security assessment is implied; the run was performed in a Linux x64 sandbox.

## References

[1]: https://www.openml.org/d/42397 "OpenML CreditCardFraudDetection dataset"

[2]: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "Kaggle ULB Credit Card Fraud dataset data card"
