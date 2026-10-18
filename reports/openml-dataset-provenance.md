# OpenML credit-card fraud dataset provenance

The benchmark runner uses OpenML dataset 42397, `CreditCardFraudDetection`, downloadable as ARFF. OpenML describes 284,807 credit-card transactions by European cardholders from September 2013 over two days, with 492 fraud labels (0.172%), and says the 28 V-features are PCA-transformed while `Time` is elapsed seconds and `Amount` is retained. OpenML's description identifies a Worldline/ULB Machine Learning Group research collaboration but also marks the immediate source and date as unknown. The metadata page reports format ARFF and MD5 `f9544642717dc57d30c6505a422da240` [1] [2].

The OpenML ARFF file's target declaration is `@ATTRIBUTE Class {True, False}` and rows use `True`/`False`; the page description calls the target 1/0. The benchmark script accepts these boolean strings and normalizes them through FraudShield's existing binary-label parser [2]. The Kaggle data card lists the license as Open Database, Contents: Database Contents License 1.0. Read and follow that source license before reusing or redistributing the data [3]. The project archive excludes the downloaded data file.

This is one old, short-horizon, anonymized benchmark—not a current or representative portfolio. It contains no customer, device, or merchant identifier columns, so it cannot validate those relationship features, cannot support customer-specific behavior conclusions, and cannot prove general production accuracy.

## Sources

[1]: https://www.openml.org/d/42397 "OpenML CreditCardFraudDetection dataset"
[2]: https://www.openml.org/api/v1/json/data/42397 "OpenML dataset metadata API"
[3]: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "Kaggle ULB Credit Card Fraud dataset data card"
