# Validation status

Updated: 2026-10-18

## Verified in this Sandbox

A fresh `npm ci` completed from the checked-in lockfile, installing 276 packages and reporting zero known vulnerabilities at install time. `npm run check` then passed all 30 tests and completed the Vite production build. The benchmark runner also passed `node --check`. Environment: x86_64, Node v22.13.0, npm 10.9.2. The complete command output is saved in [`reports/verification-run.log`](reports/verification-run.log); npm did emit deprecation warnings for two dependencies, despite the zero-vulnerability audit result.

The test suite includes the end-to-end production-configured HTTP login/session/CSRF/RBAC/logout flow, independent model-history state, training-cutoff snapshots, future-record exclusion, and a regression test showing missing customer identifiers do not collapse unrelated transactions into shared customer/device-owner history.

The reproducible OpenML benchmark completed on the full 284,807-row dataset after checksum verification. With a chronological 60/20/20 split and validation-selected thresholds, the held-out 56,962-row test period contained 75 fraud cases. The hybrid decision had PR-AUC 0.796, fraud recall 0.760, 10 false positives, 18 false negatives, and 1,967 illustrative cost units. These are point estimates under a cost matrix not calibrated to an institution. See [`reports/openml-creditcard-evaluation.md`](reports/openml-creditcard-evaluation.md) for the protocol, thresholds, costs, dataset hashes, citation and limitations, and [`reports/openml-dataset-provenance.md`](reports/openml-dataset-provenance.md) for source and license notes. The benchmark is historical and the dataset lacks customer/device/merchant identifiers, so it does not validate entity-level relationship features or establish current cross-bank accuracy.

## Not verified

The project has not been built or run on IBM Z/LinuxONE or `s390x`; this validation ran on x86_64 only. No production deployment, real TLS reverse proxy, external identity provider, production secrets manager, multi-instance session store, or independent security assessment/penetration test was verified. Local authentication tests are not a security certification. Complete the evidence gates in [`docs/DEPLOYMENT-READINESS.md`](docs/DEPLOYMENT-READINESS.md) before public deployment.
