# FraudShield Z — implementation and verification status

## Implemented in this project

- Candidate models own independent feature histories. Trained models retain the feature-state snapshot at the training cutoff, and chronological replay starts from that state and advances through later records. Timestamped device/merchant relationship features exclude future observations.
- A shared hybrid decision service combines the selected classifier score with selected temporal/relationship risk signals and validation-selected thresholds. Failure or missing-model behavior fails closed to `review_required`; demo scoring routes no longer use an independent heuristic decision path.
- Production mode has administrator/investigator authentication, role checks, same-origin cookie sessions, CSRF protections, bounded uploads, durable model/audit storage hooks and protected import/export. Development mode is loopback-only. Production still requires a real trusted TLS proxy and approved identity/secrets controls before public exposure.
- The project includes a reproducible OpenML public-benchmark runner and must describe its results as historical benchmark evidence only. The source data are not bundled. The two-day 2013 dataset has no customer/device/merchant IDs, so it cannot validate those entity-level relationship signals.
- Synthetic or public benchmark results, passing tests, and demo IBM Z architecture views are not evidence of current real-world performance or a successful IBM Z/LinuxONE deployment.

## External evidence still required

- Evaluate on suitable representative data under an authorized data-use arrangement; predeclare costs and thresholds, preserve chronological holdout, and report PR-AUC, fraud recall, false positives, false negatives and cost with class counts and confidence/uncertainty context.
- Build and run on the intended IBM Z/LinuxONE environment and complete the evidence fields in [`docs/DEPLOYMENT-READINESS.md`](docs/DEPLOYMENT-READINESS.md).
- Verify the actual TLS reverse proxy, approved identity provider, secrets management/rotation, production network isolation, deployment persistence, monitoring and backup/restore. Obtain an independent security review and penetration test before public deployment.

## Verification

- `npm ci` completed and `npm run check` passed on x86_64; the saved run record is `reports/verification-run.log`.
- The suite includes feature-history isolation, future-record exclusion, shared decision-service integration, and a real HTTP production authentication/session/CSRF/RBAC/logout test.
- LinuxONE execution and independent production security assessment remain unverified regardless of local test results.
