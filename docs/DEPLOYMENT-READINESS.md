# Deployment readiness evidence

**Current status: not verified.** This checklist is a release gate, not a certification. A local sandbox test, passing build, or successful role-login test does not establish LinuxONE compatibility or production security.

## IBM Z / LinuxONE deployment record

Do not mark this complete until the application is built and exercised on the intended IBM Z/LinuxONE environment. Record the date, source commit, operator, host/LPAR or approved runtime, OS release, `uname -m`, Node/npm versions, and whether the deployment uses a native build or a container image. The target architecture should be captured from the machine itself; do not infer it from a container tag or CI result.

On the target, run `npm ci` and `npm run check` and save the full logs and exit codes. Then start the production-configured service behind the intended private network and real TLS reverse proxy. Capture a browser login/logout flow, anonymous API rejection, investigator-vs-administrator authorization, the health response, a scoring request, model persistence across restart, and audit-log integrity. Record observed CPU/memory and latency only as measurements from that exact environment. Attach configuration with secrets, access tokens, certificates, customer data, and private hostnames removed.

Evidence fields to complete:

- Date / source commit / operator:
- IBM Z or LinuxONE model and runtime type:
- OS release / `uname -m` / Node and npm versions:
- Install, test, build, and startup exit codes and log locations:
- Reverse-proxy and TLS verification results:
- Browser login and role-authorization results:
- API health/scoring results:
- Persistence/restart and audit-integrity results:
- Resource/latency observations and known failures:
- Reviewer and sign-off:

## Production security gate

Before any public exposure, replace shared role tokens with the organization's approved identity provider and per-user identities, establish a managed secret issuance/rotation process, and verify the real TLS termination and proxy trust boundary. Restrict the API to the private proxy network; do not publish a development server or container port directly. Test origin/CSRF handling and role boundaries against the deployed proxy, check session behavior for restarts and multiple replicas, and move sessions to an appropriately protected shared store if scaling beyond one process.

Have security staff review the threat model, authentication and authorization, rate limits, file parsing and upload limits, artifact import/export, audit integrity and retention, backups, dependencies, logging redaction, privacy/legal requirements, and incident response. Run an independent penetration test against the actual deployed configuration, fix findings, and record the retest. Store evidence of certificate management, secret rotation, backup restoration, monitoring, and access reviews. Do not describe this checklist or its local automated tests as a security audit or certification.

## Fraud benchmark gate

Keep each benchmark report with its dataset citation, license, checksum, label definition, feature availability assumptions, chronological split boundaries, selection and threshold rules, class counts, PR-AUC, fraud recall, false positives, false negatives, cost matrix, and reproducibility details. Public historical results are benchmark evidence only; representative authorized data and operational validation are still required for a deployment claim.
