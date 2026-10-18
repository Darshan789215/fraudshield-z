import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer } from 'node:net';
import { once } from 'node:events';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const pause = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
async function freePort() {
  const server = createServer();
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const { port } = server.address();
  await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  return port;
}

 test('production browser login issues a role session and enforces CSRF, RBAC, and logout', { timeout: 20_000 }, async () => {
  const port = await freePort();
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fraudshield-auth-'));
  const adminToken = 'admin-integration-test-token-000000000000000000000000000000';
  const investigatorToken = 'investigator-integration-test-token-00000000000000000000';
  const child = spawn(process.execPath, ['server/index.js'], {
    cwd: process.cwd(),
    env: { ...process.env, NODE_ENV: 'production', PORT: String(port), FRAUDSHIELD_BIND_HOST: '127.0.0.1', FRAUDSHIELD_TLS_TERMINATED: '1', FRAUDSHIELD_ADMIN_TOKEN: adminToken, FRAUDSHIELD_INVESTIGATOR_TOKEN: investigatorToken, FRAUDSHIELD_DATA_DIR: dataDir },
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  const base = `http://127.0.0.1:${port}`;
  const origin = `https://127.0.0.1:${port}`;
  let stderr = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => { stderr += chunk; });
  try {
    let healthy = false;
    for (let attempt = 0; attempt < 60; attempt += 1) {
      if (child.exitCode !== null) throw new Error(`Server exited before readiness: ${stderr}`);
      try { const response = await fetch(`${base}/api/health`); if (response.ok) { healthy = true; break; } } catch { /* server is still starting */ }
      await pause(100);
    }
    assert.equal(healthy, true, `Server did not become ready: ${stderr}`);
    assert.equal((await fetch(`${base}/api/audit`)).status, 401, 'protected API rejects anonymous access');

    const login = async (accessToken) => fetch(`${base}/api/auth/login`, {
      method: 'POST', headers: { Origin: origin, 'X-Requested-With': 'XMLHttpRequest', 'Content-Type': 'application/json' },
      body: JSON.stringify({ accessToken }),
    });
    assert.equal((await login('incorrect')).status, 401, 'invalid credentials are rejected');
    const investorLogin = await login(investigatorToken);
    assert.equal(investorLogin.status, 200);
    const investorCookie = investorLogin.headers.get('set-cookie').split(';')[0];
    assert.match(investorLogin.headers.get('set-cookie'), /HttpOnly/);
    assert.match(investorLogin.headers.get('set-cookie'), /SameSite=Strict/);
    assert.match(investorLogin.headers.get('set-cookie'), /Secure/);
    const status = await fetch(`${base}/api/auth/session`, { headers: { Cookie: investorCookie } });
    assert.deepEqual(await status.json(), { ok: true, authenticated: true, role: 'investigator', mode: 'production-session' });
    const metricsResponse = await fetch(`${base}/api/metrics`, { headers: { Cookie: investorCookie } });
    const metrics = await metricsResponse.json();
    assert.equal(metrics.evaluation_status, 'not_evaluated');
    assert.equal(metrics.metrics, null);
    const segments = await (await fetch(`${base}/api/segments`, { headers: { Cookie: investorCookie } })).json();
    assert.equal(segments.segments, null);

    const adminAction = await fetch(`${base}/api/ml/train`, { method: 'POST', headers: { Cookie: investorCookie, Origin: origin, 'X-Requested-With': 'XMLHttpRequest', 'Content-Type': 'application/json' }, body: '{}' });
    assert.equal(adminAction.status, 403, 'investigator cannot train models');
    const csrfFailure = await fetch(`${base}/api/feedback`, { method: 'POST', headers: { Cookie: investorCookie, Origin: 'https://untrusted.example', 'X-Requested-With': 'XMLHttpRequest', 'Content-Type': 'application/json' }, body: '{}' });
    assert.equal(csrfFailure.status, 403, 'untrusted browser origin is rejected');
    const score = await fetch(`${base}/api/score`, { method: 'POST', headers: { Cookie: investorCookie, Origin: origin, 'X-Requested-With': 'XMLHttpRequest', 'Content-Type': 'application/json' }, body: '{}' });
    assert.equal(score.status, 200, 'investigator can reach scoring API');

    const adminLogin = await login(adminToken);
    assert.equal(adminLogin.status, 200);
    const adminCookie = adminLogin.headers.get('set-cookie').split(';')[0];
    const adminActionAllowed = await fetch(`${base}/api/ml/train`, { method: 'POST', headers: { Cookie: adminCookie, Origin: origin, 'X-Requested-With': 'XMLHttpRequest', 'Content-Type': 'application/json' }, body: '{}' });
    assert.equal(adminActionAllowed.status, 400, 'administrator passes RBAC and receives dataset validation error');

    const logout = await fetch(`${base}/api/auth/logout`, { method: 'POST', headers: { Cookie: investorCookie, Origin: origin, 'X-Requested-With': 'XMLHttpRequest', 'Content-Type': 'application/json' }, body: '{}' });
    assert.equal(logout.status, 200);
    const afterLogout = await fetch(`${base}/api/auth/session`, { headers: { Cookie: investorCookie } });
    assert.equal((await afterLogout.json()).authenticated, false);
  } finally {
    child.kill('SIGTERM');
    if (child.exitCode === null) await Promise.race([once(child, 'exit'), pause(2000)]);
    fs.rmSync(dataDir, { recursive: true, force: true });
  }
});
