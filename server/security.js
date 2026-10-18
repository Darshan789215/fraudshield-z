import { randomBytes, timingSafeEqual } from 'node:crypto';

const SESSION_TTL_MS = 8 * 60 * 60 * 1000;
const sessions = new Map();
const sameSecret = (provided, expected) => {
  if (typeof provided !== 'string' || typeof expected !== 'string' || !provided || !expected) return false;
  const a = Buffer.from(provided); const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
};

export function validateProductionSecurity(env = process.env) {
  if (env.NODE_ENV !== 'production') return;
  const admin = env.FRAUDSHIELD_ADMIN_TOKEN || '';
  const investigator = env.FRAUDSHIELD_INVESTIGATOR_TOKEN || '';
  if (admin.length < 32 || investigator.length < 32 || sameSecret(admin, investigator)) {
    throw new Error('Production requires distinct FRAUDSHIELD_ADMIN_TOKEN and FRAUDSHIELD_INVESTIGATOR_TOKEN values (each at least 32 characters).');
  }
  if (env.FRAUDSHIELD_TLS_TERMINATED !== '1') {
    throw new Error('Production requires TLS termination by a trusted reverse proxy (set FRAUDSHIELD_TLS_TERMINATED=1 only behind that proxy).');
  }
}

export function validateBindHost(host, env = process.env) {
  if (env.NODE_ENV !== 'production' && !['127.0.0.1', '::1', 'localhost'].includes(host)) {
    throw new Error('Development mode is local-only and may bind only to loopback. Use protected production mode behind a trusted reverse proxy for network access.');
  }
  return host;
}

export function authRole(req) {
  const authorization = req.get('authorization') || '';
  const match = /^Bearer\s+(.+)$/i.exec(authorization);
  return match ? roleForToken(match[1]) : null;
}

export function roleForToken(token) {
  if (sameSecret(token, process.env.FRAUDSHIELD_ADMIN_TOKEN)) return 'admin';
  if (sameSecret(token, process.env.FRAUDSHIELD_INVESTIGATOR_TOKEN)) return 'investigator';
  return null;
}

function sessionIdFromRequest(req) {
  const cookieHeader = req.get('cookie') || '';
  for (const item of cookieHeader.split(';')) {
    const separator = item.indexOf('=');
    if (separator < 0) continue;
    if (item.slice(0, separator).trim() === 'fraudshield_session') {
      try { return decodeURIComponent(item.slice(separator + 1).trim()); } catch { return null; }
    }
  }
  return null;
}

export function createSession(role, now = Date.now()) {
  if (!['admin', 'investigator'].includes(role)) throw new Error('Unsupported session role.');
  if (sessions.size > 20_000) for (const [id, session] of sessions) if (session.expiresAt <= now) sessions.delete(id);
  if (sessions.size >= 25_000) throw new Error('Session capacity reached; try again later.');
  const id = randomBytes(32).toString('base64url');
  sessions.set(id, { role, expiresAt: now + SESSION_TTL_MS });
  return { id, role, expires_at: new Date(now + SESSION_TTL_MS).toISOString() };
}

export function sessionRole(req, now = Date.now()) {
  const id = sessionIdFromRequest(req);
  if (!id) return null;
  const session = sessions.get(id);
  if (!session) return null;
  if (session.expiresAt <= now) { sessions.delete(id); return null; }
  return session.role;
}

export function revokeSession(req) {
  const id = sessionIdFromRequest(req);
  if (id) sessions.delete(id);
}

export function sessionCookie(id, production = process.env.NODE_ENV === 'production') {
  return `fraudshield_session=${encodeURIComponent(id)}; Path=/; HttpOnly; SameSite=Strict; Max-Age=${Math.floor(SESSION_TTL_MS / 1000)}${production ? '; Secure' : ''}`;
}
export function expiredSessionCookie(production = process.env.NODE_ENV === 'production') {
  return `fraudshield_session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0${production ? '; Secure' : ''}`;
}

const adminRoutes = [
  ['POST', /^\/api\/ml\/(preview|train|import|replay|retrain\/candidate|retrain\/approve)$/],
  ['GET', /^\/api\/ml\/export$/],
];
export function requiredRole(method, pathname) {
  return adminRoutes.some(([verb, pattern]) => verb === method && pattern.test(pathname)) ? 'admin' : 'investigator';
}

export const SESSION_TTL_SECONDS = SESSION_TTL_MS / 1000;
