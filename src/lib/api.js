const csrfHeaders = { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' };

export async function getJson(path, options = {}) {
  const response = await fetch(path, {
    credentials: 'same-origin',
    ...options,
    headers: { ...csrfHeaders, ...(options.headers || {}) },
  });
  const body = response.status === 204 ? {} : await response.json();
  if (response.status === 401 && !['/api/auth/login', '/api/auth/session'].includes(path) && typeof window !== 'undefined') {
    window.dispatchEvent(new Event('fraudshield:unauthorized'));
  }
  if (!response.ok) throw new Error(body.error || body.message || `Request failed: ${response.status}`);
  return body;
}

export function login(accessToken) {
  return getJson('/api/auth/login', { method: 'POST', body: JSON.stringify({ accessToken }) });
}

export function logout() {
  return getJson('/api/auth/logout', { method: 'POST', body: '{}' });
}

export const formatPercent = (value) => value === null || value === undefined ? '—' : `${Math.round(value * 100)}%`;
export const formatMoney = (value, currency = 'USD') => new Intl.NumberFormat('en-US', { style: 'currency', currency, maximumFractionDigits: 0 }).format(value);
export const labelFor = (decision) => ({ approve: 'Approve', step_up: 'Step-up', hold: 'Hold', review_required: 'Review required' }[decision] || decision);
export const toneFor = (decision) => ({ approve: 'positive', step_up: 'warning', hold: 'danger', review_required: 'danger' }[decision] || 'neutral');
