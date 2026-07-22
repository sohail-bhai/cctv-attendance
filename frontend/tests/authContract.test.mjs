import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

const values = new Map();
globalThis.localStorage = {
  getItem: (key) => values.get(key) ?? null,
  setItem: (key, value) => values.set(key, String(value)),
  removeItem: (key) => values.delete(key),
};
const events = [];
globalThis.window = { dispatchEvent: (event) => events.push(event.type) };

const client = await import('../src/api/client.js');

test('X-User-Id is never sent without a bearer session', () => {
  assert.deepEqual(client.authHeadersForUser({ id: 'synthetic-hod' }), {});
  assert.deepEqual(client.authHeadersForUser(null, { Accept: 'application/json' }), { Accept: 'application/json' });
  assert.deepEqual(client.authHeadersForUser({ id: 'synthetic-hod', sessionToken: 'synthetic-token' }), {
    Authorization: 'Bearer synthetic-token',
    'X-User-Id': 'synthetic-hod',
  });
});

test('only protected 401 invalidates the stored session', () => {
  assert.equal(client.shouldInvalidateStoredSession('/api/reports', 401), true);
  assert.equal(client.shouldInvalidateStoredSession('/api/reports', 403), false);
  assert.equal(client.shouldInvalidateStoredSession('/api/auth/login', 401), false);
});

test('401 clears stale auth while 403 preserves it', async () => {
  const key = 'sreenidhi_attendance_user';
  values.set(key, JSON.stringify({ id: 'synthetic-hod', sessionToken: 'synthetic-token' }));
  globalThis.fetch = async () => new Response(JSON.stringify({ success: false, error: 'Access denied.' }), { status: 403, headers: { 'content-type': 'application/json' } });
  const forbidden = await client.apiGetResult('/api/reports');
  assert.equal(forbidden.status, 403);
  assert.ok(values.has(key));

  globalThis.fetch = async () => new Response(JSON.stringify({ success: false, error: 'Authentication required.' }), { status: 401, headers: { 'content-type': 'application/json' } });
  const unauthorized = await client.apiGetResult('/api/reports');
  assert.equal(unauthorized.status, 401);
  assert.equal(values.has(key), false);
  assert.ok(events.includes(client.AUTH_INVALID_EVENT));
});

test('frontend source has no identity query fallback or direct protected download navigation', () => {
  const liveDemo = fs.readFileSync(new URL('../src/pages/LiveDemo.jsx', import.meta.url), 'utf8');
  const reports = fs.readFileSync(new URL('../src/pages/Reports.jsx', import.meta.url), 'utf8');
  const review = fs.readFileSync(new URL('../src/pages/ManualReview.jsx', import.meta.url), 'utf8');
  assert.equal(liveDemo.includes('user_id='), false);
  assert.ok(liveDemo.includes("apiGetBlobResult('/api/live-demo/frame')"));
  assert.equal(reports.includes('window.open(session.download_url'), false);
  assert.equal(review.includes('window.open(result.data.download_url'), false);
});
