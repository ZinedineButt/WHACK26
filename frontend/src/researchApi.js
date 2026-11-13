// Tiny client for the MarketMind research API. The backend is the source of truth:
// no statistics or verdicts are computed in the browser.
const BASE = (import.meta.env && import.meta.env.VITE_API_BASE) || 'http://localhost:5000'

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, options)
  let data = null
  try { data = await res.json() } catch { /* non-JSON error body */ }
  if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status})`)
  return data
}

export const getCatalog = () => request('/api/research/catalog')

/** body: {mode:'autonomous'|'guided'|'demo_autonomous', ...} or {preset_id} (legacy synthetic demo). */
export const startRun = body =>
  request('/api/research/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

export const getRun = runId => request(`/api/research/runs/${encodeURIComponent(runId)}`)

const TERMINAL = ['completed', 'failed']

/** Polls until the run is completed/failed. Calls onUpdate(report) each tick. */
export async function pollRun(runId, onUpdate, { intervalMs = 1000, signal } = {}) {
  for (;;) {
    if (signal && signal.aborted) throw new DOMException('aborted', 'AbortError')
    const report = await getRun(runId)
    onUpdate(report)
    if (TERMINAL.includes(report.status)) return report
    await new Promise(r => setTimeout(r, intervalMs))
  }
}

/** Paper-trade the frozen strategy of one hypothesis against the S&P 500. Places no orders. */
export const simulateStrategy = (runId, body) =>
  request(`/api/research/runs/${encodeURIComponent(runId)}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

/** POST (or GET when no body) to any research endpoint and return JSON; throws Error(message) on failure. */
export const simulateLikeRequest = (path, body) =>
  request(path, body === undefined ? undefined : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
