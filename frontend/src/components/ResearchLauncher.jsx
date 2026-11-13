import React, { useEffect, useRef, useState } from 'react'
import { getCatalog, startRun, pollRun } from '../researchApi'
import ResearchReport from './research/Report'
import PairsTester from './research/PairsTester'
import CandleBackground, { ResultStamp } from './research/CandleBackground'

const LEGACY = [
  ['demo_null', 'Legacy preset: independent returns'],
  ['demo_confounder', 'Legacy preset: shared market factor'],
  ['demo_effect', 'Legacy preset: lagged effect'],
]
// One-click LIVE examples (real FRED / Tiingo data, Gemini-generated tests). Fill the form and launch.
const EXAMPLES = [
  { label: 'Pairs: do KO & PEP come back together?', pair: { a: 'KO', b: 'PEP' } },
  { label: 'AI explores: what moves markets?', mode: 'autonomous', lookback: 3, budget: 3, theme: 'What drives moves in the stock market?' },
  { label: 'VIX → QQQ & SPY volatility', mode: 'guided', lookback: 3, budget: 2, tickers: 'QQQ, SPY', macros: ['VIXCLS'],
    question: 'Does the VIX predict the next 5-day volatility of QQQ or SPY?' },
  { label: 'Oil → energy & airline stocks', mode: 'guided', lookback: 3, budget: 2, tickers: 'XLE, JETS', macros: ['DCOILWTICO'],
    question: 'Do daily changes in WTI crude oil predict next-day returns of energy stocks (XLE) and airlines (JETS)?' },
  { label: 'Credit stress → stocks', mode: 'guided', lookback: 5, budget: 2, tickers: 'SPY', macros: ['BAMLH0A0HYM2'],
    question: 'Does widening of the high-yield credit spread predict weaker S&P 500 (SPY) returns or higher volatility over the following days?' },
  { label: 'Treasury yields → tech stocks', mode: 'guided', lookback: 5, budget: 2, tickers: 'QQQ, XLK', macros: ['DGS10'],
    question: 'Do rising 10-year Treasury yields predict lower next-day returns of technology stocks (QQQ, XLK)?' },
  { label: 'Dollar → gold', mode: 'guided', lookback: 5, budget: 2, tickers: 'GLD', macros: ['DTWEXBGS'],
    question: 'Does a stronger US dollar predict lower next-day gold (GLD) returns?' },
  { label: 'AI chips: NVDA → semis', mode: 'guided', lookback: 3, budget: 2, tickers: 'NVDA, SMH, AMD',
    question: 'Do NVIDIA daily returns lead the returns of the broader semiconductor ETF (SMH) and AMD the next day?' },
  { label: 'AI explores: volatility & credit', mode: 'autonomous', lookback: 3, budget: 3, theme: 'Market volatility, credit spreads and US equity returns' },
]
const DEFAULT_THEME = 'What drives moves in the stock market?'

const splitList = text => text.split(/[\s,]+/).map(s => s.trim().toUpperCase()).filter(Boolean)

/**
 * Functional input only (no charts). Mount anywhere:
 *   <ResearchLauncher onComplete={report => ...} onProgress={report => ...} />
 * `report` is the JSON documented in docs/MARKETMIND_RESEARCH_API.md.
 */
export default function ResearchLauncher({ onComplete, onProgress, prefill }) {
  const [catalog, setCatalog] = useState(null)
  const [mode, setMode] = useState('autonomous')
  const [theme, setTheme] = useState('')
  const [question, setQuestion] = useState('')
  const [lookback, setLookback] = useState(3)
  const [budget, setBudget] = useState(3)
  const [tickers, setTickers] = useState('')
  const [macros, setMacros] = useState([])
  const [demoKind, setDemoKind] = useState('demo_autonomous')
  const [pairRequest, setPairRequest] = useState(null)

  // Hand-off from "Ask the globe": open guided mode with the question and tickers filled in (the user starts the run).
  useEffect(() => {
    if (prefill?.pair) { setMode('pairs'); setPairRequest({ ...prefill.pair, n: prefill.nonce || Date.now() }); return }
    if (!prefill?.question) return
    setMode('guided')
    setQuestion(prefill.question)
    setTickers((prefill.tickers || []).join(', '))
  }, [prefill])
  const [runId, setRunId] = useState('')
  const [report, setReport] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [stamp, setStamp] = useState(null)
  const abortRef = useRef(null)

  useEffect(() => {
    getCatalog().then(setCatalog).catch(e => setError(e.message))
    return () => abortRef.current && abortRef.current.abort()
  }, [])

  const gemini = catalog?.providers?.gemini?.configured
  const equity = catalog?.providers?.equity_data
  const alpaca = Boolean(equity?.usable)
  const CAT = { equity_index: 'Stock indices', volatility: 'Volatility', rates: 'Interest rates', credit: 'Credit', commodities: 'Commodities',
    fx: 'Currencies', equity: 'Popular stocks and ETFs', inflation: 'Inflation', labour: 'Jobs' }
  const availGroups = Object.entries((catalog?.variables || []).filter(v => v.available && v.supported !== false).reduce((g, v) => {
    const k = CAT[v.category] || v.category
    ;(g[k] = g[k] || []).push(v)
    return g
  }, {}))
  const live = mode !== 'demo'

  function runExample(ex) {
    if (ex.pair) { setMode('pairs'); setPairRequest({ ...ex.pair, n: Date.now() }); return }
    setMode(ex.mode); setLookback(ex.lookback); setBudget(ex.budget)
    setTickers(ex.tickers || ''); setMacros(ex.macros || [])
    setQuestion(ex.question || ''); setTheme(ex.theme || '')
    const body = { mode: ex.mode, lookback_years: ex.lookback, max_hypotheses: ex.budget }
    if (ex.tickers || ex.macros?.length) body.universe = { tickers: splitList(ex.tickers || ''), macro_series: ex.macros || [] }
    if (ex.mode === 'autonomous') body.theme = ex.theme; else body.question = ex.question
    launch(body)
  }

  function buildBody() {
    if (mode === 'demo') {
      return demoKind === 'demo_autonomous' ? { mode: 'demo_autonomous' } : { preset_id: demoKind }
    }
    const body = { mode, lookback_years: lookback, max_hypotheses: budget }
    const t = splitList(tickers)
    if (t.length) body.universe = { tickers: t, macro_series: [] }
    if (mode === 'autonomous') body.theme = theme.trim() || DEFAULT_THEME
    else body.question = question.trim()
    return body
  }

  async function launch(override) {
    setError(''); setReport(null); setRunId(''); setStamp(null); setBusy(true)
    abortRef.current = new AbortController()
    try {
      const { run_id } = await startRun(override || buildBody())
      setRunId(run_id)
      const final = await pollRun(run_id, r => { setReport(r); onProgress && onProgress(r) },
                                  { signal: abortRef.current.signal })
      if (final.status === 'failed') setError(failureText(final))
      else {
        if (final.summary?.counts) setStamp({ kind: final.summary.counts.supported > 0 ? 'tick' : 'cross', n: Date.now() })
        onComplete && onComplete(final)
      }
    } catch (e) {
      if (e.name !== 'AbortError') setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const blocked = live && mode !== 'pairs' && catalog && !gemini
  const needsQuestion = mode === 'guided' && question.trim().length < 10
  return (
    <section aria-label="Launch investigation" className="rs-launcher">
      <CandleBackground />
      {stamp && <ResultStamp key={stamp.n} kind={stamp.kind} />}
      <fieldset disabled={busy}>
        <legend>Research mode</legend>
        <label><input type="radio" checked={mode === 'autonomous'} onChange={() => setMode('autonomous')} /> Autonomous Discovery</label>{' '}
        <label><input type="radio" checked={mode === 'guided'} onChange={() => setMode('guided')} /> Guided Research</label>{' '}
        <label><input type="radio" checked={mode === 'pairs'} onChange={() => setMode('pairs')} /> Pair Divergence (statistical arbitrage)</label>{' '}
        <label><input type="radio" checked={mode === 'demo'} onChange={() => setMode('demo')} /> Demo / test (synthetic data only)</label>
      </fieldset>

      {catalog && (
        <small>
          Gemini: {gemini ? 'configured' : 'NOT configured'} · Stock/ETF data: {alpaca ? equity.provider : `unavailable (${equity?.status || 'no provider'})`} ·
          Macro (FRED): {catalog.providers.fred.mode}
        </small>
      )}
      {gemini && mode !== 'pairs' && (
        <div className="rs-examples">
          <div className="rs-examples-title">Live examples on real market data (one click)</div>
          {EXAMPLES.map(ex => (
            <button key={ex.label} type="button" className="rs-example" disabled={busy || (ex.tickers && !alpaca)} onClick={() => runExample(ex)}>{ex.label}</button>
          ))}
        </div>
      )}
      {blocked && <p role="alert">Live research needs Gemini. Set GEMINI_API_KEY in backend/.env, or use the Demo mode.</p>}

      {mode === 'autonomous' && (
        <label>Research theme (optional)<br />
          <input style={{ width: '100%' }} maxLength={200} value={theme} disabled={busy}
                 placeholder={DEFAULT_THEME} onChange={e => setTheme(e.target.value)} />
        </label>
      )}
      {mode === 'guided' && (
        <label>Your research question<br />
          <textarea style={{ width: '100%' }} rows={3} maxLength={500} value={question} disabled={busy}
                    placeholder="e.g. Do rising Treasury yields predict lower technology-stock returns?"
                    onChange={e => setQuestion(e.target.value)} />
        </label>
      )}
      {mode === 'demo' && (
        <label>Demo scenario (SYNTHETIC data, not live research)<br />
          <select value={demoKind} disabled={busy} onChange={e => setDemoKind(e.target.value)}>
            <option value="demo_autonomous">Scripted autonomous demo (3 hypotheses)</option>
            {LEGACY.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
          </select>
        </label>
      )}

      {mode === 'pairs' && <PairsTester request={pairRequest} />}

      {live && mode !== 'pairs' && (
        <>
          <div>
            <label>Lookback{' '}
              <select value={lookback} disabled={busy} onChange={e => setLookback(Number(e.target.value))}>
                {(catalog?.lookback_years || [1, 3, 5]).map(y => <option key={y} value={y}>{y} year{y > 1 ? 's' : ''}</option>)}
              </select>
            </label>{' '}
            <label>Hypothesis budget{' '}
              <input type="number" min={1} max={catalog?.max_hypotheses?.max || 5} value={budget} disabled={busy}
                     onChange={e => setBudget(Math.max(1, Math.min(5, Number(e.target.value) || 1)))} style={{ width: 60 }} />
            </label>
          </div>
          <label>Stocks and ETFs to keep track of (comma separated){' '}
            <input value={tickers} disabled={busy || !alpaca} placeholder={alpaca ? 'SPY, NVDA, SMH' : 'needs TIINGO_API_KEY'}
                   onChange={e => setTickers(e.target.value)} />
          </label>
          <details className="rs-avail">
            <summary>What data is available?</summary>
            <p className="rs-sub">The AI can use any series below. Stocks and ETFs: any valid US ticker{alpaca ? '' : ' (needs a price-data key)'}, including ones you list above.</p>
            {availGroups.map(([cat, vars]) => (
              <div key={cat} className="rs-avail-group">
                <b>{cat}</b>
                <ul>{vars.map(v => <li key={v.id}><code>{v.id}</code> {v.label}{v.frequency ? <span className="rs-sub"> ({v.frequency})</span> : null}</li>)}</ul>
              </div>
            ))}
          </details>
        </>
      )}

      {mode !== 'pairs' && <div>
        <button className="rs-launch-btn" onClick={() => launch()} disabled={busy || blocked || needsQuestion}>
          {busy ? 'Running…' : 'Launch Investigation'}
        </button>
      </div>}

      {error && <p role="alert" className="rs-error">{error}</p>}
      {runId && <p className="rs-jobid">Job ID: <code>{runId}</code></p>}
      {report && (report.schema_version?.startsWith('1') ? <Progress report={report} /> : <ResearchReport report={report} />)}
    </section>
  )
}

function failureText(r) {
  const errs = r.errors || (r.error ? [r.error] : [])
  return errs.length ? errs.map(e => e.message).join('\n') : 'Run failed'
}

/** Plain-text progress + summary only (visual presentation belongs to the dashboard). */
function Progress({ report }) {
  if (report.schema_version && report.schema_version.startsWith('1')) {
    return (
      <div>
        <p>Status: {report.status} · stage: {report.current_stage}</p>
        <ol>{report.stages.map(s => <li key={s.id}>{s.id}: {s.status}</li>)}</ol>
        {report.status === 'completed' && <p>{report.summary.overall}</p>}
      </div>
    )
  }
  return null
}
