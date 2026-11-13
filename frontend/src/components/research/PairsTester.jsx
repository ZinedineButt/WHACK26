import React, { useEffect, useState } from 'react'
import { LineChart, SignedBars } from './charts'
import { simulateLikeRequest } from '../../researchApi'
import './research.css'

// One-click pairs: real, economically linked assets (stocks, ETFs and commodity prices).
const EXAMPLES = [
  { a: 'KO', b: 'PEP', label: 'Coca-Cola vs PepsiCo' },
  { a: 'V', b: 'MA', label: 'Visa vs Mastercard' },
  { a: 'XOM', b: 'CVX', label: 'Exxon vs Chevron' },
  { a: 'GLD', b: 'SLV', label: 'Gold vs Silver' },
  { a: 'DCOILWTICO', b: 'DCOILBRENTEU', label: 'WTI vs Brent crude oil' },
  { a: 'HD', b: 'LOW', label: 'Home Depot vs Lowe’s' },
]
// Synthetic demos (clearly labelled): one pair built to snap back, one pair of unrelated random walks.
const DEMOS = [
  { a: 'DEMO_SNAP_A', b: 'DEMO_SNAP_B', label: 'Demo: a pair that snaps back' },
  { a: 'DEMO_RW_A', b: 'DEMO_RW_B', label: 'Demo: a pair that does not' },
]

const VERDICT = {
  converges: ['good', 'Gaps tend to close'],
  mixed: ['warn', 'Some sign of closing'],
  did_not_hold: ['bad', 'Did not hold on new data'],
  no_evidence: ['mute', 'No reliable closing'],
  inconclusive: ['warn', 'Too few divergences'],
}
const RESULT_TXT = { pass: 'Passed', fail: 'Failed', inconclusive: 'Unclear' }

const pp = v => (v == null ? '–' : `${v > 0 ? '+' : ''}${v.toFixed(2)}`)
const pct = v => (v == null ? '–' : `${Math.round(v * 100)}%`)
const cap = t => (t ? t.charAt(0).toUpperCase() + t.slice(1) : t)

/** A rendering problem in one result must show a message, not blank the whole page. */
class ResultBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error: null } }
  static getDerivedStateFromError(error) { return { error } }
  componentDidUpdate(prev) { if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null }) }
  render() {
    if (this.state.error) return <p className="rs-error" role="alert">Could not display this result ({String(this.state.error.message || this.state.error).slice(0, 160)}). Try another pair or reload.</p>
    return this.props.children
  }
}

function Step({ n, title, sub, children }) {
  return (
    <section className="rs-section in">
      <header><span className="rs-step">{n}</span><div><h3>{title}</h3>{sub && <p>{sub}</p>}</div></header>
      {children}
    </section>
  )
}

function Fold({ title, open = false, children }) {
  return (
    <details className="rs-fold" open={open}>
      <summary>{title}</summary>
      <div className="rs-fold-body">{children}</div>
    </details>
  )
}

function Checks({ checks }) {
  return (
    <ul className="rs-tests">
      {checks.map((c, i) => (
        <li key={c.id} className={`rs-test ${c.result}`} style={{ '--i': i }}>
          <span className="rs-test-dot" />
          <div><b>{c.name}</b><span className="rs-test-res">{RESULT_TXT[c.result]}</span>
            <details className="rs-why"><summary>Why this check, and what it found</summary>
              <div className="rs-test-ex">{c.detail}</div></details></div>
        </li>))}
    </ul>
  )
}

function Literature({ lit, busy }) {
  if (busy && !lit) return <div className="rs-note rs-pulse">Searching published research and comparing…</div>
  if (!lit) return <div className="rs-note">Literature comparison unavailable.</div>
  const { critique: c, literature: l } = lit
  return (
    <>
      <div className="rs-confidence"><span>How confident should we be?</span>
        <div className="rs-meter"><i className={c.confidence} /></div><b>{c.confidence}</b>
        <small>never higher than the evidence allows{c.source !== 'gemini' ? ' · scripted text, not an AI reading' : ''}</small></div>
      <p className="rs-final"><b>How this compares with published work:</b> {c.comparison}</p>
      <Fold open title="Where this could be wrong">
        <ul className="rs-crit">{c.critique.map((p, i) => (
          <li key={i} className={p.severity} style={{ '--i': i }}><b>{p.severity}</b> {p.point}</li>))}</ul>
      </Fold>
      <Fold open title={`Papers on pairs trading and statistical arbitrage (${l.papers.length})`}>
        {l.status !== 'completed' && <p className="rs-sub">{l.note}</p>}
        <div className="rs-papers">{l.papers.map((p, i) => {
          const a = c.paper_assessments.find(x => x.paper_index === p.index)
          return (
            <article key={p.index} className="rs-paper" style={{ '--i': i }}>
              <div className="rs-paper-top"><b>[{p.index}]</b> {p.url ? <a href={p.url} target="_blank" rel="noopener noreferrer">{p.title}</a> : p.title}</div>
              <div className="rs-sub">{p.year}{p.venue ? ` · ${p.venue}` : ''}{p.cited_by != null ? ` · ${p.cited_by} citations` : ''}</div>
              {a && <div className={`rs-stance ${a.stance}`}>{a.stance.replace('_', ' ')}<span>{a.note}</span></div>}
            </article>)
        })}</div>
        <p className="rs-sub">Papers were looked up automatically ({l.provider || 'no source'}). The AI judges how they compare with this result but never invents citations or numbers.</p>
      </Fold>
    </>
  )
}

export function Result({ r, lit, litBusy }) {
  const [cls, label] = VERDICT[r.verdict] || VERDICT.no_evidence
  const oos = r.out_of_sample
  const e = r.settings.entry_z
  const z = r.chart.z
  const flat = v => z.map(() => v)
  const bars = r.horizons.map(h => ({ key: h.horizon_days, label: `${h.horizon_days} day${h.horizon_days > 1 ? 's' : ''} later`,
    value: h.all.mean ?? 0, sig: h.all.p != null && h.all.p < 0.05 }))
  const outdated = !r.placebo || !r.robustness || !r.peer_pairs || !r.checks
  const rel = r.relationship, cur = r.current, st = r.strength
  const pl = r.placebo || { p: null }
  const rb = r.robustness || { by_year: [], thresholds: [], costs: [] }
  const pe = r.peer_pairs || { pairs: [], tested: 0, share_closing: null }
  const checks = r.checks || []
  const today = Math.abs(cur.z) >= e
    ? `Right now ${cur.z < 0 ? r.a.label : r.b.label} looks cheap against ${cur.z < 0 ? r.b.label : r.a.label} (gap = ${cur.z.toFixed(1)} standard deviations).`
    : `Right now the gap is ${cur.z.toFixed(1)} standard deviations, inside the normal range.`
  const sim = r.simulation
  return (
    <div className="rs-pairres">
      {r.synthetic && <div className="rs-stopbanner" role="status"><b>Demo with synthetic data.</b> These prices were generated by the program to show what a pair that snaps back, and one that does not, look like. Nothing here is a market result.</div>}

      <div className={`rs-headline ${cls}`}>
        <div className="rs-hl-kicker">{r.a.label} vs {r.b.label}</div>
        {st?.pct != null && (
          <div className="rs-overall-strength"><span>Overall strength of evidence</span>
            <div className="rs-meter big"><i style={{ width: `${st.pct}%` }} className={st.pct >= 60 ? 'high' : st.pct >= 40 ? 'moderate' : 'low'} /></div>
            <b>{cap(st.label)} · {st.pct}/100</b></div>)}
        <p className="rs-hl-sentence"><span className={`rs-chip ${cls}`}>{label}</span> {r.verdict_reason}</p>
        <ul className="rs-pairstats">
          <li><b>{r.events.total}</b><span>times the two drifted apart (beyond {e} standard deviations)</span></li>
          <li><b>{oos && oos.n ? pct(oos.converged_share) : '–'}</b><span>closed within {r.best_horizon_days ?? '?'} day(s) on the held-out final 30% ({oos ? oos.n : 0} cases)</span></li>
          <li><b>{rel.half_life_days ? `${rel.half_life_days} d` : 'none'}</b><span>typical time to close half a gap{rel.half_life_days ? '' : ' (no steady pull-back found)'}</span></li>
          <li><b>{r.trades_test.n ? pp(r.trades_test.avg_net) : '–'}</b><span>average profit per trade after costs, held-out period ({r.trades_test.n} trades, % of one leg)</span></li>
        </ul>
        <p className="rs-hl-note">{today} A pattern that holds up is useful but is not a guarantee: relationships can break for good.</p>
      </div>

      {outdated && <div className="rs-stopbanner" role="status"><b>The backend is running an older version.</b> Stop it and start it again (<code>..venvScriptspython.exe app.py</code> in the backend folder) to get the full set of checks, similar-pair tests, literature comparison and simulation.</div>}

      {checks.length > 0 && <Fold open title="The checks (title and result; open one for the reasoning)"><Checks checks={checks} /></Fold>}

      <Step n="1" title="Do gaps close, and how fast?"
            sub="We mark every day the two assets drifted apart by more than the trigger, then measure how much the gap narrowed 1 to 20 days later. Day 1 can show nothing (they may already have moved together) while later days close.">
        <SignedBars rows={bars} />
        <p className="rs-sub">A bar to the right means the gap narrowed on average, in % points. Star = clearly not luck.</p>
        <h4>The gap over time (standard deviations from its recent normal)</h4>
        <LineChart dates={r.chart.dates} label="Pair gap z-score"
          series={[{ name: 'Gap', values: z, color: 'var(--rs-accent)' }, { name: `+${e}`, values: flat(e), color: 'var(--rs-amber)' }, { name: `−${e}`, values: flat(-e), color: 'var(--rs-amber)' }]}
          region={{ from: r.chart.split_date, label: 'Held-out 30%' }} />
        <Fold title="Numbers for each horizon">
          <table className="rs-table">
            <thead><tr><th>Days later</th><th>All data</th><th>First 70%</th><th>Final 30% (held out)</th></tr></thead>
            <tbody>{r.horizons.map(h => (
              <tr key={h.horizon_days}><td>{h.horizon_days}</td>
                <td>{pp(h.all.mean)} ({pct(h.all.converged_share)} closed, n={h.all.n})</td>
                <td>{pp(h.train.mean)} (n={h.train.n})</td>
                <td>{pp(h.test.mean)} ({pct(h.test.converged_share)}, n={h.test.n}{h.test.p != null ? `, p=${h.test.p}` : ''})</td></tr>))}</tbody>
          </table>
          <p className="rs-sub">Do-nothing baseline: on ordinary days with no real gap the average closing was {pp(r.control_mean_reversion['1'])} after 1 day and {pp(r.control_mean_reversion['10'])} after 10.</p>
        </Fold>
      </Step>

      <Step n="2" title="Is it real or just luck?"
            sub="Three stress tests: compare with randomly chosen days, check it is stable across years, and check it does not depend on the trigger level or on tiny trading costs.">
        <div className="rs-verdictbox">
          {pl.p == null ? <p>Too few divergences for the random-days comparison.</p>
            : <p><b>Random-days test:</b> the {pl.events} real divergences closed by {pp(pl.observed_mean)} points on average at {pl.horizon_days} day(s); the same number of random mild-gap days closed by {pp(pl.null_mean)} on average (chance of doing this well by luck: {pl.p < 0.001 ? '<0.1%' : `${(pl.p * 100).toFixed(1)}%`}).</p>}
        </div>
        {(rb.by_year || []).length > 0 && (<><h4>Closing by calendar year (years with at least 3 divergences)</h4>
          <SignedBars rows={rb.by_year.map(y => ({ key: y.year, label: `${y.year} (${y.events} events)`, value: y.mean, sig: false }))} /></>)}
        <Fold title="Trigger level and trading-cost sensitivity">
          <table className="rs-table"><thead><tr><th>Trigger</th><th>Divergences</th><th>Average closing</th><th>Share that closed</th></tr></thead>
            <tbody>{rb.thresholds.map(t => <tr key={t.entry_z}><td>{t.entry_z} sd</td><td>{t.events}</td><td>{pp(t.mean)}</td><td>{pct(t.converged_share)}</td></tr>)}</tbody></table>
          <table className="rs-table"><thead><tr><th>Cost per leg per side</th><th>Avg profit/trade, all data</th><th>Held-out</th></tr></thead>
            <tbody>{rb.costs.map(c => <tr key={c.cost_bps_per_leg}><td>{c.cost_bps_per_leg} bp</td><td>{pp(c.avg_net_all)}</td><td>{pp(c.avg_net_test)}</td></tr>)}</tbody></table>
        </Fold>
      </Step>

      <Step n="3" title="Does it still work on data it has not seen?"
            sub={`The best closing time was chosen using only the first 70% of days (${r.events.train} divergences) and judged once on the final 30% (${r.events.test}).`}>
        <div className="rs-verdictbox">
          {oos && oos.n ? <p>On the held-out period, {pct(oos.converged_share)} of {oos.n} divergences had narrowed after {r.best_horizon_days} day(s), by {pp(oos.mean)} points on average{oos.p != null ? ` (chance of luck: ${oos.p < 0.001 ? '<0.1%' : `${(oos.p * 100).toFixed(1)}%`})` : ''}.
            The simple trading rule made {r.trades_test.n} trades there, averaging {pp(r.trades_test.avg_net)} each after costs with {pct(r.trades_test.win_rate)} winners.</p>
            : <p>Too few held-out divergences to judge.</p>}
        </div>
      </Step>

      <Step n="4" title="Does it work on similar pairs?"
            sub="If pull-back is real for related assets, other linked pairs should show it too. The comparison pairs are fixed by the program, not chosen by the AI.">
        {pe.pairs.length ? (
          <table className="rs-table"><thead><tr><th>Pair</th><th>Result</th><th>Divergences</th><th>Held-out closing</th><th>Half-life</th></tr></thead>
            <tbody>{pe.pairs.map(p => (
              <tr key={`${p.a}-${p.b}`}><td>{p.a} vs {p.b}</td><td>{(VERDICT[p.verdict] || VERDICT.no_evidence)[1]}</td><td>{p.events}</td>
                <td>{p.oos_n ? `${pct(p.oos_converged_share)} of ${p.oos_n}` : '–'}</td><td>{p.half_life_days ? `${p.half_life_days} d` : 'none'}</td></tr>))}</tbody></table>
        ) : <div className="rs-note">No comparable pairs were available for this asset type.</div>}
        {pe.share_closing != null && <p className="rs-sub">{Math.round(pe.share_closing * pe.tested)} of {pe.tested} comparable pairs showed closing.</p>}
      </Step>

      <Step n="5" title="What do the published papers say, and what could be wrong?"
            sub="The AI compares this result with research papers it retrieved, and argues against it. It may only cite retrieved papers and may not introduce numbers.">
        <Literature lit={lit} busy={litBusy} />
      </Step>

      {sim && (
        <Step n="6" title="Try it with your own money (pretend)"
              sub="The frozen rule replayed on the held-out period only, starting with $10,000 split equally between the two legs.">
          <ul className="rs-pairstats">
            <li><b>${sim.final_value.toLocaleString('en-US', { maximumFractionDigits: 0 })}</b><span>final value ({pp(sim.return_pct)}%)</span></li>
            <li><b>{sim.trades}</b><span>trades, {pct(sim.win_rate)} winners</span></li>
            <li><b>{sim.max_drawdown_pct.toFixed(1)}%</b><span>worst fall from a peak</span></li>
          </ul>
          <LineChart dates={sim.curve.map(c => c.date)} zero={false} label="Pretend account value"
            series={[{ name: 'Pairs rule', values: sim.curve.map(c => c.value), color: 'var(--rs-accent)' }]} />
          <p className="rs-sub">{sim.note}</p>
        </Step>
      )}

      <Fold title="How this was tested">
        <ul className="rs-list">{r.assumptions.map((a, i) => <li key={i}>{a}</li>)}</ul>
        <p className="rs-sub">Data: {r.settings.first_date} to {r.settings.last_date} ({r.settings.days_used} trading days){r.synthetic ? ', SYNTHETIC' : ', real adjusted daily prices'}.</p>
      </Fold>
    </div>
  )
}

/** Test whether two assets that drift apart come back together (statistical arbitrage), at the same depth as other research. */
export default function PairsTester({ request }) {
  const [a, setA] = useState('KO')
  const [b, setB] = useState('PEP')
  const [entry, setEntry] = useState(2)
  const [years, setYears] = useState(3)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [res, setRes] = useState(null)
  const [lit, setLit] = useState(null)
  const [litBusy, setLitBusy] = useState(false)
  const [ideas, setIdeas] = useState(null)
  const [ideaBusy, setIdeaBusy] = useState(false)

  async function run(x = a, y = b) {
    const body = { a: x, b: y, entry_z: Number(entry), lookback_years: Number(years) }
    setBusy(true); setErr(''); setRes(null); setLit(null)
    try {
      const core = await simulateLikeRequest('/api/research/pairs', body)
      setRes(core)
      setLitBusy(true)
      simulateLikeRequest('/api/research/pairs/literature', body).then(setLit).catch(() => setLit(null)).finally(() => setLitBusy(false))
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  function pick(p) { setA(p.a); setB(p.b); run(p.a, p.b) }
  async function suggest() {
    setIdeaBusy(true); setErr('')
    try { setIdeas(await simulateLikeRequest('/api/research/pairs/suggest')) }
    catch (e) { setErr(e.message) } finally { setIdeaBusy(false) }
  }

  // A one-click example elsewhere on the page (or "Ask the globe") can request a specific pair.
  useEffect(() => {
    if (request?.a && request?.b) { setA(request.a); setB(request.b); run(request.a, request.b) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request])

  return (
    <div className="rs-pairs embedded">
      <p className="rs-sub"><b>Statistical arbitrage:</b> if two related assets drift apart, do they come back together? Pick a pair (stocks, ETFs or commodities). We find every day they diverged, measure whether the gap closed over the next 1 to 20 days, stress-test it, check it on data held back from the search and on similar pairs, compare it with published papers and replay a simple trading rule with costs.</p>

      <div className="rs-examples">
        <div className="rs-examples-title">One-click pairs on real prices</div>
        {EXAMPLES.map(p => <button key={p.label} type="button" className="rs-example" disabled={busy} onClick={() => pick(p)}>{p.label}</button>)}
        <button type="button" className="rs-example ai" disabled={busy || ideaBusy} onClick={suggest}>{ideaBusy ? 'Thinking…' : '✦ Let the AI suggest pairs'}</button>
      </div>
      <div className="rs-examples">
        <div className="rs-examples-title">Demo (synthetic data, to see what a result looks like)</div>
        {DEMOS.map(p => <button key={p.label} type="button" className="rs-example" disabled={busy} onClick={() => pick(p)}>{p.label}</button>)}
      </div>

      {ideas && (
        <div className="rs-ideas">
          <div className="rs-sub">{ideas.source === 'gemini' ? 'The AI suggests these linked pairs:' : 'Suggested linked pairs:'}</div>
          {ideas.pairs.map(p => (
            <button key={`${p.a}-${p.b}`} type="button" className="rs-idea" disabled={busy} onClick={() => pick(p)}>
              <b>{p.a_name} vs {p.b_name}</b><span>{p.reason}</span>
            </button>
          ))}
        </div>
      )}

      <form className="rs-pairform" onSubmit={e => { e.preventDefault(); run() }}>
        <label>Asset X <input value={a} onChange={e => setA(e.target.value.toUpperCase())} maxLength={14} placeholder="KO" /></label>
        <label>Asset Y <input value={b} onChange={e => setB(e.target.value.toUpperCase())} maxLength={14} placeholder="PEP" /></label>
        <label>Divergence trigger
          <select value={entry} onChange={e => setEntry(e.target.value)}>
            <option value={1.5}>1.5 standard deviations (more events)</option>
            <option value={2}>2 standard deviations</option>
            <option value={2.5}>2.5 standard deviations (rarer, bigger)</option>
          </select></label>
        <label>History
          <select value={years} onChange={e => setYears(e.target.value)}>
            <option value={3}>3 years</option><option value={5}>5 years</option><option value={10}>10 years</option>
          </select></label>
        <button className="rs-launch-btn" type="submit" disabled={busy || !a || !b}>{busy ? 'Testing…' : 'Launch Investigation'}</button>
      </form>
      <p className="rs-sub">Use stock or ETF tickers (e.g. KO, SPY, GLD) or price series such as DCOILWTICO (WTI oil), DCOILBRENTEU (Brent) or SP500.</p>

      {err && <p className="rs-error" role="alert">{err}</p>}
      {busy && <div className="rs-note rs-pulse">Downloading prices and running every check… this takes up to 20 seconds.</div>}
      {res && <ResultBoundary resetKey={res}><Result r={res} lit={lit} litBusy={litBusy} /></ResultBoundary>}
    </div>
  )
}
