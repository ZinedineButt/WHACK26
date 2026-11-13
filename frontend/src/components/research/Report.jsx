import React, { useState } from 'react'
import { CausalGraph, Gauge, LineChart, Num, ScatterChart, SignedBars } from './charts'
import { simulateStrategy } from '../../researchApi'
import { useInView } from './anim'
import { plain, testInfo, resultText, nodeName, causalStory, linkStrength, pPlain } from './plain'
import './research.css'

const VERDICT = {
  detected: ['good', 'Link found'], not_detected: ['bad', 'No link found'], inconclusive: ['warn', 'Unclear'],
  evidence_consistent: ['good', 'Cause plausible'], contradicted: ['bad', 'Cause contradicted'], not_established: ['warn', 'Cause not proven'],
  supported: ['good', 'Held up'], rejected: ['bad', 'Did not hold up'],
  pass: ['good', 'Passed'], fail: ['bad', 'Failed'], not_run: ['mute', 'Not run'],
  generalises: ['good', 'Works on similar assets'], partial: ['warn', 'Works on some similar assets'], does_not_generalise: ['bad', "Doesn't carry over"],
}
const Chip = ({ v, label, i = 0 }) => {
  const [cls, text] = VERDICT[v] || ['mute', v]
  return <span className={`rs-chip ${cls}`} style={{ '--i': i }}>{label ? `${label}: ` : ''}{text}</span>
}
const pct = v => (v == null ? '–' : `${(v * 100).toFixed(1)}%`)

const Waiting = ({ what }) => <div className="rs-note rs-pulse">Running {what}…</div>

/** A collapsible block inside a step: the essential result stays visible, the detail folds away. */
function Fold({ title, children, open = false }) {
  return (
    <details className="rs-fold" open={open}>
      <summary>{title}</summary>
      <div className="rs-fold-body">{children}</div>
    </details>
  )
}

function Section({ n, title, sub, children }) {
  const [ref, seen] = useInView()
  return (
    <section ref={ref} className={`rs-section ${seen ? 'in' : ''}`}>
      <header><span className="rs-step">{n}</span><div><h3>{title}</h3>{sub && <p>{sub}</p>}</div></header>
      {children}
    </section>
  )
}

function Tile({ label, value, decimals = 2, suffix = '', tone, hint }) {
  return (
    <div className={`rs-tile ${tone || ''}`} title={hint}>
      <div className="rs-tile-val"><Num value={value} decimals={decimals} suffix={suffix} /></div>
      <div className="rs-tile-label">{label}</div>
    </div>
  )
}

function TestList({ tests, n }) {
  return (
    <ul className="rs-tests">
      {tests.map((t, i) => {
        const info = testInfo(t, n)
        return (
          <li key={t.id} className={`rs-test ${t.result}`} style={{ '--i': i }} title={`Technical name: ${t.name}`}>
            <span className="rs-test-dot" />
            <div><b>{info.name}</b><span className="rs-test-res">{resultText(t.result)}</span>
              <details className="rs-why"><summary>Why this check, and what it found</summary>
                {info.what && <div className="rs-test-what">{info.what}</div>}
                <div className="rs-test-ex"><i>Result:</i> {plain(t.explanation, n)}</div></details></div>
          </li>)
      })}
    </ul>
  )
}

const TRANSFORM = { log_return: 'daily % change', log_diff: 'daily % change', diff: 'daily change', level: 'level', fwd_logvol_5: 'next 5-day volatility', fwd_logvol_10: 'next 10-day volatility' }
const tName = t => TRANSFORM[t] || t.replace(/_/g, ' ')
const capFirst = t => (t ? t.charAt(0).toUpperCase() + t.slice(1) : t)
const axisLabel = v => capFirst(`${v.label} (${tName(v.transform)})`)
const fwdDays = n => ({ fwd_logvol_5: 5, fwd_logvol_10: 10 }[n?.outcome?.transform])
// A zero delay only exists for forward-volatility outcomes, where the outcome window starts the day AFTER the predictor (no look-ahead).
const lagText = (k, n) => (k === 0 ? (fwdDays(n) ? `Next ${fwdDays(n)} days` : 'Same day') : `${k} day${k > 1 ? 's' : ''} later`)

function Backtest({ bt, curve, n }) {
  if (!bt?.applicable) return <div className="rs-note">Backtest not applicable. {bt?.reason}</div>
  const H = bt.holdout, D = bt.development
  const hrows = curve.filter(r => r.sample === "holdout")
  const rows = hrows.length ? hrows : curve
  const dates = rows.map(r => r.date)
  return (
    <>
      <p className="rs-rule">The rule, fixed in advance: each day, if {n.predictor.label} rises, {bt.sign >= 0 ? 'buy' : 'short'} {n.outcome.label} for the next {n.lag_days || 1} day(s); if it falls, do the opposite. Each trade pays {bt.cost_bps} bps in costs.</p>
      <div className="rs-tiles">
        <Tile label="Return per unit of risk, final quarter (Sharpe)" value={H.sharpe} tone={H.sharpe > 0 ? 'good' : 'bad'} hint="Annualised, after costs, on the sealed holdout" />
        <Tile label="Days called right, final quarter" value={H.hit_rate == null ? null : H.hit_rate * 100} decimals={1} suffix="%" tone={H.hit_rate > 0.5 ? 'good' : 'bad'} />
        <Tile label="Strategy return, final quarter" value={H.total_return_pct} suffix="%" tone={H.total_return_pct > 0 ? 'good' : 'bad'} />
        <Tile label="Just holding it, same period" value={H.buy_hold_total_return_pct} suffix="%" />
        <Tile label="Worst loss from a peak" value={H.max_drawdown_pct} suffix="%" tone="bad" />
        <Tile label="Return per unit of risk, older data" value={D.sharpe} hint="Development sample, for comparison only" />
      </div>
      <h4>Final-quarter test: strategy vs simply holding <Chip v={bt.verdict} /></h4>
      <LineChart dates={dates} label="Holdout equity curve"
        series={[{ name: 'Strategy (after costs)', values: rows.map(r => r.strategy_holdout), color: 'var(--rs-accent)' },
                 { name: 'Just holding it', values: rows.map(r => r.buy_hold_holdout), color: 'var(--rs-muted)' }]}
        />
      <p className="rs-sub">Passing needs three things: a positive average daily profit after costs, a confidence range that stays above zero{H.mean_daily_net_ci_95 ? ` (${H.mean_daily_net_ci_95.map(v => v.toFixed(2)).join('% to ')}% per day here)` : ''}, and more than 50% of days called right. Nothing was tuned on this period.</p>
    </>
  )
}

function Peers({ g }) {
  if (!g || g.status !== 'completed') return <div className="rs-note">{g?.verdict_rule || 'Not run.'}</div>
  const rows = g.peers.map(p => ({ key: p.variable_id, label: p.variable_id, title: p.label, value: p.r, sig: p.significant_after_holm && p.same_sign_as_primary }))
  return (
    <>
      <div className="rs-peerhead"><Chip v={g.verdict} /> <span className="rs-sub">{Math.round((g.share_same_sign ?? 0) * 100)}% of similar assets move the same way; {Math.round((g.share_significant_same_sign ?? 0) * 100)}% show a clear link. It counts as working on similar assets when at least 60% move the same way and at least 50% show a clear link.</span></div>
      <SignedBars rows={rows} />
      <Fold title="Numbers for each similar asset"><table className="rs-table"><thead><tr><th>Asset</th><th>Link (r)</th><th>Chance of luck</th><th>Same direction?</th><th>Return per risk</th><th>Days right</th></tr></thead>
        <tbody>{g.peers.map(p => (
          <tr key={p.variable_id}><td>{p.variable_id}</td><td>{p.r.toFixed(3)}</td><td>{p.p_holm < 0.001 ? '<0.1%' : `${(p.p_holm * 100).toFixed(p.p_holm < 0.1 ? 1 : 0)}%`}</td>
            <td>{p.same_sign_as_primary ? '✓' : '✗'}</td><td>{p.backtest?.sharpe?.toFixed(2) ?? '–'}</td><td>{pct(p.backtest?.hit_rate)}</td></tr>))}</tbody></table></Fold>
      {g.skipped?.length > 0 && <p className="rs-sub">Skipped: {g.skipped.map(s => `${s.variable_id} (${s.reason})`).join('; ')}</p>}
    </>
  )
}

function Critique({ lit, crit }) {
  return (
    <>
      <div className="rs-confidence"><span>How confident should we be?</span><div className="rs-meter"><i className={crit.confidence} /></div><b>{crit.confidence}</b>
        <small>never higher than the evidence allows{crit.source !== 'gemini' ? ' · scripted demo text' : ''}</small></div>
      <p className="rs-final">{crit.final_assessment}</p>
      <Fold title="Where this could be wrong"><ul className="rs-crit">{crit.critique.map((c, i) => (
        <li key={i} className={c.severity} style={{ '--i': i }}><b>{c.severity}</b> {plain(c.point)}<span className="rs-ref">{c.grounded_in}</span></li>))}</ul></Fold>
      <Fold title={`Papers behind this (${lit.papers.length})${lit.status !== 'completed' ? ' · ' + (lit.note || lit.status) : ''}`}><div className="rs-papers">{lit.papers.map((p, i) => {
        const a = crit.paper_assessments.find(x => x.paper_index === p.index)
        return (
          <article key={p.index} className="rs-paper" style={{ '--i': i }}>
            <div className="rs-paper-top"><b>[{p.index}]</b> {p.url ? <a href={p.url} target="_blank" rel="noopener noreferrer">{p.title}</a> : p.title}</div>
            <div className="rs-sub">{p.authors ? `${p.authors} · ` : ''}{p.year}{p.venue ? ` · ${p.venue}` : ''}{p.cited_by != null ? ` · ${p.cited_by} citations` : ''}</div>
            {p.abstract && lit.provider === 'curated_reference' && <div className="rs-paper-sum">{p.abstract}</div>}
            {a && <div className={`rs-stance ${a.stance}`}>{a.stance} · relevance {a.relevance}<span>{a.note}</span></div>}
          </article>)
      })}</div></Fold>
      {lit.papers.length > 0 && <p className="rs-sub">{lit.provider === 'curated_reference' ? 'Demo only: real published papers picked by hand as background for the topic. Click a title to open the publisher page.' : `Papers were looked up automatically (${lit.provider}). The AI judges how relevant they are, but it never invents citations.`}</p>}
    </>
  )
}

const EXPECT = { supports: ['good', 'Related research is supportive'], mixed: ['warn', 'Related research is mixed'],
  contradicts: ['bad', 'Related research is sceptical'], insufficient: ['mute', 'Little directly relevant research'] }

function PriorResearch({ h }) {
  const lp = h.literature_prior
  if (!lp) return null
  const papers = h.literature?.papers || []
  const cited = papers.filter(p => lp.cited_papers.includes(p.index))
  const rest = papers.filter(p => !lp.cited_papers.includes(p.index))
  const [cls, text] = EXPECT[lp.expectation] || EXPECT.insufficient
  const link = p => (p.url ? <a href={p.url} target="_blank" rel="noopener noreferrer">{p.title}</a> : p.title)
  return (
    <div className="rs-prior">
      <h3>What related research suggests</h3>
      <div className="rs-chips"><span className={`rs-chip ${cls}`}>{text}</span>
        <span>Realism score<span className="rs-pips">{[1, 2, 3, 4, 5].map(i => <i key={i} className={i <= lp.viability ? 'on' : ''} />)}</span>{lp.viability}/5</span></div>
      <p>{lp.summary}</p>
      {lp.concerns && <p className="rs-sub">Concern: {lp.concerns}</p>}
      {cited.length > 0 && <ul className="rs-list">{cited.map(p => <li key={p.index}>{link(p)} <span className="rs-sub">({p.year}{p.venue ? `, ${p.venue}` : ''})</span></li>)}</ul>}
      {rest.length > 0 && <details className="rs-rejected"><summary>{rest.length} other paper{rest.length === 1 ? '' : 's'} found</summary>
        <ul className="rs-list">{rest.map(p => <li key={p.index}>{link(p)} <span className="rs-sub">({p.year})</span></li>)}</ul></details>}
      <p className="rs-sub">{lp.source === 'gemini' ? 'Summary written by the AI from the abstracts of the papers found. It is not a full literature review and contains no statistics.'
        : lp.source === 'curated' ? 'Demo only: hand-picked real papers and a short written summary. Nothing here is a statistical result.' : `Not assessed by the AI (${lp.source}).`}</p>
    </div>
  )
}

const money = v => (v == null ? '–' : `$${Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 })}`)
const signed = (v, d = 1) => (v == null ? '–' : `${v >= 0 ? '+' : ''}${v.toFixed(d)}%`)
const SIM_VERDICT = { outperformed: ['good', 'Beat the S&P 500'], underperformed: ['bad', 'Lagged the S&P 500'], inconclusive: ['warn', 'Ahead, but could be luck'] }

export function SimResult({ sim }) {
  const [vc, vt] = SIM_VERDICT[sim.vs_benchmark.verdict]
  const S = sim.strategy, B = sim.benchmark_buy_hold, A = sim.asset_buy_hold
  return (
        <div className="rs-simout">
          <div className="rs-chips"><span className={`rs-chip ${vc}`}>{vt}</span>
            <span className="rs-sub">{sim.start} → {sim.end} · {sim.n_days} trading days · benchmark: {sim.benchmark.label} ({sim.benchmark.variable_id})</span></div>
          <div className="rs-versus">
            <div className="rs-vs strat"><div className="rs-vs-label">Strategy</div><div className="rs-vs-val"><Num value={S.final_value} decimals={0} prefix="$" /></div>
              <div className="rs-vs-sub">{signed(S.total_return_pct)} · Sharpe {S.sharpe?.toFixed(2) ?? '–'} · max DD {S.max_drawdown_pct.toFixed(1)}%</div></div>
            <div className="rs-vs bench"><div className="rs-vs-label">Hold {sim.benchmark.label}</div><div className="rs-vs-val"><Num value={B.final_value} decimals={0} prefix="$" /></div>
              <div className="rs-vs-sub">{signed(B.total_return_pct)} · Sharpe {B.sharpe?.toFixed(2) ?? '–'} · max DD {B.max_drawdown_pct.toFixed(1)}%</div></div>
            <div className="rs-vs diff"><div className="rs-vs-label">Difference</div>
              <div className={`rs-vs-val ${sim.vs_benchmark.excess_final_value >= 0 ? 'up' : 'down'}`}>{sim.vs_benchmark.excess_final_value >= 0 ? '+' : '−'}{money(Math.abs(sim.vs_benchmark.excess_final_value))}</div>
              <div className="rs-vs-sub">{signed(sim.vs_benchmark.excess_return_pct)} vs benchmark</div></div>
          </div>
          <LineChart dates={sim.curve.map(r => r.date)} zero={false} label="Strategy versus S&P 500 buy and hold"
            series={[{ name: `Strategy (${money(sim.capital)} start)`, values: sim.curve.map(r => r.strategy), color: 'var(--rs-accent)' },
                     { name: sim.benchmark.label, values: sim.curve.map(r => r.benchmark), color: 'var(--rs-amber)' },
                     { name: `Hold ${A.label}`, values: sim.curve.map(r => r.asset), color: 'var(--rs-muted)' }]} />
          <Fold title="Full comparison table and caveats"><table className="rs-table"><thead><tr><th /><th>Final value</th><th>Return</th><th>CAGR</th><th>Volatility</th><th>Sharpe</th><th>Max drawdown</th></tr></thead>
            <tbody>{[['Strategy', S], [`Hold ${sim.benchmark.label}`, B], [`Hold ${A.label}`, A]].map(([n, m]) => (
              <tr key={n}><td>{n}</td><td>{money(m.final_value)}</td><td>{signed(m.total_return_pct)}</td><td>{m.cagr_pct == null ? '–' : signed(m.cagr_pct)}</td>
                <td>{m.ann_vol_pct.toFixed(1)}%</td><td>{m.sharpe?.toFixed(2) ?? '–'}</td><td>{m.max_drawdown_pct.toFixed(1)}%</td></tr>))}</tbody></table>
          <p className="rs-sub"><b>{vt}.</b> {sim.vs_benchmark.verdict_rule} Mean daily active return CI: {sim.vs_benchmark.active_return_ci_95 ? `[${sim.vs_benchmark.active_return_ci_95.map(v => v.toFixed(3)).join(', ')}]%` : 'n/a'}.
            Beta to benchmark {sim.vs_benchmark.beta?.toFixed(2) ?? '–'}. {sim.period_note}</p>
          <div className="rs-note"><b>Current signal ({sim.current_signal.as_of}):</b> {sim.current_signal.meaning}. {sim.current_signal.note}</div>
          <ul className="rs-list rs-sub">{sim.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul></Fold>
        </div>  )
}

/** Implement the frozen strategy as a paper trade and compare it with holding the S&P 500. */
function SimulatePanel({ runId, h }) {
  const [capital, setCapital] = useState(10000)
  const [period, setPeriod] = useState('holdout')
  const [cost, setCost] = useState(2)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [sim, setSim] = useState(null)
  if (!h.backtest?.applicable) return <div className="rs-note">No strategy to simulate. {h.backtest?.reason}</div>
  async function go() {
    setBusy(true); setErr(''); setSim(null)
    try { setSim(await simulateStrategy(runId, { hypothesis_id: h.hypothesis_id, capital: Number(capital), period, cost_bps: Number(cost) })) }
    catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  return (
    <div className="rs-sim">
      <div className="rs-simform">
        <label>Starting capital <input type="number" min="100" max="10000000" step="500" value={capital} onChange={e => setCapital(e.target.value)} /></label>
        <label>Period <select value={period} onChange={e => setPeriod(e.target.value)}>
          <option value="holdout">Out-of-sample holdout (recommended)</option><option value="full">Full sample</option><option value="development">In-sample only</option></select></label>
        <label>Cost (bps) <input type="number" min="0" max="50" step="0.5" value={cost} onChange={e => setCost(e.target.value)} style={{ width: 70 }} /></label>
        <button className="rs-launch-btn" onClick={go} disabled={busy}>{busy ? 'Simulating…' : 'Run simulation'}</button>
      </div>
      <p className="rs-sub">Paper trade only: replays the frozen rule on the stored data. No orders are placed anywhere.</p>
      {err && <p className="rs-error" role="alert">{err}</p>}
      {sim && <SimResult sim={sim} />}
    </div>
  )
}

const dirWord = (n, c) => {
  const up = c.r >= 0
  return up ? `a rise in ${n.predictor.label} tends to be followed by a rise in ${n.outcome.label}`
    : `a rise in ${n.predictor.label} tends to be followed by a fall in ${n.outcome.label}`
}

/** The one-minute story: one sentence, five checkpoints. */
function Headline({ h }) {
  const n = h.normalized, c = h.correlation, v = h.validation, ca = h.causality, bt = h.backtest, g = h.generalisation
  const stopped = h.status === 'no_correlation'
  let sentence, tone
  if (stopped) { sentence = `No reliable link: ${n.predictor.label} does not help predict ${n.outcome.label}. We stopped there.`; tone = 'bad' }
  else if (v?.reliability_verdict === 'supported') { sentence = `A link that held up: ${dirWord(n, c)}, and this still worked on data we locked away.`; tone = 'good' }
  else if (v?.reliability_verdict === 'rejected') { sentence = `It looked linked, but it did not hold up on fresh data: ${dirWord(n, c)} in the past, but not when tested.`; tone = 'bad' }
  else if (v) { sentence = `Mixed evidence: ${dirWord(n, c)}, but the tests were not clear enough to trust.`; tone = 'warn' }
  else { sentence = `Testing: ${n.question}`; tone = 'mute' }
  const grade = st => (st?.pct == null ? null : st.pct >= 60 ? true : st.pct < 40 ? false : null)
  const sTxt = st => (st?.pct == null ? '' : `${st.label}, ${st.pct}/100`)
  const cap = t => (t ? t.charAt(0).toUpperCase() + t.slice(1) : t)
  const band = pct => (pct == null ? null : { pct, label: pct >= 80 ? 'strong' : pct >= 60 ? 'moderate' : pct >= 40 ? 'mixed' : pct >= 20 ? 'weak' : 'very weak' })
  const step = (label, ok, detail, st) => ({ label, ok, detail, st })
  const lp = h.literature_prior
  const EXPECT_TXT = { supports: 'Papers lean supportive', mixed: 'Papers are mixed', contradicts: 'Papers are sceptical', insufficient: 'Little directly relevant research' }
  const steps = [
    step('Link in the data', c.strength ? grade(c.strength) : c.association_verdict === 'detected', `${c.r >= 0 ? 'Rises together' : 'Moves oppositely'} (r = ${c.r.toFixed(2)})`, c.strength),
    ...(stopped ? [] : [
      step('Cause and effect', ca ? (ca.causal_verdict === 'contradicted' ? false : grade(ca.strength)) : null,
        ca ? `${ca.strength ? `${ca.strength.checks_passed} of ${ca.strength.checks_run} checks agree` : ''}${ca.causal_verdict === 'contradicted' ? ' \u00b7 contradicted' : ' \u00b7 not proven'}` : '...', ca?.strength),
      step('Works on unseen data', v ? grade(v.strength) : null,
        v ? `${v.strength ? `${v.strength.checks_passed} of ${v.strength.checks_run}` : `${v.tests.filter(t => t.result === 'pass').length} of ${v.tests.length}`} checks passed` : '...', v?.strength),
      ...(bt?.applicable ? [step('Profitable as a strategy', bt.verdict === 'pass' ? true : bt.verdict === 'fail' ? false : null,
        `${signed(bt.holdout.total_return_pct, 0)} vs ${signed(bt.holdout.buy_hold_total_return_pct, 0)} just holding it`,
        band({ pass: 100, inconclusive: 50, fail: 0 }[bt.verdict]))] : []),
      ...(g && g.status === 'completed' ? [step('Works on similar assets', g.verdict === 'generalises' ? true : g.verdict === 'partial' ? null : false,
        `${Math.round((g.share_same_sign ?? 0) * 100)}% move the same way`,
        band({ generalises: 100, partial: 60, does_not_generalise: 0 }[g.verdict]))] : []),
    ]),
    ...(lp ? [step('What other research says', lp.expectation === 'supports' ? true : lp.expectation === 'contradicts' ? false : null,
      EXPECT_TXT[lp.expectation] || EXPECT_TXT.insufficient, band(lp.viability * 20))] : []),
  ]
  return (
    <div className={`rs-headline ${tone}`}>
      <div className="rs-hl-kicker">The one-minute answer</div>
      {h.strength?.pct != null && (
        <div className="rs-overall-strength"><span>Overall strength of evidence</span>
          <div className="rs-meter big"><i style={{ width: `${h.strength.pct}%` }} className={h.strength.pct >= 60 ? 'high' : h.strength.pct >= 40 ? 'moderate' : 'low'} /></div>
          <b>{cap(h.strength.label)} · {h.strength.pct}/100</b></div>)}
      <p className="rs-hl-sentence">{sentence}</p>
      <ol className="rs-ladder">
        {steps.map((st, i) => (
          <li key={st.label} className={st.ok === true ? 'ok' : st.ok === false ? 'no' : 'maybe'} style={{ '--i': i }}>
            <div className="rs-lad-head"><span className="rs-lad-icon">{st.ok === true ? '\u2713' : st.ok === false ? '\u2717' : '?'}</span><b>{st.label}</b></div>
            <small>{st.detail}</small>
            {st.st?.pct != null && (
              <div className={`rs-lad-meter ${st.ok === true ? 'ok' : st.ok === false ? 'no' : 'maybe'}`} title={`${cap(st.st.label)} (${st.st.pct}/100)`}>
                <i style={{ width: `${Math.max(st.st.pct, 4)}%` }} /><span>{cap(st.st.label)} {st.st.pct}</span>
              </div>)}
          </li>))}
      </ol>
      {!stopped && <p className="rs-hl-note">A pattern that holds up is useful, but it is not proof that one thing causes the other.</p>}
    </div>
  )
}

function HypothesisView({ h, viz, runId }) {
  const c = h.correlation, ca = h.causality, v = h.validation, n = h.normalized
  if (!c || !n) return <div className="rs-note">This hypothesis has not produced results yet.</div>
  const passed = v?.tests ? v.tests.filter(t => t.result === 'pass').length : 0
  const vm = viz?.validation_metrics || {}
  const stopped = h.status === 'no_correlation'
  return (
    <div className="rs-hyp">
      <div className="rs-idea">
        <h2>{n.question}</h2>
        <div className="rs-flow"><span>{n.predictor.label}</span><em>{tName(n.predictor.transform)}</em><b>→ {lagText(n.lag_days, n)} →</b><span>{n.outcome.label}</span><em>{tName(n.outcome.transform)}</em></div>
        <p><b>Why it might work (the AI’s idea):</b> {n.mechanism}</p>
        <p><b>If there were no link:</b> {n.null_hypothesis}</p>
      </div>
      <Headline h={h} />
      <PriorResearch h={h} />
      {stopped && <div className="rs-stopbanner" role="status"><b>No correlation found.</b> This is one of the &ldquo;Initial hypotheses with no correlation&rdquo;: testing stopped here, so causal checks, validation, backtest and peer tests were not run.</div>}

      <Section n="1" title="Is there a link?" sub={`We compare ${n.predictor.label} on one day with ${n.outcome.label} ${n.lag_days ? `${n.lag_days} day(s) later` : fwdDays(n) ? `over the following ${fwdDays(n)} days (it starts the day after, so there is no look-ahead)` : 'over the following days'}. Each dot is one day.`}>
        <ScatterChart points={viz?.scatter} fit={viz?.scatter_fit} r={c.r} ci={c.ci_95} n={c.n} xlabel={axisLabel(n.predictor)} ylabel={axisLabel(n.outcome)} />
        <p className="rs-sub"><b>In plain terms:</b> a {linkStrength(c.r)} link (r = {c.r.toFixed(2)}, where 0 means none and 1 means perfect). Chance of seeing this by luck alone: {pPlain(c.p_value_hac)}. Blue dots are older data, amber dots are the final quarter that was kept hidden.</p>
        <details className="rs-tech"><summary>Technical details</summary><p className="rs-sub">Robust (Newey-West) p = {c.p_value_hac.toExponential(2)} · Spearman rho = {c.spearman_rho.toFixed(3)} · same-day r = {c.same_time_r.toFixed(3)} · {plain(c.verdict_rule, n)}</p></details>
        <Fold open title="Does the link come and go, and which time delay works best?"><div className="rs-two">
          <div><h4>Does the link come and go? (60-day window)</h4>
            <LineChart dates={(viz?.rolling_r || []).map(r => r.date)} label="Rolling correlation"
              series={[{ name: 'Rolling correlation', values: (viz?.rolling_r || []).map(r => r.r), color: 'var(--rs-blue)' }]}
              region={vm.split ? { from: vm.split.holdout[0], label: 'Final quarter' } : null} /></div>
          <div><h4>Which time delay works best? (★ = clearly not luck)</h4>
            <SignedBars highlight={c.primary_lag} rows={c.lag_family.map(l => ({ key: l.lag, label: lagText(l.lag, n), value: l.r, sig: l.p_holm < 0.05 }))} /></div>
        </div></Fold>
      </Section>

      {!stopped && (<>
      <Section n="2" title="Causation or coincidence?" sub={`A link is not the same as cause and effect. These checks ask whether ${n.predictor.label} really comes first and whether something else could explain the pattern.`}>
        {ca ? (<>
        <div className="rs-verdictbox"><Chip v={ca.causal_verdict} /><p>{causalStory(ca.causal_verdict, n)}</p>
          <details className="rs-tech"><summary>Technical note</summary><p className="rs-sub">{plain(ca.reasoning, n)}</p><p className="rs-sub">{plain(ca.verdict_rule, n)}</p></details></div>
        <Fold open title="The checks and the cause-and-effect map"><div className="rs-two">
          <div><h4>The checks</h4><TestList tests={ca.tests} n={n} /></div>
          <div><h4>Map of who might influence whom</h4><CausalGraph graph={viz?.causal_graph} /></div>
        </div></Fold>
        <Fold title="Other explanations and hidden drivers"><div className="rs-two">
          <div><h4>Other things that could explain it</h4><ul className="rs-list">{ca.alternative_explanations.map((a, i) => <li key={i}>{plain(a, n)}</li>)}</ul></div>
          <div><h4>Hidden drivers that could move both</h4><ul className="rs-list">{ca.possible_confounders.map((a, i) => <li key={i}><b>{nodeName(a.variable_id, viz?.causal_graph)}{String(a.variable_id).startsWith('latent:') ? ' (cannot be measured)' : ''}</b>: {plain(a.rationale, n)}</li>)}</ul></div>
        </div></Fold>
        </>) : <Waiting what="causal checks" />}
      </Section>

      <Section n="3" title="Does it still work on data it has not seen?" sub="We find the link using the oldest half of the data, re-check it on the next quarter, and test it once on the final quarter, which stays locked away until the end.">
        {v ? (<>
        <div className="rs-gauges">
          <Gauge value={passed} max={v.tests.length} label="Checks passed" sub={`of ${v.tests.length}`} color="var(--rs-green)" fmtv={x => Math.round(x)} />
          <Gauge value={h.backtest?.holdout?.hit_rate} max={1} label="Days called right on locked-away data" sub="hit rate" color="var(--rs-accent)" fmtv={x => `${(x * 100).toFixed(0)}%`} />
          <Gauge value={vm.r_by_split ? Math.abs(vm.r_by_split.holdout) : null} max={0.5} label="Link strength on locked-away data" sub="r (0 to 1)" color="var(--rs-blue)" fmtv={x => x.toFixed(2)} />
        </div>
        <div className="rs-verdictbox"><Chip v={v.reliability_verdict} /><p>{plain(v.reasoning, n)}</p>
          <details className="rs-tech"><summary>Technical note</summary><p className="rs-sub">{plain(v.claim_tested, n)} {plain(v.verdict_rule, n)}</p></details></div>
        <Fold open title={`All ${v.tests.length} validation checks`}><TestList tests={v.tests} n={n} /></Fold>
        <Fold open title="What if you had traded it?"><Backtest bt={h.backtest} curve={viz?.backtest_equity || []} n={n} /></Fold>
        </>) : <Waiting what="validation" />}
      </Section>

      <Section n="4" title="Does it work on similar assets?" sub="If it is real, the same idea should work on similar assets. The comparison assets are fixed by the program, not chosen by the AI.">
        {h.generalisation ? <Peers g={h.generalisation} /> : <Waiting what="peer tests" />}
      </Section>

      <Section n="5" title="What could be wrong, and what do the papers say?" sub="The AI argues against its own result, using the real research papers listed below.">
        {h.critique && h.literature ? <Critique lit={h.literature} crit={h.critique} /> : <div className="rs-note">Critique runs after all hypotheses finish.</div>}
      </Section>

      {h.backtest?.applicable && h.backtest.verdict !== 'fail' && (
      <Section n="6" title="Try it with your own money (pretend)" sub="Run the frozen rule with your own starting capital and compare it with simply holding the S&P 500.">
        {h.critique ? <SimulatePanel runId={runId} h={h} /> : <div className="rs-note">Available once the research has finished.</div>}
      </Section>)}
      </>)}
    </div>
  )
}

function HypCard({ h, i, sel, setSel }) {
  return (
    <button className={`rs-card ${sel === h.hypothesis_id ? 'sel' : ''} ${h.status === 'no_correlation' ? 'stopped' : h.status}`} style={{ '--i': i }}
            onClick={() => h.correlation && setSel(h.hypothesis_id)} disabled={!h.correlation}>
      <b>{h.hypothesis_id}</b>
      <span>{h.normalized?.question || h.proposal?.question}</span>
      {h.status === 'no_correlation' && <small className="rs-card-r">r = {h.correlation.r.toFixed(3)} · HAC p = {h.correlation.p_value_hac.toFixed(2)} · stopped</small>}
      {h.verdict && h.status !== 'no_correlation' && <div className="rs-chips"><Chip v={h.verdict.association} /><Chip v={h.verdict.reliability} label="Unseen data" /></div>}
      {!h.verdict && h.status === 'running' && <small className="rs-pulse">{h.stages.find(s => s.status === 'running')?.message || 'working'}…</small>}
    </button>
  )
}

const PHASE_LABELS = { hypothesis_generation: 'Generating ideas', literature_review: 'Checking related research', preregistered: 'Pre-registering', correlation: 'Measuring correlation', causality: 'Probing causation',
  validation: 'Validating', backtest: 'Backtesting', generalisation: 'Testing peers', family_correction: 'Correcting for multiple tests', critique: 'Literature & self-critique', complete: 'Done' }

export function PipelineStrip({ report }) {
  const keys = Object.keys(PHASE_LABELS)
  const at = keys.indexOf(report.current_phase)
  return (
    <div className="rs-pipe">
      {keys.map((k, i) => (
        <div key={k} className={`rs-pipe-step ${i < at || report.status === 'completed' ? 'done' : i === at && report.status === 'running' ? 'active' : ''}`}>
          <i />{PHASE_LABELS[k]}
        </div>))}
    </div>
  )
}

class Boundary extends React.Component {
  state = { err: null }
  static getDerivedStateFromError(err) { return { err } }
  render() {
    return this.state.err
      ? <div className="rs-note" role="alert">The results view hit a display error ({String(this.state.err.message || this.state.err)}). The research data itself is intact; fetch it from /api/research/runs.</div>
      : this.props.children
  }
}

export default function ResearchReport(props) {
  return <Boundary key={props.report.run_id || props.report.id}><ReportBody {...props} /></Boundary>
}

function ReportBody({ report }) {
  const hs = report.hypotheses || []
  const firstReady = hs.find(h => h.correlation)?.hypothesis_id
  const [picked, setSel] = useState(null)
  const sel = picked || firstReady
  const cur = hs.find(h => h.hypothesis_id === sel)
  const tested = hs.filter(h => !['rejected', 'no_correlation'].includes(h.status))
  const stopped = hs.filter(h => h.status === 'no_correlation')
  const rejected = hs.filter(h => h.status === 'rejected')
  const running = report.status === 'running' || report.status === 'queued'
  return (
    <div className="rs-root">
      {running && <PipelineStrip report={report} />}
      {report.status === 'completed' && <PipelineStrip report={report} />}
      {report.status === 'completed' && (
        <div className="rs-overall">
          <h2>{report.summary.counts.supported} of {report.multiple_testing.attempted} ideas held up under testing{report.summary.counts.no_correlation ? ` · ${report.summary.counts.no_correlation} showed no link` : ''}</h2>
          <p>{report.summary.overall}</p>
        </div>)}
      <div className="rs-cards">
        {tested.map((h, i) => <HypCard key={h.hypothesis_id} h={h} i={i} sel={sel} setSel={setSel} />)}
      </div>
      {stopped.length > 0 && (
        <div className="rs-group">
          <h3>Initial hypotheses with no correlation <small>(tested, nothing found, stopped early)</small></h3>
          <div className="rs-cards">{stopped.map((h, i) => <HypCard key={h.hypothesis_id} h={h} i={i} sel={sel} setSel={setSel} />)}</div>
        </div>)}
      {rejected.length > 0 && (
        <details className="rs-rejected"><summary>{rejected.length} other idea{rejected.length === 1 ? '' : 's'} the AI proposed but we did not test (click to see why)</summary>
          <ul className="rs-list">{rejected.map(h => <li key={h.hypothesis_id}><b>{h.hypothesis_id}</b> {h.normalized?.question || h.proposal?.question}
            <div className="rs-sub">{h.eligibility.reasons.join('; ')}</div></li>)}</ul></details>)}
      {cur && <HypothesisView key={cur.hypothesis_id} h={cur} runId={report.run_id || report.id} viz={report.visualization_data?.[cur.hypothesis_id]} />}
      {report.warnings?.map((w, i) => <p key={i} className="rs-warn">⚠ {w}</p>)}
    </div>
  )
}
