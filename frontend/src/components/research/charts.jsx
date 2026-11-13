import React from 'react'
import { useCountUp, useInView } from './anim'

const W = 640, H = 300, PAD = { l: 48, r: 14, t: 14, b: 42 }
const lin = (d0, d1, r0, r1) => v => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0)
const nice = (min, max) => { const p = (max - min) * 0.06 || 1; return [min - p, max + p] }
const fmt = (v, d = 2) => (v == null ? '–' : Number(v).toFixed(d))

export function Num({ value, decimals = 2, suffix = '', prefix = '' }) {
  const v = useCountUp(value, { decimals })
  return <span>{value == null ? '–' : `${prefix}${v.toFixed(decimals)}${suffix}`}</span>
}

function Axes({ sx, sy, xt, yt, xlab, ylab, zero }) {
  return (
    <g className="rs-axes">
      {yt.map(t => <g key={t}><line x1={PAD.l} x2={W - PAD.r} y1={sy(t)} y2={sy(t)} className="rs-grid" /><text x={PAD.l - 6} y={sy(t) + 4} textAnchor="end">{fmt(t, Math.abs(t) < 10 ? 1 : 0)}</text></g>)}
      {xt.map(t => <text key={t} x={sx(t)} y={H - PAD.b + 15} textAnchor="middle">{typeof t === 'number' ? fmt(t, 1) : t}</text>)}
      {zero && sy(0) >= PAD.t && sy(0) <= H - PAD.b && <line x1={PAD.l} x2={W - PAD.r} y1={sy(0)} y2={sy(0)} className="rs-zero" />}
      {xlab && <text x={W / 2} y={H - 4} textAnchor="middle" className="rs-axlab">{xlab}</text>}
      {ylab && <text transform={`translate(11 ${(H - PAD.b + PAD.t) / 2}) rotate(-90)`} textAnchor="middle" className="rs-axlab"><title>{ylab}</title>{ylab.length > 40 ? `${ylab.slice(0, 38)}...` : ylab}</text>}
    </g>
  )
}

/** Correlation scatter: points pop in, the fit line draws, big r counter. */
export function ScatterChart({ points, fit, r, ci, xlabel, ylabel, n }) {
  const [ref, seen] = useInView()
  if (!points?.length) return <div ref={ref} className="rs-empty">No data</div>
  const xs = points.map(p => p.x), ys = points.map(p => p.y)
  const [x0, x1] = nice(Math.min(...xs), Math.max(...xs)), [y0, y1] = nice(Math.min(...ys), Math.max(...ys))
  const sx = lin(x0, x1, PAD.l, W - PAD.r), sy = lin(y0, y1, H - PAD.b, PAD.t)
  const step = Math.ceil(points.length / 700)
  const shown = points.filter((_, i) => i % step === 0)
  const xt = [x0 + (x1 - x0) * 0.1, (x0 + x1) / 2, x1 - (x1 - x0) * 0.1], yt = [y0 + (y1 - y0) * 0.1, (y0 + y1) / 2, y1 - (y1 - y0) * 0.1]
  return (
    <div ref={ref} className="rs-chartwrap">
      <div className="rs-bigstat">
        <div className="rs-bigstat-label">Correlation</div>
        <div className="rs-bigstat-val"><Num value={seen ? r : 0} decimals={3} /></div>
        {ci && <div className="rs-bigstat-sub">95% CI [{fmt(ci[0], 3)}, {fmt(ci[1], 3)}] · n = {n}</div>}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="rs-svg" role="img" aria-label={`Scatter of ${xlabel} against ${ylabel}, r = ${fmt(r, 3)}`}>
        <Axes sx={sx} sy={sy} xt={xt} yt={yt} xlab={xlabel} ylab={ylabel} zero />
        {seen && shown.map((p, i) => (
          <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={3} className={`rs-dot ${p.sample === 'holdout' ? 'hold' : 'dev'}`}
                  style={{ animationDelay: `${Math.min(i * 2, 900)}ms` }} />
        ))}
        {seen && fit && (
          <line x1={sx(fit.x_min)} y1={sy(fit.intercept + fit.slope * fit.x_min)} x2={sx(fit.x_max)} y2={sy(fit.intercept + fit.slope * fit.x_max)}
                className="rs-fit" pathLength="1" />
        )}
      </svg>
      <div className="rs-legend"><span><i className="dev" />Older data</span><span><i className="hold" />Final quarter (hidden until the end)</span><span><i className="fit" />Best-fit line</span></div>
    </div>
  )
}

/** Generic multi-line chart with draw-in animation and an optional shaded region. */
export function LineChart({ series, dates, region, yfmt = 2, zero = true, label }) {
  const [ref, seen] = useInView()
  const all = series.flatMap(s => s.values.filter(v => v != null))
  if (!all.length) return <div ref={ref} className="rs-empty">No data</div>
  const [y0, y1] = nice(Math.min(...all, zero ? 0 : Infinity), Math.max(...all, zero ? 0 : -Infinity))
  const n = dates.length
  const sx = lin(0, n - 1, PAD.l, W - PAD.r), sy = lin(y0, y1, H - PAD.b, PAD.t)
  const path = vals => {
    let d = '', pen = false
    vals.forEach((v, i) => { if (v == null) { pen = false; return } d += `${pen ? 'L' : 'M'}${sx(i).toFixed(1)} ${sy(v).toFixed(1)}`; pen = true })
    return d
  }
  const ticks = [0, Math.floor(n / 3), Math.floor((2 * n) / 3), n - 1].map(i => i)
  const yt = [y0 + (y1 - y0) * 0.1, (y0 + y1) / 2, y1 - (y1 - y0) * 0.1]
  const regionX = region ? dates.findIndex(d => d >= region.from) : -1
  return (
    <div ref={ref} className="rs-chartwrap">
      <svg viewBox={`0 0 ${W} ${H}`} className="rs-svg" role="img" aria-label={label}>
        {regionX >= 0 && <g><rect x={sx(regionX)} y={PAD.t} width={W - PAD.r - sx(regionX)} height={H - PAD.b - PAD.t} className="rs-region" />
          <text x={sx(regionX) + 6} y={PAD.t + 12} className="rs-regionlab">{region.label}</text></g>}
        <Axes sx={sx} sy={sy} xt={[]} yt={yt} zero={zero} />
        {ticks.map(i => <text key={i} x={sx(i)} y={H - PAD.b + 15} textAnchor={i === n - 1 ? "end" : "middle"} className="rs-axes-t">{dates[i]?.slice(0, 7)}</text>)}
        {seen && series.map(s => <path key={s.name} d={path(s.values)} className={`rs-line ${s.cls || ''}`} pathLength="1" style={{ stroke: s.color }} />)}
      </svg>
      <div className="rs-legend">{series.map(s => <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>)}</div>
    </div>
  )
}

/** Animated horizontal gauge ring. */
export function Gauge({ value, max = 1, label, sub, color = 'var(--rs-accent)', fmtv }) {
  const [ref, seen] = useInView()
  const frac = value == null ? 0 : Math.max(0, Math.min(1, value / max))
  const R = 46, C = 2 * Math.PI * R
  const v = useCountUp(seen && value != null ? value : 0, { decimals: max > 1 ? 0 : 3 })
  return (
    <div ref={ref} className="rs-gauge">
      <svg viewBox="0 0 120 120">
        <circle cx="60" cy="60" r={R} className="rs-gauge-bg" />
        <circle cx="60" cy="60" r={R} className="rs-gauge-fg" style={{ stroke: color, strokeDasharray: C, strokeDashoffset: seen ? C * (1 - frac) : C }}
                transform="rotate(-90 60 60)" />
        <text x="60" y="58" textAnchor="middle" className="rs-gauge-val">{value == null ? '–' : fmtv ? fmtv(v) : v}</text>
        <text x="60" y="76" textAnchor="middle" className="rs-gauge-sub">{sub}</text>
      </svg>
      <div className="rs-gauge-label">{label}</div>
    </div>
  )
}

/** Signed horizontal bars (lag profile / peer correlations). */
export function SignedBars({ rows, highlight }) {
  const [ref, seen] = useInView()
  const m = Math.max(0.05, ...rows.map(r => Math.abs(r.value)))
  return (
    <div ref={ref} className="rs-bars">
      {rows.map((r, i) => (
        <div className={`rs-barrow ${highlight === r.key ? 'hl' : ''}`} key={r.key} style={{ '--i': i }}>
          <div className="rs-barlabel" title={r.title}>{r.label}</div>
          <div className="rs-bartrack">
            <div className="rs-barmid" />
            <div className={`rs-barfill ${r.value >= 0 ? 'pos' : 'neg'} ${r.sig ? 'sig' : ''}`}
                 style={{ width: seen ? `${(Math.abs(r.value) / m) * 50}%` : '0%', [r.value >= 0 ? 'left' : 'right']: '50%', transitionDelay: `${i * 70}ms` }} />
          </div>
          <div className="rs-barval">{fmt(r.value, 3)}{r.sig ? ' ★' : ''}</div>
        </div>
      ))}
    </div>
  )
}

/** Causal DAG: predictor left, outcome right, observed controls above, unobserved drivers below. */
const STATUS_COLOR = { consistent_with_evidence: 'green', contradicted: 'red', supported_by_test: 'red', not_supported: 'muted',
  not_established: 'amber', hypothesised: 'amber', proposed_untested: 'blue' }
const STATUS_RANK = { supported_by_test: 5, contradicted: 5, consistent_with_evidence: 4, not_established: 3, hypothesised: 2, proposed_untested: 1, not_supported: 0 }

function boundary(c, toward, hw, hh, gap = 5) {
  const dx = toward[0] - c[0], dy = toward[1] - c[1]
  const t = Math.min(hw / (Math.abs(dx) || 1e-6), hh / (Math.abs(dy) || 1e-6))
  const L = Math.hypot(dx, dy) || 1
  return [c[0] + dx * t + (dx / L) * gap, c[1] + dy * t + (dy / L) * gap]
}

const short = (id, n = 17) => { const t = id.replace('latent:', '').replace(/_/g, ' '); return t.length > n ? `${t.slice(0, n - 1)}…` : t }

export function CausalGraph({ graph }) {
  const [ref, seen] = useInView()
  if (!graph?.nodes?.length) return <div className="rs-empty">No graph</div>
  const GW = 680, GH = 300, BH = 38
  const pred = graph.nodes.find(n => n.role === 'predictor'), out = graph.nodes.find(n => n.role === 'outcome')
  if (!pred || !out) return <div className="rs-empty">No graph</div>
  const rest = graph.nodes.filter(n => n !== pred && n !== out)
  const top = rest.filter(n => n.observed), bottom = rest.filter(n => !n.observed)
  const widest = Math.max(top.length, bottom.length, 1)
  const BW = Math.max(84, Math.min(128, (GW - 2 * 148) / widest - 10))     // shrink boxes so a crowded row never overlaps
  const maxChars = BW < 110 ? 12 : 17
  const pos = { [pred.id]: [BW / 2 + 14, GH / 2], [out.id]: [GW - BW / 2 - 14, GH / 2] }
  const row = (list, y) => list.forEach((n, i) => { pos[n.id] = [BW + 20 + ((GW - 2 * (BW + 20)) * (i + 0.5)) / list.length, y] })
  row(top, 48); row(bottom, GH - 48)

  // one edge per ordered pair; keep the most informative status
  const best = new Map()
  graph.edges.forEach(e => {
    if (!pos[e.source] || !pos[e.target] || e.source === e.target) return
    const k = `${e.source}>${e.target}`
    if (!best.has(k) || (STATUS_RANK[e.status] ?? 0) > (STATUS_RANK[best.get(k).status] ?? 0)) best.set(k, e)
  })
  const edges = [...best.values()]
  // the x->y edge is also the only one allowed to be drawn when its reverse exists: drop y->x unless it is informative
  const filtered = edges.filter(e => !(best.has(`${e.target}>${e.source}`) && e.status === 'not_supported' && e.source === out.id))

  return (
    <div ref={ref} className="rs-chartwrap">
      <svg viewBox={`0 0 ${GW} ${GH}`} className="rs-svg" role="img" aria-label="Causal graph">
        <defs>{['green', 'red', 'amber', 'blue', 'muted'].map(c => (
          <marker key={c} id={`rs-ah-${c}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0 1L9 5L0 9z" fill={`var(--rs-${c})`} /></marker>))}</defs>
        {filtered.map((e, i) => {
          const a = pos[e.source], b = pos[e.target]
          const p1 = boundary(a, b, BW / 2, BH / 2, 2), p2 = boundary(b, a, BW / 2, BH / 2, 7)
          const col = STATUS_COLOR[e.status] || 'muted'
          const proposed = e.kind === 'model_proposed'
          return (
            <path key={`${e.source}>${e.target}`} d={`M${p1[0].toFixed(1)} ${p1[1].toFixed(1)} L${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`}
                  className={`rs-edge ${proposed ? 'fade' : 'draw'} ${seen ? 'on' : ''}`} pathLength={proposed ? undefined : 1}
                  style={{ stroke: `var(--rs-${col})`, animationDelay: `${350 + i * 150}ms` }} markerEnd={`url(#rs-ah-${col})`}>
              <title>{`${e.source} → ${e.target}: ${e.status.replace(/_/g, ' ')}`}</title>
            </path>
          )
        })}
        {graph.nodes.map((n, i) => {
          const p = pos[n.id]
          if (!p) return null
          return (
            <g key={n.id} transform={`translate(${p[0]} ${p[1]})`}>
              <g className={`rs-node ${seen ? 'on' : ''}`} style={{ animationDelay: `${i * 110}ms` }}>
                <rect x={-BW / 2} y={-BH / 2} width={BW} height={BH} rx="8" className={`${n.role} ${n.observed ? '' : 'latent'}`} />
                <text y="4" textAnchor="middle">{short(n.id, maxChars)}<title>{n.label || n.id}</title></text>
                <text y={BH / 2 + 13} textAnchor="middle" className="rs-node-sub">{n.role}{n.observed ? '' : ' · unobserved'}</text>
              </g>
            </g>
          )
        })}
      </svg>
      <div className="rs-legend">
        <span><i style={{ background: 'var(--rs-green)' }} /> consistent with evidence</span><span><i style={{ background: 'var(--rs-red)' }} /> contradicted / confounded</span>
        <span><i style={{ background: 'var(--rs-amber)' }} /> not established</span><span><i style={{ background: 'var(--rs-blue)' }} /> AI-proposed, untested</span>
        <span><i style={{ background: 'var(--rs-muted)' }} /> not supported</span>
      </div>
    </div>
  )
}
