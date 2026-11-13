import React, { useEffect, useRef } from 'react'

const W = 10, GAP = 7, STEP = W + GAP

function makeCandle(prev) {
  const o = prev
  const move = (Math.random() - 0.48) * 14
  const c = o + move
  return { o, c, h: Math.max(o, c) + Math.random() * 7, l: Math.min(o, c) - Math.random() * 7 }
}

/** Slow candlestick chart drifting behind the Research tab, under a dark tint so it never competes with the report. */
export default function CandleBackground() {
  const ref = useRef(null)
  useEffect(() => {
    const cv = ref.current
    if (!cv) return undefined
    const ctx = cv.getContext('2d')
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let w = 0, h = 0, raf = 0, last = 0, offset = 0
    let candles = []
    let price = 0

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      w = window.innerWidth; h = window.innerHeight
      cv.width = w * dpr; cv.height = h * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      const need = Math.ceil(w / STEP) + 6
      if (!candles.length) price = 0
      while (candles.length < need) { const c = makeCandle(candles.length ? candles[candles.length - 1].c : 0); candles.push(c) }
    }

    const draw = () => {
      ctx.clearRect(0, 0, w, h)
      // vertical scale: keep the visible window centred on the middle of the screen
      const vis = candles.slice(0, Math.ceil(w / STEP) + 2)
      const hi = Math.max(...vis.map(c => c.h)), lo = Math.min(...vis.map(c => c.l))
      const span = Math.max(hi - lo, 40)
      const y = v => h * 0.18 + (1 - (v - lo) / span) * h * 0.64
      vis.forEach((c, i) => {
        const x = i * STEP - offset
        const up = c.c >= c.o
        ctx.strokeStyle = ctx.fillStyle = up ? 'rgba(74,196,128,0.9)' : 'rgba(239,106,106,0.9)'
        ctx.lineWidth = 1.5
        ctx.beginPath(); ctx.moveTo(x + W / 2, y(c.h)); ctx.lineTo(x + W / 2, y(c.l)); ctx.stroke()
        const top = y(Math.max(c.o, c.c)), bot = y(Math.min(c.o, c.c))
        ctx.fillRect(x, top, W, Math.max(2, bot - top))
      })
    }

    const tick = ts => {
      if (ts - last > 33) {                 // ~30 fps is plenty for a slow drift
        last = ts
        offset += 0.35
        if (offset >= STEP) { offset -= STEP; candles.shift(); candles.push(makeCandle(candles[candles.length - 1].c)) }
        draw()
      }
      raf = requestAnimationFrame(tick)
    }

    resize()
    draw()
    window.addEventListener('resize', resize)
    if (!reduce) raf = requestAnimationFrame(tick)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [])

  return (
    <div className="rs-bg" aria-hidden="true">
      <canvas ref={ref} />
      <div className="rs-bg-tint" />
    </div>
  )
}

/** A single tick or cross that fades in and out when a run finishes. */
export function ResultStamp({ kind }) {
  if (!kind) return null
  return <div className={`rs-stamp ${kind}`} aria-hidden="true">{kind === 'tick' ? '✓' : '✗'}</div>
}
