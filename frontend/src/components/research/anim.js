import { useEffect, useRef, useState } from 'react'

export const reducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches

/** Animated number: eases from 0 (or the previous value) to `value`. */
export function useCountUp(value, { duration = 1100, decimals = 2 } = {}) {
  const [v, setV] = useState(0)
  const from = useRef(0)
  useEffect(() => {
    if (value == null || Number.isNaN(value)) return undefined
    if (reducedMotion()) { setV(value); return undefined }
    const start = performance.now()
    const a = from.current
    let raf
    const tick = now => {
      const t = Math.min(1, (now - start) / duration)
      const e = 1 - Math.pow(1 - t, 3)
      setV(a + (value - a) * e)
      if (t < 1) raf = requestAnimationFrame(tick)
      else from.current = value
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [value, duration])
  return value == null ? null : Number(v.toFixed(decimals))
}

/** true once the element has scrolled into view (so charts animate when seen, not off-screen).
 *  Returns a callback ref so it still works when the element mounts later or is swapped. */
export function useInView(rootMargin = '0px 0px -10% 0px') {
  const [el, setEl] = useState(null)
  const [seen, setSeen] = useState(false)
  useEffect(() => {
    if (!el || seen) return undefined
    if (typeof IntersectionObserver === 'undefined' || reducedMotion()) { setSeen(true); return undefined }
    const io = new IntersectionObserver(es => es.some(e => e.isIntersecting) && (setSeen(true), io.disconnect()), { rootMargin })
    io.observe(el)
    return () => io.disconnect()
  }, [el, seen, rootMargin])
  return [setEl, seen]
}
