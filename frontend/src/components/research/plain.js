// Plain-English layer: turns the engine's statistical wording into sentences a non-statistician can read.
// Display only: the numbers and verdicts come from the backend untouched.

/** Plain names + one-line "what this asks" for every check the engine runs. */
export const TEST_PLAIN = {
  lead_lag: ['Does the cause come first?', 'Checks the timing: does a move in {P} show up before moves in {O}, and not the other way round?'],
  confounder_control: ['Does it survive other explanations?', 'Re-runs the comparison while allowing for common drivers such as the overall market. If the link disappears, {P} was only a bystander.'],
  granger: ['Does {P} add new information?', 'Checks whether {P} helps predict {O} beyond what {O}’s own recent history already tells us.'],
  placebo_shift: ['Is it stronger than pure luck?', 'Shifts {P} randomly in time many times. A real link should beat these fake versions.'],
  negative_control: ['Does it stay quiet where it should?', 'Runs {P} against an unrelated series it has no reason to affect. A real, specific link should stay quiet there.'],
  event_study: ['Does it react to real events?', 'Removes the days around Federal Reserve meetings and checks the link does not depend on them.'],
  validation_replication: ['Does it repeat on newer data?', 'The link is found on the oldest half of the data, then re-checked on the next quarter without any adjustment.'],
  holdout_single_evaluation: ['Does it work on data it never saw?', 'The final quarter of the data was locked away and is tested exactly once. This is the most honest test.'],
  hac_holm: ['Is it significant after fair adjustment?', 'Tests the link on all the data, allowing for noisy, clustered returns and for the fact that several time delays were tried.'],
  block_bootstrap: ['Is the link steady when data is reshuffled?', 'Resamples the history in blocks thousands of times. The range of results should stay clear of zero.'],
  placebo_max_stat: ['Does it beat the best fake?', 'Compares the link with the strongest result found from randomly time-shifted copies of {P}.'],
  window_stability: ['Is it steady over time?', 'Splits history into four periods and checks the link points the same way in most of them.'],
  outlier_sensitivity: ['Is it driven by a few wild days?', 'Trims the most extreme days and checks the link barely changes.'],
  direction_consistency: ['Does it point the same way throughout?', 'Checks the link has the same direction in the training, validation and final-quarter data.'],
  family_multiple_testing: ['Does it survive all the ideas tried?', 'The more ideas tested in one run, the more likely one looks good by luck. This test raises the bar to compensate.'],
}

const RESULT = { pass: 'Passed', fail: 'Failed', inconclusive: 'Unclear', not_run: 'Not run' }
export const resultText = r => RESULT[r] || r

const fill = (s, n) => s.replace(/\{P\}/g, n?.predictor?.label || 'the first series').replace(/\{O\}/g, n?.outcome?.label || 'the second series')

export function testInfo(t, n) {
  const p = TEST_PLAIN[t.id]
  return p ? { name: fill(p[0], n), what: fill(p[1], n) } : { name: t.name, what: '' }
}

const JARGON = [
  [/\bHolm[- ]corrected\b|\bafter Holm correction\b|\bHolm correction\b/gi, 'after adjusting for repeated testing'],
  [/\(?lags? 1-3, Bonferroni\)?/gi, ''],
  [/\bprewhitened\b/gi, 'filtered'],
  [/\bHAC\b/g, 'robust'],
  [/\bout-of-sample\b/gi, 'on unseen data'],
  [/\bholdout\b/gi, 'final-quarter test'],
  [/\bconfounders?\b/gi, m => (m.toLowerCase().endsWith('s') ? 'hidden drivers' : 'hidden driver')],
  [/\battenuation\b/gi, 'shrinkage'],
]

/** Replace the X / Y placeholders and light jargon in engine sentences. */
export function plain(text, n) {
  if (!text) return text
  const P = n?.predictor?.label || 'the first series'
  const O = n?.outcome?.label || 'the second series'
  let s = String(text)
  s = s.replace(/\bX-leads-Y\b/g, `${P} leading ${O}`).replace(/\bY-leads-X\b/g, `${O} leading ${P}`)
  s = s.replace(/\bX\s*(?:->|→)\s*Y\b/g, `${P} → ${O}`).replace(/\bY\s*(?:->|→)\s*X\b/g, `${O} → ${P}`)
  s = s.replace(/\bX['’]s\b/g, `${P}’s`).replace(/\bY['’]s\b/g, `${O}’s`)
  s = s.replace(/\bX\b/g, P).replace(/\bY\b/g, O)
  for (const [re, to] of JARGON) s = s.replace(re, to)
  return s.replace(/\s{2,}/g, ' ').replace(/\s+([.,;])/g, '$1')
}

/** A readable name for a graph variable id (including latent:... ones). */
export function nodeName(id, graph) {
  const nd = graph?.nodes?.find(x => x.id === id)
  if (nd && nd.label && nd.label !== id) return nd.label
  return String(id).replace(/^latent:/, '').replace(/_/g, ' ')
}

/** A paragraph that explains the causal verdict without statistics jargon. */
export function causalStory(verdict, n) {
  const P = n?.predictor?.label || 'the first series', O = n?.outcome?.label || 'the second series'
  if (verdict === 'evidence_consistent')
    return `Every check we could run agrees that ${P} moves before ${O}, and the link holds up when we allow for other explanations. That is encouraging, but it still does not prove that ${P} causes ${O}: something we cannot measure could be driving both.`
  if (verdict === 'contradicted')
    return `The checks point away from ${P} causing ${O}. Either the link disappears once other factors are allowed for, or the timing suggests it runs the other way round.`
  return `Some checks passed, but not all of them (or some could not be run), so we cannot say that ${P} drives ${O}. Treat it as a pattern worth watching, not as an explanation.`
}

export function linkStrength(r) {
  const a = Math.abs(r)
  const w = a >= 0.5 ? 'strong' : a >= 0.3 ? 'moderate' : a >= 0.1 ? 'weak' : 'negligible'
  return `${w} ${r >= 0 ? 'positive' : 'negative'}`
}

export function pPlain(p) {
  if (p == null) return '–'
  if (p < 0.001) return 'under 0.1% (very unlikely to be luck)'
  if (p < 0.01) return `${(p * 100).toFixed(1)}% (unlikely to be luck)`
  if (p < 0.05) return `${(p * 100).toFixed(1)}% (probably not luck)`
  return `${(p * 100).toFixed(0)}% (could easily be luck)`
}
