"""Seeded synthetic datasets with known generating processes."""
import numpy as np
import pandas as pd

N_OBS = 750
START = "2021-01-04"
DEFAULT_SEED = 42

# Ground truth lives here and is only attached to the report AFTER all tests
# ran. It is never passed to the hypothesis generator (Gemini / template).
GROUND_TRUTH = {
    "demo_null": {
        "generating_process": "x, y, market factor are mutually independent i.i.d. Student-t(5) draws.",
        "has_common_factor": False, "direct_effect_x_to_y": False, "lag": None,
        "expected_causal_verdict": "not_established",
    },
    "demo_confounder": {
        "generating_process": "x = 0.8*M + e1, y = 0.8*M + e2 with M the market factor; no x->y term.",
        "has_common_factor": True, "direct_effect_x_to_y": False, "lag": None,
        "expected_causal_verdict": "contradicted",
    },
    "demo_effect": {
        "generating_process": "x is AR(1, phi=0.4); y_t = 0.15*x_t + 0.35*x_{t-1} + e_t; "
                              "market factor independent of both.",
        "has_common_factor": False, "direct_effect_x_to_y": True, "lag": 1,
        "expected_causal_verdict": "evidence_consistent",
    },
}


def _t(rng, n):
    return rng.standard_t(5, n) / np.sqrt(5 / 3)  # unit variance fat tails


def generate(preset_id: str, seed: int = DEFAULT_SEED, n: int = N_OBS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor = _t(rng, n)
    if preset_id == "demo_null":
        x, y = _t(rng, n), _t(rng, n)
    elif preset_id == "demo_confounder":
        x = 0.8 * factor + _t(rng, n)
        y = 0.8 * factor + _t(rng, n)
    elif preset_id == "demo_effect":
        e = _t(rng, n)
        ex = _t(rng, n)
        x = np.zeros(n)
        for t in range(n):
            x[t] = (0.4 * x[t - 1] if t else 0.0) + ex[t]
        y = np.zeros(n)
        y[0] = e[0] + 0.15 * x[0]
        y[1:] = 0.15 * x[1:] + 0.35 * x[:-1] + e[1:]
    else:
        raise ValueError(f"unknown preset {preset_id}")
    idx = pd.bdate_range(START, periods=n)
    return pd.DataFrame({"x": x, "y": y, "factor": factor}, index=idx)
