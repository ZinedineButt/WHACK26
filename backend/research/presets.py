"""Allowlist of investigations. Free-text / arbitrary tickers are NOT supported."""

PRESETS = {
    "demo_null": {
        "id": "demo_null",
        "label": "Independent returns example",
        "description": "Two synthetic daily return series generated independently.",
        "x_label": "Synthetic stock X returns (%)",
        "y_label": "Synthetic stock Y returns (%)",
    },
    "demo_confounder": {
        "id": "demo_confounder",
        "label": "Shared-market-factor example",
        "description": "Two synthetic daily return series that both load on a common market factor.",
        "x_label": "Synthetic stock X returns (%)",
        "y_label": "Synthetic stock Y returns (%)",
    },
    "demo_effect": {
        "id": "demo_effect",
        "label": "Lagged-effect example",
        "description": "Synthetic daily returns where X's past returns feed into Y (planted ground truth).",
        "x_label": "Synthetic stock X returns (%)",
        "y_label": "Synthetic stock Y returns (%)",
    },
}
# NOTE: `spy_vs_qqq` (optional in the spec) is intentionally not implemented:
# no reliable offline market-data retrieval exists in this repo.


def public_presets():
    return [
        {k: p[k] for k in ("id", "label", "description")}
        for p in PRESETS.values()
    ]
