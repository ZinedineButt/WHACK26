"""Decision thresholds. Fixed BEFORE any demo dataset was run; do not tune
them to obtain a desired verdict on a particular preset."""

MIN_N = 60                    # below this, association verdict is inconclusive
ASSOC_DETECT_P = 0.05         # HAC p-value for 'detected' (strength score carries the nuance)
ASSOC_DETECT_ABS_R = 0.10     # minimum |r| for 'detected' (practical size)
ASSOC_NULL_P = 0.05           # HAC p >= this -> 'not_detected'
ROLLING_WINDOW = 60

LAG_RANGE = 5                 # lead/lag scan over k = -5..+5
ALPHA = 0.05
GRANGER_MAXLAG = 3
CONFOUND_ATTENUATION = 0.5    # controlled beta shrinks by >=50% -> confounder explains it
PLACEBO_DRAWS = 499

SPLIT = (0.5, 0.25, 0.25)     # chronological train / validation / holdout
MIN_VALIDATION_N = 200
BOOTSTRAP_B = 1000
BOOTSTRAP_BLOCK = 20
STABILITY_WINDOWS = 4
STABILITY_MIN_SAME_SIGN = 3
OUTLIER_WINSOR = 0.01
OUTLIER_MAX_REL_CHANGE = 0.25
SEED = 42

# backtest / generalisation / critique (fixed before any run)
COST_BPS = 2.0                 # per unit of turnover, each side
BT_MIN_N = 60                  # min out-of-sample days for a backtest verdict
BT_BOOT_B = 1000
BT_BOOT_BLOCK = 10
PEER_MAX = 5
PEER_SAME_SIGN_SHARE = 0.6     # generalises: >=60% of peers share the sign ...
PEER_SIGNIFICANT_SHARE = 0.5   # ... and >=50% are significant after Holm
