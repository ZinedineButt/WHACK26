"""Variable transforms. All operate on a level series that has ALREADY been
aligned to the common observation grid, so no transform uses information from
a date later than the one it is stamped with, EXCEPT the explicitly forward
outcome transforms (fwd_logvol_*), which are only valid as outcomes."""
import numpy as np
import pandas as pd

from . import registry as R

ANNUALISE = np.sqrt(252)


def _log_ret(level: pd.Series) -> pd.Series:
    lv = level.where(level > 0)
    return 100 * np.log(lv).diff()


def apply(level: pd.Series, transform: str) -> pd.Series:
    if transform == R.LOG_RETURN:
        return _log_ret(level)
    if transform == R.DIFF:
        return level.diff()
    if transform == R.LOG_DIFF:
        return _log_ret(level)
    if transform == R.LEVEL:
        return level.astype(float)
    if transform in R.FORWARD_OUTCOMES:
        h = R.FORWARD_OUTCOMES[transform]
        r = _log_ret(level)
        # vol over returns t+1..t+h, stamped at t (forward-looking by construction)
        vol = r.rolling(h).std(ddof=1).shift(-h) * ANNUALISE
        return np.log(vol.where(vol > 0))
    if transform == R.TRAILING_LOGVOL_5:
        r = _log_ret(level)
        vol = r.rolling(5).std(ddof=1) * ANNUALISE      # returns t-4..t: known at t
        return np.log(vol.where(vol > 0))
    raise ValueError(f"unknown transform {transform}")
