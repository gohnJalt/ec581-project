"""BIST100 regime filter (Section 4.3 of the brief).

Long-eligible bars: BIST100 close > rolling-HP-filtered BIST100 close.
Short-eligible bars: BIST100 close < rolling-HP-filtered BIST100 close.
"""

from __future__ import annotations

import pandas as pd

from src.features.hp import rolling_hp_trend


def bist_regime_flag(
    bist_close: pd.Series, lam: float = 1600, window: int = 504) -> pd.Series:
    """+1 when BIST close > HP trend, -1 when below, 0 in warmup."""
    trend = rolling_hp_trend(bist_close, lam=lam, window=window)
    flag = pd.Series(0, index=bist_close.index, dtype=float, name="regime")
    flag[bist_close > trend] = 1.0
    flag[bist_close < trend] = -1.0
    flag.iloc[:window] = 0.0
    return flag
