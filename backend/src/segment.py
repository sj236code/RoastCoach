# backend/src/segment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


@dataclass
class Rep:
    start_idx: int
    end_idx: int
    mid_idx: int  # deepest point / max flex point
    quality: float


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    w = int(w)
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def _nan_interp(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs (only for segmentation/resampling)."""
    x = x.astype(float)
    n = len(x)
    idx = np.arange(n)
    good = np.isfinite(x)
    if good.sum() < 2:
        return x
    return np.interp(idx, idx[good], x[good])


def _score_driver(sig: np.ndarray, conf: Optional[np.ndarray] = None) -> float:
    """Heuristic: want few NaNs + decent amplitude + smooth periodic energy."""
    x = sig.astype(float)
    finite = np.isfinite(x)
    frac_good = finite.mean()
    if frac_good < 0.6:
        return 0.0

    xi = _nan_interp(x)
    amp = float(np.nanpercentile(xi, 95) - np.nanpercentile(xi, 5))
    if amp < 10:
        return 0.0

    y = xi - np.mean(xi)
    yf = np.fft.rfft(y)
    p = np.abs(yf) ** 2
    if len(p) < 8:
        lf_ratio = 0.5
    else:
        lf = p[1:6].sum()
        total = p[1:].sum() + 1e-8
        lf_ratio = float(lf / total)

    conf_term = 1.0
    if conf is not None and len(conf) == len(sig):
        conf_term = float(np.clip(np.nanmean(conf), 0.0, 1.0))

    return frac_good * (amp / 90.0) * lf_ratio * conf_term


def choose_driver_column(df: pd.DataFrame) -> str:
    candidates_priority = [
        "left_knee_flex", "right_knee_flex",
        "left_hip_flex", "right_hip_flex",
        "trunk_incline",
        "left_elbow_flex", "right_elbow_flex",
    ]
    available = [c for c in candidates_priority if c in df.columns]
    if not available:
        raise ValueError("No candidate driver angles found in dataframe.")

    conf = df["conf"].to_numpy() if "conf" in df.columns else None
    scores = {c: _score_driver(df[c].to_numpy(), conf=conf) for c in available}

    knee = [c for c in available if "knee" in c]
    if knee:
        best_knee = max(knee, key=scores.get)
        if scores[best_knee] >= 0.15:
            return best_knee

    return max(scores, key=scores.get)


def _local_maxima_indices(x: np.ndarray) -> np.ndarray:
    """
    Stable maxima detector:
    local max if x[i] >= neighbors AND strictly greater than at least one neighbor.
    Helps with flat/rounded tops.
    """
    xm1 = x[:-2]
    x0 = x[1:-1]
    xp1 = x[2:]
    is_ge = (x0 >= xm1) & (x0 >= xp1)
    is_strict = (x0 > xm1) | (x0 > xp1)
    idx = np.where(is_ge & is_strict)[0] + 1
    return idx


def _prune_peaks_by_time(
    peaks: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    min_peak_distance_s: float,
) -> np.ndarray:
    """
    Enforce minimum time between peaks by keeping the higher peak within each window.
    """
    if len(peaks) <= 1:
        return peaks

    keep = [int(peaks[0])]
    for p in peaks[1:]:
        p = int(p)
        last = keep[-1]
        if (t[p] - t[last]) >= min_peak_distance_s:
            keep.append(p)
        else:
            if x[p] > x[last]:
                keep[-1] = p
    return np.array(keep, dtype=int)


def find_reps_from_driver(
    df: pd.DataFrame,
    driver_col: str,
    smooth_window: int = 9,
    min_rep_seconds: float = 0.7,
    max_rep_seconds: float = 8.0,
    min_peak_distance_s: float = 0.5,
    min_amp_deg: float = 15.0,
    min_dip_deg: float = 10.0,
    mid_frac_bounds: Tuple[float, float] = (0.20, 0.80),
    end_buffer_s: float = 0.35,
) -> Tuple[List[Rep], np.ndarray]:
    """
    Segment reps from a driver angle.

    Rep = top (max) -> bottom (min) -> top (max)

    Key reinforces:
    - prune peaks too close (wavering tops)
    - require meaningful amplitude
    - require BOTH endpoints are true tops above the valley (prevents half reps)
    - require valley is not at edges (prevents partial rep + plateau splits)
    - ignore peaks too close to end of clip (optional)
    """
    if "t" not in df.columns:
        raise ValueError("Dataframe must contain time column 't' in seconds.")

    t = df["t"].to_numpy().astype(float)
    x_raw = df[driver_col].to_numpy().astype(float)

    x = _nan_interp(x_raw)
    x_s = _moving_average(x, smooth_window)

    peaks = _local_maxima_indices(x_s)
    if len(peaks) < 2:
        return [], x_s

    # Optional: drop peaks too close to the end of the signal
    t_end = float(t[-1])
    peaks = peaks[t[peaks] <= (t_end - end_buffer_s)]
    if len(peaks) < 2:
        return [], x_s

    peaks = _prune_peaks_by_time(peaks, t, x_s, min_peak_distance_s=min_peak_distance_s)
    if len(peaks) < 2:
        return [], x_s

    reps: List[Rep] = []

    for i in range(len(peaks) - 1):
        a = int(peaks[i])
        b = int(peaks[i + 1])
        if b <= a + 3:
            continue

        dur = float(t[b] - t[a])
        if dur < min_rep_seconds or dur > max_rep_seconds:
            continue

        seg = x_s[a:b + 1]
        seg_max = float(np.max(seg))
        seg_min = float(np.min(seg))
        amp = seg_max - seg_min
        if amp < min_amp_deg:
            continue

        # deepest point
        mid_local = int(np.argmin(seg))
        mid = a + mid_local

        # valley must be reasonably centered (avoid partial reps / plateau splits)
        frac = mid_local / max(1, (len(seg) - 1))
        lo, hi = mid_frac_bounds
        if frac < lo or frac > hi:
            continue

        # **Critical fix**: BOTH endpoints must be "tops" above the valley
        # (prevents counting a half rep that ends without returning to top)
        top_min = float(min(x_s[a], x_s[b]))
        dip_two_sided = top_min - seg_min
        if dip_two_sided < min_dip_deg:
            continue

        conf_q = float(np.nanmean(df["conf"].to_numpy()[a:b + 1])) if "conf" in df.columns else 1.0
        quality = float(np.clip((amp / 90.0) * conf_q, 0.0, 1.0))

        reps.append(Rep(start_idx=a, end_idx=b, mid_idx=mid, quality=quality))

    return reps, x_s


def resample_rep_angles(
    df: pd.DataFrame,
    rep: Rep,
    angle_cols: List[str],
    N: int = 100
) -> np.ndarray:
    """Resample a rep window to N timesteps for the given angle columns."""
    seg = df.iloc[rep.start_idx:rep.end_idx + 1]
    t = seg["t"].to_numpy().astype(float)

    t0, t1 = t[0], t[-1]
    if t1 - t0 < 1e-6:
        return np.full((N, len(angle_cols)), np.nan)

    tau = (t - t0) / (t1 - t0)
    tau_new = np.linspace(0, 1, N)

    out = np.zeros((N, len(angle_cols)), dtype=float)
    for j, c in enumerate(angle_cols):
        y = seg[c].to_numpy().astype(float)
        y = _nan_interp(y)
        out[:, j] = np.interp(tau_new, tau, y)

    return out


# ------------------- Optional helpers -------------------

def filter_reps(
    reps: List[Rep],
    min_quality: float = 0.20,
    drop_first: bool = False,
) -> List[Rep]:
    if not reps:
        return []
    reps_sorted = sorted(reps, key=lambda r: r.start_idx)
    if drop_first and len(reps_sorted) >= 3:
        reps_sorted = reps_sorted[1:]
    return [r for r in reps_sorted if r.quality >= min_quality]


def mean_std_normalized_rep(
    df: pd.DataFrame,
    reps: List[Rep],
    angle_cols: List[str],
    N: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not reps:
        J = len(angle_cols)
        empty = np.full((N, J), np.nan)
        return empty, empty, np.full((0, N, J), np.nan)

    mats = [resample_rep_angles(df, r, angle_cols, N=N) for r in reps]
    stack = np.stack(mats, axis=0)  # (R, N, J)
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    return mean, std, stack
