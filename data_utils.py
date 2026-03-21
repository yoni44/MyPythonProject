"""
Data loading and processing utilities for BME688 multi-dimensional sensor data.
Structure: 8 sensors × 10 heating steps × m GR values per block.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
SUBSTANCES = ["Acetone", "Redidlo", "Softasept", "Savo", "Vinegar"]
N_SENSORS, N_STEPS = 8, 10
SENSOR_COL, STEP_COL, GR_COL, ERROR_COL, CYCLE_COL = 0, 8, 7, 12, 11

# Windowing: drop first n unstable, take m steady-state values per (sensor, HS)
N_WARMUP, M_STEADY = 30, 10

# Target samples per substance for visualization (sliding windows used to reach this)
TARGET_SAMPLES_PER_SUBSTANCE = 300
WINDOW_STRIDE = 1  # stride for sliding windows (1 = max overlap, ~300 samples/substance)

# Statistical features for calibration and feature extraction
STAT_NAMES = ["median", "mean", "std", "min", "max", "q25", "q75", "range", "cv", "count"]


def apply_windowing(
    block: Dict[Tuple[int, int], List[float]], n: int = N_WARMUP, m: int = M_STEADY
) -> Dict[Tuple[int, int], List[float]]:
    """
    Raw data windowing: discard first n unstable samples, take exactly m steady-state values
    per (sensor, HS). If fewer than n+m values, take last m (fallback to include more samples).
    """
    out = {}
    for (s, st), vals in block.items():
        arr = np.array(vals, dtype=float)
        if len(arr) >= n + m:
            steady = arr[n : n + m].tolist()
            out[(s, st)] = steady
        elif len(arr) >= m:
            # Fallback: take last m values (more samples for visualization)
            steady = arr[-m:].tolist()
            out[(s, st)] = steady
        else:
            out[(s, st)] = []
    return out


def apply_windowing_multi(
    block: Dict[Tuple[int, int], List[float]],
    n: int = N_WARMUP,
    m: int = M_STEADY,
    stride: int = WINDOW_STRIDE,
    max_windows: int = 100,
) -> List[Dict[Tuple[int, int], List[float]]]:
    """
    Extract multiple windowed sub-blocks from one raw block using sliding windows.
    Use to increase sample count (e.g. ~300 per substance for visualization).
    Returns list of windowed blocks.
    """
    if stride < 1:
        stride = 1
    # Find min length across (sensor, step) for valid keys (need at least m values)
    lengths = {}
    for (s, st), vals in block.items():
        L = len(vals)
        if L >= m:
            lengths[(s, st)] = L
    if not lengths:
        return []

    min_len = min(lengths.values())
    keys_ok = [k for k, L in lengths.items() if L >= n + m]

    if not keys_ok:
        # Fallback: take last m for at least one window
        out_blk = {}
        for (s, st), vals in block.items():
            arr = np.array(vals, dtype=float)
            if len(arr) >= m:
                out_blk[(s, st)] = arr[-m:].tolist()
            else:
                out_blk[(s, st)] = []
        return [out_blk] if out_blk else []

    # Steady-state windows: [n, n+m), [n+stride, n+stride+m), ...
    n_windows = 0
    start = n
    while start + m <= min_len and n_windows < max_windows:
        n_windows += 1
        start += stride
    if n_windows == 0:
        # Single window [n:n+m] or last m
        return [apply_windowing(block, n=n, m=m)]

    result = []
    for i in range(min(n_windows, max_windows)):
        start = n + i * stride
        if start + m > min_len:
            break
        sub = {}
        for (s, st), vals in block.items():
            arr = np.array(vals, dtype=float)
            if len(arr) >= start + m:
                sub[(s, st)] = arr[start : start + m].tolist()
            elif len(arr) >= m:
                sub[(s, st)] = arr[-m:].tolist()  # fallback: keep last m for validity
            else:
                sub[(s, st)] = []
        if sub:
            result.append(sub)
    return result if result else [apply_windowing(block, n=n, m=m)]


def _parse_bmerawdata_block(data_block: list) -> Dict[Tuple[int, int], List[float]]:
    """Parse dataBlock into dict (sensor, step) -> list of GR values."""
    gr = defaultdict(list)
    for row in data_block:
        if len(row) < 13:
            continue
        s, st, g = row[SENSOR_COL], row[STEP_COL], row[GR_COL]
        err = row[ERROR_COL] if len(row) > ERROR_COL else 0
        if g > 0 and err == 0:
            gr[(s, st)].append(g)
    return dict(gr)


def _parse_csv_block(df: pd.DataFrame) -> Dict[Tuple[int, int], List[float]]:
    """Parse CSV into dict (sensor, step) -> list of GR values."""
    gr = defaultdict(list)
    for snum in range(1, 9):
        for stnum in range(1, 11):
            col = f"bme688_{snum}_gas_res_step{stnum}"
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            vals = vals[vals > 0]
            gr[(snum - 1, stnum - 1)].extend(vals.tolist())
    return dict(gr)


def load_block_bmerawdata(fpath: Path) -> List[Dict[Tuple[int, int], List[float]]]:
    """Load .bmerawdata file. Returns list of blocks (one per scanning cycle)."""
    try:
        data = json.load(open(fpath, "r", encoding="utf-8"))
    except Exception:
        return []
    data_block = data.get("rawDataBody", {}).get("dataBlock", [])
    if not data_block:
        return []
    # Group by (sensor, step, cycle) then by cycle
    by_cycle = defaultdict(lambda: defaultdict(list))
    for row in data_block:
        if len(row) < 13:
            continue
        s, st, g = row[SENSOR_COL], row[STEP_COL], row[GR_COL]
        cycle = row[CYCLE_COL] if len(row) > CYCLE_COL else 0
        err = row[ERROR_COL] if len(row) > ERROR_COL else 0
        if g > 0 and err == 0:
            by_cycle[cycle][(s, st)].append(g)
    return [dict(c) for c in by_cycle.values()]


def load_block_csv(fpath: Path) -> List[Dict[Tuple[int, int], List[float]]]:
    """Load CSV file. Each file = 1 block."""
    try:
        df = pd.read_csv(fpath)
    except Exception:
        return []
    gr = _parse_csv_block(df)
    return [gr] if gr else []


def load_blocks_from_file(fpath: Path) -> List[Dict[Tuple[int, int], List[float]]]:
    """Load blocks from file (bmerawdata or csv)."""
    if fpath.suffix.lower() == ".csv":
        return load_block_csv(fpath)
    return load_block_bmerawdata(fpath)


def load_folder_blocks(
    folder: Path, pattern: str
) -> Tuple[List[Dict], List[str]]:
    """Load all blocks from files matching pattern. Returns (blocks, block_labels)."""
    files = sorted(folder.glob(pattern))
    blocks, labels = [], []
    for f in files:
        blks = load_blocks_from_file(f)
        for i, b in enumerate(blks):
            blocks.append(b)
            labels.append(f"{f.stem}" + (f"_{i}" if len(blks) > 1 else ""))
    return blocks, labels


def compute_stats(vals: np.ndarray) -> Dict[str, float]:
    """Compute statistical features for a 1D array."""
    if len(vals) == 0:
        return {k: np.nan for k in STAT_NAMES}
    q25, q75 = np.percentile(vals, [25, 75])
    std = np.std(vals)
    mean = np.mean(vals)
    cv = (std / mean * 100) if mean != 0 else np.nan
    return {
        "median": np.median(vals),
        "mean": mean,
        "std": std if not np.isnan(std) else 0,
        "min": np.min(vals),
        "max": np.max(vals),
        "q25": q25,
        "q75": q75,
        "range": np.max(vals) - np.min(vals),
        "cv": cv,
        "count": len(vals),
    }


def compute_baseline(
    blocks: List[Dict],
    apply_windowing_flag: bool = True,
    n: int = N_WARMUP,
    m: int = M_STEADY,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Compute calibration baseline from Normal Air. Pool GR across blocks, stats per (sensor, step)."""
    pooled = defaultdict(list)
    for blk in blocks:
        if apply_windowing_flag:
            blk = apply_windowing(blk, n=n, m=m)
        for (s, st), vals in blk.items():
            if len(vals) >= m:
                pooled[(s, st)].extend(vals)
    return {k: compute_stats(np.array(v)) for k, v in pooled.items()}


def is_sensor_valid(
    block: Dict,
    sensor: int,
    min_steps_with_data: int = 5,
    min_count_per_step: int = M_STEADY,
    gr_min: float = 1,
    gr_max: float = 1e10,
) -> bool:
    """Gate bad sensors: require enough steady-state readings per step, reasonable range."""
    steps_ok = 0
    for st in range(N_STEPS):
        vals = block.get((sensor, st), [])
        if len(vals) >= min_count_per_step:
            arr = np.array(vals)
            if np.all(arr >= gr_min) and np.all(arr <= gr_max):
                steps_ok += 1
    return steps_ok >= min_steps_with_data


def normalize_block(
    block: Dict, baseline: Dict[Tuple[int, int], Dict[str, float]]
) -> Dict[Tuple[int, int], np.ndarray]:
    """Normalize block: (GR - median) / std, using baseline. Returns normalized arrays."""
    out = {}
    for (s, st), vals in block.items():
        b = baseline.get((s, st), {})
        med = b.get("median", np.nan)
        std = b.get("std", 1.0)
        if np.isnan(med) or std <= 0:
            std = 1.0
        arr = np.array(vals, dtype=float)
        out[(s, st)] = (arr - med) / std
    return out


def extract_features(
    block: Dict,
    baseline: Dict,
    valid_sensors: Optional[List[int]] = None,
) -> np.ndarray:
    """Extract flat feature vector: stats per (sensor, step) for valid sensors."""
    if valid_sensors is None:
        valid_sensors = list(range(N_SENSORS))
    feats = []
    for s in valid_sensors:
        for st in range(N_STEPS):
            vals = block.get((s, st), [])
            stats = compute_stats(np.array(vals))
            feats.extend([stats[k] for k in STAT_NAMES])
    return np.array(feats, dtype=float)


def get_valid_sensors(block: Dict) -> List[int]:
    """Return list of valid sensor indices for this block."""
    return [s for s in range(N_SENSORS) if is_sensor_valid(block, s)]
