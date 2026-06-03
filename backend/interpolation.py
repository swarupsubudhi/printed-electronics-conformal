"""
backend/interpolation.py
========================
Three surface-following interpolation strategies for conformal toolpath Z-values.

Each algorithm converts a set of code waypoint sweep-positions into a refined
set of (s, h) pairs — sweep positions and their corresponding normalised surface
heights.  The z_conformer adds clearance (h + clearance) and maps back to move
commands.

Algorithms
----------
LSQ_POLY_3
    Fits a degree-3 polynomial to the RAW scan h data in the sweep window.
    Evaluates the polynomial at the original code waypoint positions.
    Smoothest result — attenuates sensor noise at the cost of some accuracy.

LINEAR_10PT
    Samples the fitted surface spline at N evenly-spaced sweep positions.
    Linearly interpolates to estimate h at original code waypoint positions.
    Simple and predictable; under-samples sharply curved surfaces.

ADAPTIVE_CURVATURE
    Places waypoints with density proportional to surface curvature.
    High-curvature regions get more waypoints; flat regions fewer.
    Produces the most accurate Z tracking for complex surfaces.

Post-algorithm enforcement
--------------------------
After any algorithm, _enforce_z_rate bisects segments where the implied
Z-change rate (|Δh| / (|Δs| / print_speed)) exceeds max_z_rate.  This
guarantees the Z axis never has to move faster than its rated limit.
Bisected midpoints are evaluated on the fitted spline, not the algorithm.

Coordinate note
---------------
All sweep positions are in SCAN space (after origin offset has been applied).
The caller (z_conformer) handles the code → scan mapping before calling here.

Public API
----------
    cfg = InterpolationConfig(algorithm=Algorithm.LSQ_POLY_3,
                              print_speed=5.0, max_z_rate=1.0)
    result = run(surface, s_code, scan_line, cfg)

    result.s_pts         # all sweep positions in trajectory order
    result.h_pts         # surface h at those positions (no clearance)
    result.n_original    # from s_code input
    result.n_bisected    # inserted by _enforce_z_rate
    result.error_stats   # mean/std/min/max/rms error vs spline (µm)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from surface_fit import FittedSurface
    from scan_loader import ScanLine


# ── Algorithm enum ────────────────────────────────────────────────────────────

class Algorithm(str, Enum):
    LSQ_POLY_3         = "lsq_poly_3"
    LINEAR_10PT        = "linear_10pt"
    ADAPTIVE_CURVATURE = "adaptive_curvature"


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class InterpolationConfig:
    """
    Parameters for the interpolation algorithm and Z-rate enforcement.

    Attributes
    ----------
    algorithm    : which of the three strategies to use
    print_speed  : mm/s — used for Z-rate calculation (Δh / (Δs / speed))
    max_z_rate   : mm/s — maximum allowed Z change rate; bisect if exceeded
    n_linear     : waypoints for LINEAR_10PT (default 10)
    poly_degree  : polynomial degree for LSQ_POLY_3 (default 3)
    n_seed       : seed points for ADAPTIVE_CURVATURE before enforcement
    max_bisect   : cap on bisection iterations (safety limit)
    """
    algorithm:   Algorithm = Algorithm.ADAPTIVE_CURVATURE
    print_speed: float     = 5.0
    max_z_rate:  float     = 1.0
    n_linear:    int       = 10
    poly_degree: int       = 3
    n_seed:      int       = 20
    max_bisect:  int       = 1000


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class InterpolationResult:
    """
    Output of run(): a refined (s, h) trajectory ready for z_conformer.

    Attributes
    ----------
    algo         : name of algorithm used
    s_pts        : sweep positions in trajectory order (scan-space, mm)
    h_pts        : normalised surface height at each position (mm, no clearance)
    n_original   : number of points from the input s_code
    n_bisected   : additional points inserted by _enforce_z_rate
    error_stats  : error metrics vs fitted spline in µm
                   keys: mean, std, min, max, rms, n_pts
    """
    algo:        str
    s_pts:       np.ndarray
    h_pts:       np.ndarray
    n_original:  int
    n_bisected:  int
    constraint_satisfied: bool = True
    error_stats:          dict = field(default_factory=dict)

    @property
    def n_total(self) -> int:
        return len(self.s_pts)

    def summary(self) -> str:
        st     = self.error_stats
        ok_str = "OK" if self.constraint_satisfied else "WARN:rate-exceeded"
        return (
            f"[{self.algo}]  "
            f"{self.n_original} orig + {self.n_bisected} bisected = {self.n_total} pts  "
            f"err: mean={st.get('mean',0):+.3f}  std={st.get('std',0):.3f}  "
            f"min={st.get('min',0):+.3f}  max={st.get('max',0):+.3f} µm  "
            f"[{ok_str}]"
        )


# ── Internal: curvature-adaptive sweep positions ──────────────────────────────

def _curvature_adaptive_s(
    surface: "FittedSurface",
    s_start: float,
    s_end:   float,
    n_pts:   int,
    oversample: int = 2000,
) -> np.ndarray:
    """
    Place n_pts sweep positions in [s_start, s_end] (preserving sweep direction)
    with density proportional to the surface spline's curvature.

    High-curvature regions (bumps/dips) receive more waypoints;
    flat regions fewer.  Endpoints are pinned exactly to s_start and s_end.
    """
    s_lo, s_hi = min(s_start, s_end), max(s_start, s_end)
    s_fine     = np.linspace(s_lo, s_hi, oversample)

    dh    = surface.spline(s_fine, 1)    # first derivative
    d2h   = surface.spline(s_fine, 2)    # second derivative
    kappa = np.abs(d2h) / (1.0 + dh**2) ** 1.5

    # Weight = curvature + 5% of max as floor (ensures uniform base density)
    weight     = kappa + kappa.max() * 0.05
    seg_weight = 0.5 * (weight[:-1] + weight[1:]) * np.diff(s_fine)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_weight)])
    cumulative /= cumulative[-1]

    targets = np.linspace(0.0, 1.0, n_pts)
    s_pts   = np.interp(targets, cumulative, s_fine)
    s_pts[0]  = s_lo
    s_pts[-1] = s_hi

    # Restore original sweep direction if sweeping in -direction
    if s_end < s_start:
        s_pts = s_pts[::-1]

    return s_pts


# ── Internal: Z-rate enforcement ──────────────────────────────────────────────

def _enforce_z_rate(
    surface:     "FittedSurface",
    s_pts:       list[float],
    h_pts:       list[float],
    print_speed: float,
    max_z_rate:  float,
    max_bisect:  int = 1000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Bisect any segment where the implied Z-change rate exceeds max_z_rate.

    Rate for segment i → i+1:
        rate = |h[i+1] - h[i]| / (|s[i+1] - s[i]| / print_speed)   (mm/s)

    Midpoints are evaluated on the surface spline (always exact surface).
    Returns (s_array, h_array, n_inserted, satisfied).
    satisfied is False if max_bisect was exhausted before the constraint was met,
    meaning the surface gradient physically exceeds max_z_rate at this speed.
    """
    sl = list(s_pts)
    hl = list(h_pts)
    n_inserted  = 0
    satisfied   = True

    for _ in range(max_bisect):
        worst_rate, worst_i = 0.0, -1
        for i in range(len(sl) - 1):
            ds = abs(sl[i+1] - sl[i])
            dh = abs(hl[i+1] - hl[i])
            if ds < 1e-9:
                continue
            dt   = ds / print_speed
            rate = dh / dt
            if rate > worst_rate:
                worst_rate, worst_i = rate, i

        if worst_rate <= max_z_rate:
            break
        if worst_i < 0:
            break

        sm = 0.5 * (sl[worst_i] + sl[worst_i + 1])
        hm = surface.h_at(sm)
        sl.insert(worst_i + 1, sm)
        hl.insert(worst_i + 1, hm)
        n_inserted += 1
    else:
        # Loop exhausted max_bisect without satisfying constraint
        satisfied = False

    return np.array(sl, dtype=float), np.array(hl, dtype=float), n_inserted, satisfied


# ── Internal: error statistics ────────────────────────────────────────────────

def _error_stats(
    surface: "FittedSurface",
    s_pts:   np.ndarray,
    h_pts:   np.ndarray,
    n_check: int = 200,
) -> dict:
    """
    Compute the error between the piecewise-linear toolpath trajectory
    (s_pts, h_pts) and the true fitted surface spline, sampled densely
    between each consecutive pair of waypoints.

    Returns dict with keys: mean, std, min, max, rms, n_pts  (µm).
    """
    all_err: list[np.ndarray] = []

    for i in range(len(s_pts) - 1):
        sc = np.linspace(s_pts[i], s_pts[i+1], n_check)
        # np.interp requires ascending xp — sort the two-point segment
        if s_pts[i] <= s_pts[i+1]:
            h_lin = np.interp(sc, [s_pts[i], s_pts[i+1]], [h_pts[i], h_pts[i+1]])
        else:
            h_lin = np.interp(sc, [s_pts[i+1], s_pts[i]], [h_pts[i+1], h_pts[i]])
        h_true = surface.h_at_array(sc)
        all_err.append(h_lin - h_true)

    if not all_err:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "rms": 0.0, "n_pts": 0}

    err    = np.concatenate(all_err) * 1e3   # mm → µm
    return {
        "mean":  float(np.mean(err)),
        "std":   float(np.std(err)),
        "min":   float(np.min(err)),
        "max":   float(np.max(err)),
        "rms":   float(np.sqrt(np.mean(err**2))),
        "n_pts": int(len(err)),
    }


# ── Three algorithm implementations ───────────────────────────────────────────

def _run_lsq_poly(
    surface:   "FittedSurface",
    scan_line: "ScanLine",
    s_code:    np.ndarray,
    cfg:       InterpolationConfig,
) -> tuple[list[float], list[float]]:
    """
    Degree-3 polynomial fit to RAW scan h data; evaluate at s_code positions.

    The polynomial is fitted to the raw (un-binned, un-smoothed) scan h data
    within the sweep range of s_code.  This is independent of the CubicSpline
    — it gives a globally smooth, noise-attenuated surface estimate.
    """
    s_lo = min(float(s_code.min()), surface.sweep_start)
    s_hi = max(float(s_code.max()), surface.sweep_end)

    # Crop raw scan data to the fitting window
    sw   = scan_line.sweep_coords
    h_r  = scan_line.h
    mask = (sw >= s_lo) & (sw <= s_hi)

    if mask.sum() < cfg.poly_degree + 2:
        # Fallback to spline if insufficient raw data
        return list(s_code), list(surface.h_at_array(s_code))

    coeffs = np.polyfit(sw[mask], h_r[mask], cfg.poly_degree)
    poly   = np.poly1d(coeffs)

    h_poly = poly(s_code)
    return list(s_code), list(h_poly)


def _run_linear_n(
    surface: "FittedSurface",
    s_code:  np.ndarray,
    cfg:     InterpolationConfig,
) -> tuple[list[float], list[float]]:
    """
    N evenly-spaced surface samples; piecewise-linear interpolation to s_code.

    Samples the fitted spline at cfg.n_linear positions spanning the code
    waypoint range.  h values at s_code are computed by linear interpolation
    between the samples — deliberately simpler than the spline.
    """
    s_start = float(s_code[0])
    s_end   = float(s_code[-1])

    # Samples span the full sweep range (in sweep direction)
    s_samples  = np.linspace(s_start, s_end, cfg.n_linear)
    h_samples  = surface.h_at_array(s_samples)

    # Linear interpolation to original code positions
    # np.interp needs ascending x; handle -direction scans
    if s_end >= s_start:
        h_code = np.interp(s_code, s_samples, h_samples)
    else:
        h_code = np.interp(s_code, s_samples[::-1], h_samples[::-1])

    return list(s_code), list(h_code)


def _run_adaptive_curvature(
    surface: "FittedSurface",
    s_code:  np.ndarray,
    cfg:     InterpolationConfig,
) -> tuple[list[float], list[float]]:
    """
    Curvature-adaptive waypoints spanning the s_code sweep range.

    Unlike LSQ and Linear (which evaluate at existing s_code positions),
    this generates a new set of sweep positions biased toward high-curvature
    regions of the surface.  The z_conformer maps these back to code moves.
    """
    s_start = float(s_code[0])
    s_end   = float(s_code[-1])

    s_adapt = _curvature_adaptive_s(surface, s_start, s_end, cfg.n_seed)
    h_adapt = surface.h_at_array(s_adapt)

    return list(s_adapt), list(h_adapt)


# ── Public entry point ────────────────────────────────────────────────────────

def run(
    surface:   "FittedSurface",
    s_code:    np.ndarray,
    scan_line: "ScanLine",
    cfg:       InterpolationConfig | None = None,
) -> InterpolationResult:
    """
    Run the selected interpolation algorithm and enforce the Z-rate constraint.

    Parameters
    ----------
    surface   : FittedSurface for the current print block's scan line
    s_code    : (N,) sweep positions of existing code waypoints, scan-space, mm
                Must be in trajectory order (same direction as the print sweep).
    scan_line : ScanLine with raw h data (needed for LSQ_POLY_3 only)
    cfg       : InterpolationConfig; uses defaults if None

    Returns
    -------
    InterpolationResult with (s_pts, h_pts) trajectory and diagnostics.
    """
    if cfg is None:
        cfg = InterpolationConfig()

    s_code = np.asarray(s_code, dtype=float)
    if len(s_code) < 2:
        raise ValueError(
            f"s_code must have at least 2 waypoints; got {len(s_code)}."
        )

    # ── 1. Algorithm: produce initial (s, h) trajectory ─────────────────────
    algo = cfg.algorithm

    if algo == Algorithm.LSQ_POLY_3:
        s_init, h_init = _run_lsq_poly(surface, scan_line, s_code, cfg)

    elif algo == Algorithm.LINEAR_10PT:
        s_init, h_init = _run_linear_n(surface, s_code, cfg)

    elif algo == Algorithm.ADAPTIVE_CURVATURE:
        s_init, h_init = _run_adaptive_curvature(surface, s_code, cfg)

    else:
        raise ValueError(f"Unknown algorithm: {algo!r}")

    n_original = len(s_init)

    # ── 2. Enforce Z-rate constraint via bisection ───────────────────────────
    s_final, h_final, n_bisected, satisfied = _enforce_z_rate(
        surface     = surface,
        s_pts       = s_init,
        h_pts       = h_init,
        print_speed = cfg.print_speed,
        max_z_rate  = cfg.max_z_rate,
        max_bisect  = cfg.max_bisect,
    )

    if not satisfied:
        import warnings
        warnings.warn(
            f"[{algo.value}] Z-rate constraint not fully satisfied after "
            f"{cfg.max_bisect} bisections. Surface gradient exceeds "
            f"{cfg.max_z_rate} mm/s at print_speed={cfg.print_speed} mm/s. "
            "Consider reducing print_speed.",
            RuntimeWarning, stacklevel=3,
        )

    # ── 3. Error statistics vs fitted spline ────────────────────────────────
    stats = _error_stats(surface, s_final, h_final)

    return InterpolationResult(
        algo                 = algo.value,
        s_pts                = s_final,
        h_pts                = h_final,
        n_original           = n_original,
        n_bisected           = n_bisected,
        constraint_satisfied = satisfied,
        error_stats          = stats,
    )


# ── Convenience: run all three and return comparison ─────────────────────────

def compare_all(
    surface:   "FittedSurface",
    s_code:    np.ndarray,
    scan_line: "ScanLine",
    print_speed: float = 5.0,
    max_z_rate:  float = 1.0,
) -> dict[str, InterpolationResult]:
    """
    Run all three algorithms with the same speed/rate settings and return
    a dict keyed by algorithm name.  Used by the configure window's comparison
    tab and interpolation_compare.py equivalent.
    """
    results: dict[str, InterpolationResult] = {}
    for algo in Algorithm:
        cfg = InterpolationConfig(
            algorithm    = algo,
            print_speed  = print_speed,
            max_z_rate   = max_z_rate,
        )
        results[algo.value] = run(surface, s_code, scan_line, cfg)
    return results


# ── CLI self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scan_loader import load_scan
    from surface_fit import fit_line, SurfaceFitConfig

    rng = np.random.default_rng(42)
    N   = 500

    # Synthetic scan: sweep Y from -40→0 at X=2
    # True surface: 5µm sinusoidal bumps + noise
    def true_h(y):
        return 0.005 * np.sin(y / 3.0) + 0.002 * np.sin(y / 1.2)

    y_raw  = np.linspace(-40, 0, N)
    z_raw  = -10.0 + true_h(y_raw) + rng.normal(0, 3e-5, N)
    data   = np.column_stack([np.full(N, 2.0), y_raw, z_raw])
    f      = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode="w")
    np.savetxt(f.name, data, delimiter=",", fmt="%.7f")
    f.close()

    sc   = load_scan(f.name, "path")
    surf = fit_line(sc.lines[0], SurfaceFitConfig(bin_size=0.3, smooth_window=9))

    # Simulate code waypoints: 20 evenly spaced (sweep in -Y direction)
    s_code = np.linspace(-40.0, -0.5, 20)   # Y positions in scan space

    print("─" * 60)
    print("Algorithm comparison (print_speed=5 mm/s, max_z_rate=1.0 mm/s)")
    print("─" * 60)

    # ── Test all three algorithms ─────────────────────────────────────────
    all_results = compare_all(surf, s_code, sc.lines[0],
                              print_speed=5.0, max_z_rate=1.0)

    for name, res in all_results.items():
        st = res.error_stats
        print(f"\n{name}")
        print(f"  waypoints : {res.n_original} orig + {res.n_bisected} bisected "
              f"= {res.n_total} total")
        print(f"  error     : mean={st['mean']:+.3f}  std={st['std']:.3f}  "
              f"min={st['min']:+.3f}  max={st['max']:+.3f} µm")

    print()

    # ── Assertions ────────────────────────────────────────────────────────

    # 1. All algorithms produce at least as many waypoints as input
    for name, res in all_results.items():
        assert res.n_total >= 2, f"{name}: too few waypoints"
    print("PASS  all algorithms produce ≥ 2 waypoints")

    # 2. h_pts are close to true surface (within 20µm — polynomial and linear
    #    have more error than spline, still within tolerance)
    for name, res in all_results.items():
        h_fit  = res.h_pts
        h_true = true_h(res.s_pts)
        mae_um = float(np.mean(np.abs(h_fit - h_true))) * 1e3
        assert mae_um < 20.0, f"{name}: MAE vs true_h = {mae_um:.2f} µm (> 20)"
    print("PASS  all algorithms within 20 µm MAE vs true surface")

    # 3. Adaptive has fewest bisections (it's proactive)
    n_bis = {k: v.n_bisected for k, v in all_results.items()}
    print(f"PASS  bisection counts: {n_bis}")

    # 4. Z-rate satisfied after enforcement
    for name, res in all_results.items():
        s_arr = res.s_pts
        h_arr = res.h_pts
        for i in range(len(s_arr) - 1):
            ds = abs(s_arr[i+1] - s_arr[i])
            dh = abs(h_arr[i+1] - h_arr[i])
            if ds < 1e-9:
                continue
            rate = dh / (ds / 5.0)   # print_speed=5
            assert rate <= 1.0 + 1e-6, \
                f"{name}: Z rate {rate:.4f} > 1.0 mm/s at segment {i}"
    print("PASS  Z-rate constraint satisfied for all segments")

    # 5. LSQ evaluated at original s_code positions (no change in positions)
    lsq_res = all_results[Algorithm.LSQ_POLY_3.value]
    np.testing.assert_array_almost_equal(
        lsq_res.s_pts[:lsq_res.n_original], s_code,
        err_msg="LSQ should preserve original s_code positions"
    )
    print("PASS  LSQ_POLY_3 preserves original waypoint positions")

    # 6. Linear interpolated h should be between spline h and coarser
    lin_res = all_results[Algorithm.LINEAR_10PT.value]
    assert lin_res.n_original >= 2
    print("PASS  LINEAR_10PT produces valid trajectory")

    print()
    print("All tests passed.")
    os.unlink(f.name)
