"""
backend/surface_fit.py
======================
Bins, smooths, and fits a CubicSpline of normalised surface height h(s)
for each forward scan line.

The spline maps scan-space sweep-axis positions to normalised heights:

    h(s)  where  h = Z_raw - Z_datum

    s  = sweep-axis coordinate in SCAN space (mm)
    h  = positive → surface is higher than datum (bump toward laser)
         negative → surface is lower  than datum (dip away  from laser)

This module is intentionally free of all UI and toolpath logic.
It only transforms raw scan lines into interpolatable surface models.

Public API
----------
    cfg = SurfaceFitConfig(bin_size=0.2, smooth_window=11)

    surf  = fit_line(scan_line, cfg)          # one FittedSurface
    surfs = fit_all(loaded_scan, cfg)         # list[FittedSurface]

    h_val   = surf.h_at(s)                   # scalar query
    h_array = surf.h_at_array(s_vec)         # vectorised query
    stats   = surf.residuals(scan_line)       # dict of error metrics

Coordinate note
---------------
All sweep positions (s_start, s_end, spline domain) are in SCAN space.
The caller (z_conformer) maps code XY → scan XY via origin_matcher
before querying this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d

if TYPE_CHECKING:
    from scan_loader import LoadedScan, ScanLine


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class SurfaceFitConfig:
    """
    Parameters controlling the bin → smooth → spline pipeline.

    bin_size      : width of each averaging bin along the sweep axis (mm)
    smooth_window : uniform filter window applied to binned h values
                    (must be odd; set to 1 to disable smoothing)
    sweep_length  : crop each line to this length from its scan start (mm)
                    None → use the full line extent
    min_pts_raw   : minimum raw points in the fitting region; raise if fewer
    min_bins      : minimum populated bins required to fit a spline
    """
    bin_size:      float = 0.20
    smooth_window: int   = 11
    sweep_length:  float | None = None
    min_pts_raw:   int   = 10
    min_bins:      int   = 4


# ── Output data class ─────────────────────────────────────────────────────────

@dataclass
class FittedSurface:
    """
    One fitted surface model for a single scan line.

    Attributes
    ----------
    step_val    : step-axis position of this scan line (mm, scan-space)
    sweep_axis  : 0=X, 1=Y
    sweep_start : start of fitted domain (scan-space, mm)
    sweep_end   : end of fitted domain   (scan-space, mm)
    spline      : CubicSpline h(s) — normalised height vs sweep position
    n_pts_raw   : raw points used in the fitting region
    n_bins      : populated bins used to build the spline
    bin_sweep   : (n_bins,) sweep positions of bin centres (for diagnostics)
    bin_h       : (n_bins,) averaged and smoothed h values per bin
    """
    step_val:    float
    sweep_axis:  int
    sweep_start: float
    sweep_end:   float
    spline:      CubicSpline
    n_pts_raw:   int
    n_bins:      int
    bin_sweep:   np.ndarray
    bin_h:       np.ndarray

    # ── Query interface ────────────────────────────────────────────────────

    def h_at(self, s: float) -> float:
        """
        Evaluate normalised surface height at scan-space sweep position s.
        CubicSpline extrapolates outside [sweep_start, sweep_end] — the
        caller should stay within the fitted domain where possible.
        """
        return float(self.spline(s))

    def h_at_array(self, s: np.ndarray) -> np.ndarray:
        """Vectorised h_at for an array of sweep positions."""
        return self.spline(np.asarray(s, dtype=float))

    # ── Diagnostics ────────────────────────────────────────────────────────

    def residuals(self, scan_line: "ScanLine") -> dict:
        """
        Compute error between raw scan h values and the fitted spline.

        Only points within [sweep_start, sweep_end] are included.
        Returns dict with keys: mean, std, min, max, rms, n_pts (all in µm).
        """
        s_all = scan_line.sweep_coords
        h_all = scan_line.h

        mask   = (s_all >= self.sweep_start) & (s_all <= self.sweep_end)
        s_crop = s_all[mask]
        h_crop = h_all[mask]

        if len(s_crop) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "rms": 0, "n_pts": 0}

        h_fit  = self.h_at_array(s_crop)
        err    = (h_crop - h_fit) * 1e3    # mm → µm

        return {
            "mean":  float(np.mean(err)),
            "std":   float(np.std(err)),
            "min":   float(np.min(err)),
            "max":   float(np.max(err)),
            "rms":   float(np.sqrt(np.mean(err**2))),
            "n_pts": int(len(err)),
        }

    def summary(self) -> str:
        ax = ["X", "Y"][self.sweep_axis]
        return (
            f"FittedSurface  step={self.step_val:.4f}  "
            f"{ax}:[{self.sweep_start:.3f}, {self.sweep_end:.3f}]  "
            f"bins={self.n_bins}  raw_pts={self.n_pts_raw}"
        )


# ── Core fitting function ──────────────────────────────────────────────────────

def fit_line(
    scan_line: "ScanLine",
    cfg: SurfaceFitConfig | None = None,
) -> FittedSurface:
    """
    Fit a surface spline h(s) for one forward scan line.

    Pipeline
    --------
    1. Optionally crop to cfg.sweep_length from the physical scan start
    2. Bin h values along the sweep axis (bin width = cfg.bin_size)
    3. Average h within each populated bin
    4. Apply uniform smoothing filter to binned h (window = cfg.smooth_window)
    5. Fit CubicSpline(bin_sweep_positions, smoothed_h)

    Parameters
    ----------
    scan_line : ScanLine from scan_loader (points sorted ascending by sweep)
    cfg       : SurfaceFitConfig; uses defaults if None

    Returns
    -------
    FittedSurface

    Raises
    ------
    RuntimeError : fewer than cfg.min_pts_raw raw points or cfg.min_bins bins
    """
    if cfg is None:
        cfg = SurfaceFitConfig()

    sweep_vals = scan_line.sweep_coords   # ascending (from scan_loader sort)
    h_vals     = scan_line.h
    sax        = scan_line.sweep_axis

    # ── 1. Determine fitting region ──────────────────────────────────────────
    # scan_start_xy gives the physical start of the scan — its sweep component
    # is the sweep position from which we measure sweep_length.
    scan_start_sweep = (scan_line.scan_start_xy[0] if sax == 0
                        else scan_line.scan_start_xy[1])

    s_lo_data = float(sweep_vals.min())
    s_hi_data = float(sweep_vals.max())

    if cfg.sweep_length is not None:
        # Crop from the physical scan start in the scan direction
        sweep_dir = scan_line.sweep_direction   # +1 or -1
        s_crop_end = scan_start_sweep + sweep_dir * cfg.sweep_length
        s_lo = min(scan_start_sweep, s_crop_end)
        s_hi = max(scan_start_sweep, s_crop_end)
        # Clamp to actual data extent
        s_lo = max(s_lo, s_lo_data)
        s_hi = min(s_hi, s_hi_data)
    else:
        s_lo, s_hi = s_lo_data, s_hi_data

    mask = (sweep_vals >= s_lo) & (sweep_vals <= s_hi)
    n_raw = int(mask.sum())

    if n_raw < cfg.min_pts_raw:
        raise RuntimeError(
            f"Only {n_raw} raw points in sweep region "
            f"[{s_lo:.3f}, {s_hi:.3f}] mm for line at "
            f"step={scan_line.step_val:.4f}. "
            f"Check sweep_length or scan coverage."
        )

    sc = sweep_vals[mask]
    hc = h_vals[mask]

    # ── 2. Bin along sweep axis ──────────────────────────────────────────────
    span   = s_hi - s_lo
    n_bins = max(cfg.min_bins, int(np.ceil(span / cfg.bin_size)) + 1)
    edges  = np.linspace(s_lo, s_hi, n_bins + 1)
    labels = np.clip(np.digitize(sc, edges) - 1, 0, n_bins - 1)

    bs_list, bh_list = [], []
    for b in range(n_bins):
        sel = labels == b
        if sel.sum() == 0:
            continue
        bs_list.append(float(sc[sel].mean()))
        bh_list.append(float(hc[sel].mean()))

    bin_sweep = np.array(bs_list, dtype=float)
    bin_h     = np.array(bh_list, dtype=float)

    # Ensure strict ascending order (should already be, but guard)
    order     = np.argsort(bin_sweep)
    bin_sweep = bin_sweep[order]
    bin_h     = bin_h[order]

    n_pop = len(bin_sweep)
    if n_pop < cfg.min_bins:
        # Sparse scan line: points cluster at a few locations rather than
        # distributing evenly, so re-binning does not help.
        # Fall back: deduplicate raw points by rounding to 3 decimal places
        # and use them directly as spline knots.
        import warnings as _w
        _w.warn(
            f"Only {n_pop} populated bins at bin_size={cfg.bin_size:.3f} mm "
            f"for line at step={scan_line.step_val:.4f} "
            f"({n_raw} pts, span={span:.2f} mm). "
            f"Falling back to deduplicated raw points for spline fit.",
            UserWarning, stacklevel=2,
        )
        # Round sweep to 3 dp and average h within each unique position
        sc_r = np.round(sc, 3)
        unique_s = np.unique(sc_r)
        bs_raw, bh_raw = [], []
        for us in unique_s:
            mask_u = sc_r == us
            bs_raw.append(float(us))
            bh_raw.append(float(hc[mask_u].mean()))
        bin_sweep = np.array(bs_raw, dtype=float)
        bin_h     = np.array(bh_raw, dtype=float)
        n_pop     = len(bin_sweep)
        if n_pop < cfg.min_bins:
            raise RuntimeError(
                f"Only {n_pop} unique sweep positions for sparse line at "
                f"step={scan_line.step_val:.4f}. "
                f"Check that scan coverage overlaps the code path."
            )

    # ── 3. Smooth binned h ───────────────────────────────────────────────────
    w = min(cfg.smooth_window, n_pop)
    if w > 1:
        # Ensure odd window for symmetric smoothing
        if w % 2 == 0:
            w -= 1
        bin_h_smooth = uniform_filter1d(bin_h, size=w)
    else:
        bin_h_smooth = bin_h.copy()

    # ── 4. Fit CubicSpline ───────────────────────────────────────────────────
    spline = CubicSpline(bin_sweep, bin_h_smooth)

    return FittedSurface(
        step_val    = scan_line.step_val,
        sweep_axis  = sax,
        sweep_start = float(bin_sweep[0]),
        sweep_end   = float(bin_sweep[-1]),
        spline      = spline,
        n_pts_raw   = n_raw,
        n_bins      = n_pop,
        bin_sweep   = bin_sweep,
        bin_h       = bin_h_smooth,
    )


# ── Batch fit ─────────────────────────────────────────────────────────────────

def fit_all(
    loaded_scan: "LoadedScan",
    cfg: SurfaceFitConfig | None = None,
) -> list[FittedSurface]:
    """
    Fit surface splines for all forward scan lines.

    Returns a list in the same order as loaded_scan.lines (ascending step_val).
    Used by z_conformer to build the surface model once before processing all
    print blocks.

    Parameters
    ----------
    loaded_scan : result of scan_loader.load_scan()
    cfg         : SurfaceFitConfig; uses defaults if None

    Returns
    -------
    list[FittedSurface] — one per scan line
    """
    if cfg is None:
        cfg = SurfaceFitConfig()

    surfaces: list[FittedSurface] = []
    for ln in loaded_scan.lines:
        surf = fit_line(ln, cfg)
        surfaces.append(surf)
        print(f"[fit]    {surf.summary()}")

    print(f"[fit]    {len(surfaces)} surface(s) fitted  "
          f"(bin={cfg.bin_size}mm  smooth={cfg.smooth_window})")
    return surfaces


# ── Lookup helper ─────────────────────────────────────────────────────────────

def surface_for_step(
    surfaces: list[FittedSurface],
    step_position: float,
) -> FittedSurface:
    """
    Return the FittedSurface whose step_val is closest to step_position.

    Used by z_conformer: for each print block, find the scan line whose
    step position best matches the code block's step position (in scan-space,
    after origin offset has been applied).
    """
    dists = [abs(s.step_val - step_position) for s in surfaces]
    return surfaces[int(np.argmin(dists))]


# ── CLI self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scan_loader import load_scan

    rng = np.random.default_rng(7)
    N   = 400   # dense scan

    # Synthetic surface: 5 scan lines at X=0..4, sweeping Y=-40→0
    # True surface: h = 0.005*sin(Y/4) + small noise  (5µm amplitude)
    def true_h(y):
        return 0.005 * np.sin(y / 4.0)

    rows = []
    for step in range(5):
        y = np.linspace(-40, 0, N)
        z_raw = -10.0 + true_h(y) + rng.normal(0, 5e-5, N)   # 50nm noise
        rows.append(np.column_stack([np.full(N, float(step)), y, z_raw]))

    data = np.vstack(rows)
    f    = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode="w")
    np.savetxt(f.name, data, delimiter=",", fmt="%.7f")
    f.close()

    sc = load_scan(f.name, "path")

    # ── Basic fit ──────────────────────────────────────────────────────────
    cfg = SurfaceFitConfig(bin_size=0.5, smooth_window=7)
    surfs = fit_all(sc, cfg)
    assert len(surfs) == 5, f"Expected 5 surfaces, got {len(surfs)}"
    print("PASS  fit_all returns 5 surfaces")

    # ── h_at accuracy (should track true_h within a few µm) ───────────────
    test_s = np.linspace(-38, -2, 50)
    for surf, ln in zip(surfs, sc.lines):
        h_fit  = surf.h_at_array(test_s)
        h_true = true_h(test_s)
        mae_um = float(np.mean(np.abs(h_fit - h_true))) * 1e3
        assert mae_um < 5.0, \
            f"Line {ln.index}: fit MAE = {mae_um:.2f} µm (> 5 µm threshold)"
    print(f"PASS  h_at accuracy < 5 µm MAE vs true surface")

    # ── Residuals ──────────────────────────────────────────────────────────
    res = surfs[0].residuals(sc.lines[0])
    assert abs(res["mean"]) < 1.0,  f"mean residual too large: {res['mean']:.3f} µm"
    assert res["std"]  < 1.0,       f"std residual too large:  {res['std']:.3f} µm"
    print(f"PASS  residuals  mean={res['mean']:+.3f}µm  std={res['std']:.3f}µm")

    # ── surface_for_step lookup ────────────────────────────────────────────
    found = surface_for_step(surfs, step_position=2.0)
    assert abs(found.step_val - 2.0) < 0.1
    print(f"PASS  surface_for_step  step_val={found.step_val:.4f}")

    # ── sweep_length crop ──────────────────────────────────────────────────
    cfg_crop = SurfaceFitConfig(bin_size=0.5, smooth_window=7, sweep_length=20.0)
    surf_crop = fit_line(sc.lines[0], cfg_crop)
    extent = surf_crop.sweep_end - surf_crop.sweep_start
    assert abs(extent - 20.0) < 1.0, \
        f"Cropped extent {extent:.2f} mm, expected ~20 mm"
    print(f"PASS  sweep_length crop  extent={extent:.2f} mm")

    # ── Scalar h_at ────────────────────────────────────────────────────────
    h_scalar = surfs[2].h_at(-20.0)
    h_vec    = surfs[2].h_at_array(np.array([-20.0]))[0]
    assert abs(h_scalar - h_vec) < 1e-12
    print("PASS  h_at scalar / array consistency")

    print()
    print("All tests passed.")
    os.unlink(f.name)
