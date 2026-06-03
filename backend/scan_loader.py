"""
backend/scan_loader.py
======================
Loads XYZ point cloud files from laser surface scans and returns
a structured, Z-normalised representation ready for surface fitting
and 2D/3D visualisation.

Public API
----------
    scan = load_scan("surface.xyz", scan_type="path")

    scan.n_lines               number of forward scan lines
    scan.lines[i]              ScanLine — one forward pass
    scan.lines[i].h            (N,) normalised heights  h = Z_raw - Z_datum
    scan.first_point_xy        (x, y) of first scan point — used by origin_matcher
    scan.z_datum_raw           raw Z of datum (mm)
    scan.sweep_axis / .step_axis
    scan.interleaved           True if forward+return detected and stripped

Z sign convention
-----------------
    The laser points downward. Raw scan Z values are negative absolute
    distances from the laser origin:

        more negative  →  surface is physically lower (farther from laser)
        less negative  →  surface is physically higher (closer to laser)

    Datum: mean raw Z of all points within the first DATUM_WINDOW_MM (10 µm)
    along the sweep direction of the first forward scan line.

        h = Z_raw - Z_datum
        h > 0  →  surface rose toward laser (bump above datum)
        h < 0  →  surface fell away  (dip below datum)

    All toolpath Z corrections operate on h, not on raw Z.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

DATUM_WINDOW_MM   = 0.010   # 10 µm — points within this distance of sweep start
DATUM_MIN_PTS     = 3       # minimum points in datum window before fallback
DATUM_FALLBACK_N  = 5       # use first N points if window too sparse

CLUSTER_TOL       = 0.05    # mm — step-axis tolerance for line clustering
MIN_SWEEP_SPAN    = 5.0     # mm — minimum sweep span to qualify as a scan line
MIN_POINTS        = 50      # minimum point count to qualify as a scan line


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ScanLine:
    """
    One forward scan pass.

    Attributes
    ----------
    index          : 0-based index in the forward-lines list (step-axis order)
    step_val       : mean position along the step axis (mm) — used for surface lookup
    sweep_axis     : 0=X, 1=Y
    step_axis      : 0=X, 1=Y
    points         : (N, 3) raw XYZ sorted ASCENDING by sweep axis (for spline fitting)
    h              : (N,) normalised heights  h = Z_raw - Z_datum  (mm), same order
    scan_start_xy  : (x, y) of the FIRST point in the original file order
                     (the physical scan start — used for origin matching, NOT points[0])
    """
    index:         int
    step_val:      float
    sweep_axis:    int
    step_axis:     int
    points:        np.ndarray   # (N, 3) sweep-axis ascending
    h:             np.ndarray   # (N,)   normalised Z heights
    scan_start_xy: tuple[float, float] = (0.0, 0.0)

    # ── Convenience accessors ──────────────────────────────────────────────

    @property
    def sweep_coords(self) -> np.ndarray:
        """Sweep-axis positions for all points (1-D array, ascending)."""
        return self.points[:, self.sweep_axis]

    @property
    def step_coords(self) -> np.ndarray:
        """Step-axis positions for all points (1-D array)."""
        return self.points[:, self.step_axis]

    @property
    def raw_z(self) -> np.ndarray:
        """Raw Z values (negative, laser-distance convention)."""
        return self.points[:, 2]

    @property
    def xy(self) -> np.ndarray:
        """(N, 2) XY positions (sweep-axis ascending order)."""
        return self.points[:, :2]

    @property
    def sweep_start(self) -> float:
        """Minimum sweep-axis value (ascending sort — smallest value)."""
        return float(self.points[0, self.sweep_axis])

    @property
    def sweep_end(self) -> float:
        """Maximum sweep-axis value (ascending sort — largest value)."""
        return float(self.points[-1, self.sweep_axis])

    @property
    def sweep_direction(self) -> int:
        """
        +1 if the physical scan ran in the +sweep direction, -1 if -sweep.
        Determined from scan_start_xy relative to the sweep extent midpoint.
        """
        mid = 0.5 * (self.sweep_start + self.sweep_end)
        start_sweep = (self.scan_start_xy[0] if self.sweep_axis == 0
                       else self.scan_start_xy[1])
        return 1 if start_sweep <= mid else -1

    @property
    def first_xy(self) -> tuple[float, float]:
        """
        (x, y) of the physical scan start — used for origin matching.
        This is the file-order first point, preserved before any sorting.
        """
        return self.scan_start_xy

    @property
    def n_points(self) -> int:
        return len(self.points)


@dataclass
class LoadedScan:
    """
    Complete structured representation of a loaded XYZ scan.

    Contains only the forward scan lines (return passes stripped if detected).
    All heights are normalised relative to Z_datum.

    Attributes
    ----------
    filepath    : original file path
    scan_type   : 'path' or 'area' (user-declared at load time)
    sweep_axis  : 0=X, 1=Y — the long axis each line sweeps along
    step_axis   : 0=X, 1=Y — the short axis that steps between lines
    interleaved : True if forward+return passes were detected and stripped
    z_datum_raw : raw Z of the datum point (mm, negative)
    lines       : forward scan lines, sorted ascending by step_val
    n_raw_lines : total detected lines before forward extraction
    """
    filepath:    str
    scan_type:   str                  # 'path' | 'area'
    sweep_axis:  int
    step_axis:   int
    interleaved: bool
    z_datum_raw: float
    lines:       list[ScanLine]       = field(default_factory=list)
    n_raw_lines: int                  = 0

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def sweep_name(self) -> str:
        return ["X", "Y"][self.sweep_axis]

    @property
    def step_name(self) -> str:
        return ["X", "Y"][self.step_axis]

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def first_point_xy(self) -> tuple[float, float]:
        """
        XY of the very first point of the first forward scan line.
        Passed to origin_matcher to compute the code → scan offset.
        """
        if not self.lines:
            raise ValueError("No scan lines loaded.")
        return self.lines[0].first_xy

    @property
    def step_positions(self) -> list[float]:
        """Step-axis position of each forward scan line (for surface lookup)."""
        return [ln.step_val for ln in self.lines]

    def all_raw_points(self) -> np.ndarray:
        """
        All raw XYZ points stacked — for 2D top-view scatter plot (optional).
        Returns (N_total, 3).
        """
        return np.vstack([ln.points for ln in self.lines])

    def all_surface_xy(self) -> list[np.ndarray]:
        """
        Per-line (N, 2) XY arrays — for 2D top-view polyline display (default).
        """
        return [ln.xy for ln in self.lines]

    def summary(self) -> str:
        sn, tn = self.sweep_name, self.step_name
        datum_um = self.z_datum_raw * 1e3   # mm → µm
        lines = [
            f"LoadedScan  n_lines={self.n_lines}  "
            f"sweep={sn}  step={tn}  "
            f"type={self.scan_type}  "
            f"interleaved={self.interleaved}  "
            f"z_datum={datum_um:.3f} µm  "
            f"(raw={self.z_datum_raw:.6f} mm)",
        ]
        for ln in self.lines:
            h_min = float(ln.h.min())
            h_max = float(ln.h.max())
            lines.append(
                f"  [line {ln.index:2d}]  {tn}={ln.step_val:8.4f}  "
                f"{sn}:[{ln.sweep_start:8.3f} → {ln.sweep_end:8.3f}]  "
                f"{ln.n_points:6,} pts  "
                f"h=[{h_min:+.4f}, {h_max:+.4f}] mm"
            )
        return "\n".join(lines)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_xyz_raw(filepath: Path) -> np.ndarray:
    """
    Load comma-separated XYZ file.
    Returns (N, 3) float64 array [x, y, z].
    Handles both comma and whitespace delimiters.
    """
    try:
        data = np.loadtxt(str(filepath), delimiter=",")
    except ValueError:
        data = np.loadtxt(str(filepath))   # whitespace fallback

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            f"Expected a file with at least 3 columns (x, y, z); "
            f"got shape {data.shape} in '{filepath.name}'."
        )
    return data[:, :3].astype(np.float64)


def _detect_axes(data: np.ndarray) -> tuple[int, int]:
    """
    Determine sweep axis (long, varies within each line) and step axis
    (short, clusters at discrete values between lines).

    Returns (sweep_axis, step_axis) where each is 0 (X) or 1 (Y).
    """
    x_span = data[:, 0].max() - data[:, 0].min()
    y_span = data[:, 1].max() - data[:, 1].min()

    if x_span >= y_span:
        sweep_axis, step_axis = 0, 1
    else:
        sweep_axis, step_axis = 1, 0

    print(f"[scan]   axis detect: sweep={'XY'[sweep_axis]} "
          f"(span={max(x_span,y_span):.3f} mm)  "
          f"step={'XY'[step_axis]} "
          f"(span={min(x_span,y_span):.3f} mm)")
    return sweep_axis, step_axis


def _detect_scan_lines(data: np.ndarray,
                        step_axis: int,
                        cluster_tol: float = CLUSTER_TOL,
                        min_sweep_span: float = MIN_SWEEP_SPAN,
                        min_points: int = MIN_POINTS) -> list[np.ndarray]:
    """
    Cluster points into scan lines by their step-axis value.

    U-turn arcs (intermediate step values) are filtered out by the
    min_sweep_span threshold — they have negligible sweep extent.

    Returns list of (N, 3) arrays in chronological file order.
    """
    step_vals  = data[:, step_axis]
    sweep_axis = 1 - step_axis
    s_min      = step_vals.min()
    labels     = np.round((step_vals - s_min) / cluster_tol).astype(int)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        groups[lbl].append(idx)

    raw_lines = []
    for indices in groups.values():
        idx_arr    = np.array(indices)
        sub        = data[idx_arr]
        sweep_span = sub[:, sweep_axis].max() - sub[:, sweep_axis].min()
        if len(sub) >= min_points and sweep_span >= min_sweep_span:
            idx_sorted = idx_arr[np.argsort(idx_arr)]   # chronological order
            raw_lines.append((int(idx_sorted[0]), data[idx_sorted]))

    raw_lines.sort(key=lambda t: t[0])
    return [arr for _, arr in raw_lines]


def _extract_forward_lines(all_lines: list[np.ndarray],
                            sweep_axis: int) -> tuple[list[np.ndarray], bool]:
    """
    Extract forward scan passes; silently discard return passes.

    Detection: if consecutive lines alternate sweep direction
    (+sweep / -sweep / +sweep ...), the file has interleaved fwd+return
    passes — take every other line (indices 0, 2, 4, ...).
    Otherwise assume all lines are forward passes.

    Returns (forward_lines, interleaved_flag).
    """
    if len(all_lines) < 2:
        return all_lines, False

    directions = [
        1 if line[-1, sweep_axis] > line[0, sweep_axis] else -1
        for line in all_lines
    ]
    interleaved = all(
        directions[i] != directions[i + 1]
        for i in range(len(directions) - 1)
    )

    if interleaved:
        forward = [all_lines[i] for i in range(0, len(all_lines), 2)]
    else:
        forward = all_lines

    return forward, interleaved


def _compute_datum(first_line: np.ndarray,
                   sweep_axis: int,
                   window_mm: float = DATUM_WINDOW_MM,
                   min_pts: int = DATUM_MIN_PTS,
                   fallback_n: int = DATUM_FALLBACK_N) -> float:
    """
    Compute the Z datum from the first DATUM_WINDOW_MM (10 µm) of the
    first forward scan line along the sweep direction.

    If fewer than min_pts points fall in the window (sparse scan),
    fall back to the mean of the first fallback_n points.

    Returns raw Z datum value (mm, negative laser-distance convention).
    """
    sweep_vals   = first_line[:, sweep_axis]
    sweep_start  = float(sweep_vals[0])
    window_mask  = np.abs(sweep_vals - sweep_start) <= window_mm
    datum_pts    = first_line[window_mask, 2]

    if len(datum_pts) < min_pts:
        n_use    = min(fallback_n, len(first_line))
        datum_pts = first_line[:n_use, 2]
        print(f"[scan]   datum window sparse ({len(datum_pts)} pts < {min_pts}); "
              f"using first {n_use} points as fallback")
    else:
        print(f"[scan]   datum: {len(datum_pts)} points in first "
              f"{window_mm*1e3:.0f} µm window")

    return float(np.mean(datum_pts))


def _sort_forward_lines_by_step(lines: list[np.ndarray],
                                 step_axis: int) -> list[np.ndarray]:
    """
    Sort forward lines in ascending step-axis order.
    This matches the natural print order in the code.txt
    (step value increases line by line).
    """
    return sorted(lines, key=lambda ln: float(ln[:, step_axis].mean()))


# ── Public entry point ─────────────────────────────────────────────────────────

def load_scan(filepath: str | Path,
              scan_type: str = "path") -> LoadedScan:
    """
    Load and structure an XYZ surface scan file.

    Parameters
    ----------
    filepath  : path to the comma-separated .xyz file
    scan_type : 'path' (current) or 'area' (future)
                Declared by the user in the load-files step.

    Returns
    -------
    LoadedScan with normalised heights and structured scan lines.

    Raises
    ------
    FileNotFoundError : file does not exist
    ValueError        : file format invalid, or fewer than 1 line detected
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Scan file not found: {path}")

    scan_type = scan_type.lower().strip()
    if scan_type not in ("path", "area"):
        raise ValueError(f"scan_type must be 'path' or 'area'; got '{scan_type}'.")

    print(f"[scan]   loading '{path.name}'  (type={scan_type})")

    # ── 1. Load raw point cloud ──────────────────────────────────────────────
    data = _load_xyz_raw(path)
    print(f"[scan]   {len(data):,} points loaded")

    # ── 2. Detect axes ───────────────────────────────────────────────────────
    sweep_axis, step_axis = _detect_axes(data)

    # ── 3. Cluster into scan lines ───────────────────────────────────────────
    all_lines = _detect_scan_lines(data, step_axis)
    n_raw     = len(all_lines)
    print(f"[scan]   {n_raw} scan line(s) clustered")

    if n_raw < 1:
        raise ValueError(
            f"No valid scan lines detected in '{path.name}'. "
            "Check MIN_SWEEP_SPAN / MIN_POINTS thresholds."
        )

    # ── 4. Extract forward lines ─────────────────────────────────────────────
    forward_raw, interleaved = _extract_forward_lines(all_lines, sweep_axis)
    mode = "alternating fwd+return → stripped to fwd only" if interleaved \
           else "all same direction → using all lines"
    print(f"[scan]   {len(forward_raw)} forward line(s)  [{mode}]")

    # ── 5. Sort forward lines ascending by step-axis value ───────────────────
    forward_sorted = _sort_forward_lines_by_step(forward_raw, step_axis)

    # ── 6. Compute Z datum from first 10 µm of first forward line ────────────
    z_datum = _compute_datum(forward_sorted[0], sweep_axis)
    print(f"[scan]   z_datum_raw = {z_datum:.6f} mm  "
          f"({z_datum*1e3:.3f} µm from laser origin)")

    # ── 7. Build ScanLine objects with normalised heights ────────────────────
    scan_lines: list[ScanLine] = []
    for idx, raw_pts in enumerate(forward_sorted):
        # Preserve file-order first point BEFORE any reordering.
        # This is the physical scan start — used by origin_matcher.
        scan_start_xy = (float(raw_pts[0, 0]), float(raw_pts[0, 1]))

        # Sort points ascending by sweep axis — required for CubicSpline fitting.
        sweep_order = np.argsort(raw_pts[:, sweep_axis])
        pts_ordered = raw_pts[sweep_order]

        h         = pts_ordered[:, 2] - z_datum   # normalised height (mm)
        step_mean = float(pts_ordered[:, step_axis].mean())

        scan_lines.append(ScanLine(
            index         = idx,
            step_val      = step_mean,
            sweep_axis    = sweep_axis,
            step_axis     = step_axis,
            points        = pts_ordered,
            h             = h,
            scan_start_xy = scan_start_xy,
        ))

    # ── 8. Assemble and return ───────────────────────────────────────────────
    result = LoadedScan(
        filepath    = str(path),
        scan_type   = scan_type,
        sweep_axis  = sweep_axis,
        step_axis   = step_axis,
        interleaved = interleaved,
        z_datum_raw = z_datum,
        lines       = scan_lines,
        n_raw_lines = n_raw,
    )

    print(f"[scan]   done")
    print(result.summary())
    return result


# ── Utility: nearest scan line for a given step-axis position ─────────────────

def nearest_line(scan: LoadedScan, step_position: float) -> ScanLine:
    """
    Return the ScanLine whose step_val is closest to `step_position`.
    Used by z_conformer to look up the surface spline for a given print line.
    """
    dists = [abs(ln.step_val - step_position) for ln in scan.lines]
    return scan.lines[int(np.argmin(dists))]


# ── CLI self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scan_loader.py <scan.xyz> [path|area]")
        sys.exit(1)

    scan_type_arg = sys.argv[2] if len(sys.argv) > 2 else "path"
    sc = load_scan(sys.argv[1], scan_type=scan_type_arg)

    print()
    print(sc.summary())
    print()
    print(f"first_point_xy = {sc.first_point_xy}")
    print(f"step_positions = {[round(v,4) for v in sc.step_positions]}")
    print()
    print("Per-line h statistics (mm):")
    for ln in sc.lines:
        print(f"  line {ln.index:2d}  "
              f"mean_h={ln.h.mean():+.4f}  "
              f"std_h={ln.h.std():.4f}  "
              f"peak-to-peak={ln.h.max()-ln.h.min():.4f}")
