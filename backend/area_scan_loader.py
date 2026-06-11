"""
backend/area_scan_loader.py
===========================
Loads a PLY point cloud from a laser area scan and returns a structured,
Z-normalised representation ready for 2D surface fitting.

The scan covers the target area plus a 1 mm overhang margin on all sides.
The print origin (0, 0) in code-space maps to the scan-space point at:

    x_scan_origin = x_grid_min + EDGE_MARGIN_MM
    y_scan_origin = y_grid_min + EDGE_MARGIN_MM

This is the same single-point-correspondence used by origin_matcher for
path scans — the OriginMatch offset is:

    dx = x_scan_origin - code_first_print_x
    dy = y_scan_origin - code_first_print_y

Z sign convention
-----------------
Identical to scan_loader: raw Z values are negative absolute distances
from the laser origin (more negative = physically lower).

    h = Z_raw - Z_datum
    h > 0  → surface rose toward laser (bump above datum)
    h < 0  → surface fell away (dip below datum)

Datum is the mean Z of the 3×3 cell neighbourhood around the origin corner.

Public API
----------
    area = load_area_scan("surface.ply")

    area.x_unique          (Nx,) sorted X grid positions
    area.y_unique          (Ny,) sorted Y grid positions
    area.Z_grid            (Ny, Nx) raw Z values  (row=Y, col=X)
    area.H_grid            (Ny, Nx) normalised heights h = Z - z_datum
    area.z_datum_raw       scalar raw Z datum (mm)
    area.scan_origin_xy    (x, y) origin corner in scan-space
    area.x_min / x_max     scan extents before inset
    area.y_min / y_max     scan extents before inset
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

EDGE_MARGIN_MM  = 1.0    # overhang margin the scan adds on each side
GRID_PITCH_MM   = 0.1    # nominal grid spacing (used for rounding to identify cells)
DATUM_RADIUS    = 1      # cells around origin corner used for datum (±1 → 3×3 patch)


# ── Data class ─────────────────────────────────────────────────────────────────

@dataclass
class AreaScan:
    """
    Structured representation of a loaded PLY area scan.

    Attributes
    ----------
    filepath        : original file path
    x_unique        : (Nx,) sorted unique X grid positions (mm)
    y_unique        : (Ny,) sorted unique Y grid positions (mm)
    Z_grid          : (Ny, Nx) raw Z values; missing cells are NaN before fill
    H_grid          : (Ny, Nx) normalised heights  h = Z_raw - z_datum
    z_datum_raw     : scalar datum Z (mm, negative laser-distance convention)
    scan_origin_xy  : (x, y) of the inset origin corner in scan-space
    x_min / x_max   : scan grid extents (before inset)
    y_min / y_max   : scan grid extents (before inset)
    n_filled        : number of missing grid cells filled by NN interpolation
    """
    filepath:       str
    x_unique:       np.ndarray    # (Nx,)
    y_unique:       np.ndarray    # (Ny,)
    Z_grid:         np.ndarray    # (Ny, Nx)
    H_grid:         np.ndarray    # (Ny, Nx)
    z_datum_raw:    float
    scan_origin_xy: tuple[float, float]
    x_min:          float
    x_max:          float
    y_min:          float
    y_max:          float
    n_filled:       int = 0

    @property
    def nx(self) -> int:
        return len(self.x_unique)

    @property
    def ny(self) -> int:
        return len(self.y_unique)

    @property
    def pitch_x(self) -> float:
        return float(np.median(np.diff(self.x_unique)))

    @property
    def pitch_y(self) -> float:
        return float(np.median(np.diff(self.y_unique)))

    def summary(self) -> str:
        return (
            f"AreaScan  grid={self.nx}×{self.ny}  "
            f"pitch=({self.pitch_x*1e3:.0f}µm, {self.pitch_y*1e3:.0f}µm)  "
            f"X:[{self.x_unique[0]:.3f}, {self.x_unique[-1]:.3f}]  "
            f"Y:[{self.y_unique[0]:.3f}, {self.y_unique[-1]:.3f}]  "
            f"z_datum={self.z_datum_raw:.4f} mm  "
            f"origin=({self.scan_origin_xy[0]:.4f}, {self.scan_origin_xy[1]:.4f})  "
            f"filled={self.n_filled}"
        )


# ── PLY reader ─────────────────────────────────────────────────────────────────

def _read_ply(filepath: Path) -> np.ndarray:
    """
    Read ASCII PLY with x y z vertex properties.
    Returns (N, 3) float64 array.
    """
    points = []
    in_data = False
    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "end_header":
                in_data = True
                continue
            if not in_data:
                continue
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    points.append((float(parts[0]),
                                   float(parts[1]),
                                   float(parts[2])))
                except ValueError:
                    continue

    if not points:
        raise ValueError(f"No valid XYZ points found in '{filepath.name}'.")

    return np.array(points, dtype=np.float64)


# ── Grid builder ───────────────────────────────────────────────────────────────

def _build_grid(pts: np.ndarray, pitch: float = GRID_PITCH_MM
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Scatter (N,3) points onto a regular (Ny, Nx) Z grid.

    Points are assigned to the nearest grid node by rounding to `pitch`.
    When multiple points map to the same cell, their Z values are averaged.
    Missing cells are filled by nearest-neighbour propagation.

    Returns (x_unique, y_unique, Z_grid, n_filled).
    Z_grid is (Ny, Nx) with row-major Y indexing.
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # Round to grid pitch to identify cells
    x_idx = np.round(x / pitch).astype(int)
    y_idx = np.round(y / pitch).astype(int)

    x_idx_min, y_idx_min = x_idx.min(), y_idx.min()
    x_idx -= x_idx_min
    y_idx -= y_idx_min

    nx = x_idx.max() + 1
    ny = y_idx.max() + 1

    # Accumulate Z values per cell
    Z_sum   = np.zeros((ny, nx), dtype=np.float64)
    Z_count = np.zeros((ny, nx), dtype=np.int32)

    for xi, yi, zi in zip(x_idx, y_idx, z):
        Z_sum[yi, xi]   += zi
        Z_count[yi, xi] += 1

    # Mean Z where populated, NaN where empty
    with np.errstate(invalid="ignore", divide="ignore"):
        Z_grid = np.where(Z_count > 0, Z_sum / Z_count, np.nan)
    n_missing = int(np.sum(Z_count == 0))

    # Nearest-neighbour fill for missing cells
    if n_missing > 0:
        from scipy.ndimage import distance_transform_edt
        valid_mask   = ~np.isnan(Z_grid)
        _, nn_idx    = distance_transform_edt(
            ~valid_mask, return_distances=True, return_indices=True
        )
        Z_grid_filled = Z_grid.copy()
        iy, ix        = np.where(~valid_mask)
        for row, col in zip(iy, ix):
            nr, nc = nn_idx[0][row, col], nn_idx[1][row, col]
            Z_grid_filled[row, col] = Z_grid[nr, nc]
        Z_grid = Z_grid_filled

    # Reconstruct real-world coordinates for the grid axes
    x_unique = (np.arange(nx) + x_idx_min) * pitch
    y_unique = (np.arange(ny) + y_idx_min) * pitch

    return x_unique, y_unique, Z_grid, n_missing


# ── Datum computation ──────────────────────────────────────────────────────────

def _compute_datum(
    Z_grid:   np.ndarray,
    x_unique: np.ndarray,
    y_unique: np.ndarray,
    origin_x: float,
    origin_y: float,
    radius:   int = DATUM_RADIUS,
) -> float:
    """
    Compute Z datum as the mean of the grid cells within `radius` cells
    of the origin corner (x_scan_origin, y_scan_origin).

    Falls back to the single nearest cell if the patch is all NaN.
    """
    # Find the grid indices closest to the origin
    ix = int(np.argmin(np.abs(x_unique - origin_x)))
    iy = int(np.argmin(np.abs(y_unique - origin_y)))

    ix_lo = max(0, ix - radius)
    ix_hi = min(len(x_unique) - 1, ix + radius)
    iy_lo = max(0, iy - radius)
    iy_hi = min(len(y_unique) - 1, iy + radius)

    patch = Z_grid[iy_lo:iy_hi + 1, ix_lo:ix_hi + 1]
    valid = patch[~np.isnan(patch)]

    if len(valid) == 0:
        # Absolute fallback: overall mean
        return float(np.nanmean(Z_grid))

    return float(np.mean(valid))


# ── Public entry point ─────────────────────────────────────────────────────────

def load_area_scan(
    filepath:         str | Path,
    edge_margin_mm:   float = EDGE_MARGIN_MM,
    grid_pitch_mm:    float = GRID_PITCH_MM,
) -> AreaScan:
    """
    Load and structure a PLY area scan file.

    Parameters
    ----------
    filepath       : path to the ASCII PLY file
    edge_margin_mm : overhang margin the scan adds on each side (default 1.0 mm)
    grid_pitch_mm  : nominal grid spacing for cell assignment (default 0.1 mm)

    Returns
    -------
    AreaScan with normalised H_grid and scan_origin_xy set to the inset corner.

    Raises
    ------
    FileNotFoundError : file does not exist
    ValueError        : file cannot be parsed or too few points
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Area scan file not found: {path}")

    print(f"[area_scan]  loading '{path.name}'")

    # 1. Read raw points
    pts = _read_ply(path)
    print(f"[area_scan]  {len(pts):,} points read")

    if len(pts) < 4:
        raise ValueError(f"Too few points ({len(pts)}) in '{path.name}'.")

    # 2. Build regular grid
    x_unique, y_unique, Z_grid, n_filled = _build_grid(pts, pitch=grid_pitch_mm)
    print(f"[area_scan]  grid {len(x_unique)}×{len(y_unique)}  "
          f"pitch={grid_pitch_mm*1e3:.0f}µm  "
          f"missing filled={n_filled}")

    x_min, x_max = float(x_unique[0]),  float(x_unique[-1])
    y_min, y_max = float(y_unique[0]),  float(y_unique[-1])

    # 3. Origin corner = grid min + edge margin
    x_origin = x_min + edge_margin_mm
    y_origin = y_min + edge_margin_mm
    print(f"[area_scan]  scan_origin = ({x_origin:.4f}, {y_origin:.4f}) mm  "
          f"[grid_min + {edge_margin_mm} mm inset]")

    # 4. Z datum from neighbourhood of origin corner
    z_datum = _compute_datum(Z_grid, x_unique, y_unique, x_origin, y_origin)
    print(f"[area_scan]  z_datum_raw = {z_datum:.6f} mm  "
          f"({z_datum*1e3:.3f} µm from laser origin)")

    # 5. Normalised height grid
    H_grid = Z_grid - z_datum

    result = AreaScan(
        filepath       = str(path),
        x_unique       = x_unique,
        y_unique       = y_unique,
        Z_grid         = Z_grid,
        H_grid         = H_grid,
        z_datum_raw    = z_datum,
        scan_origin_xy = (x_origin, y_origin),
        x_min          = x_min,
        x_max          = x_max,
        y_min          = y_min,
        y_max          = y_max,
        n_filled       = n_filled,
    )

    print(f"[area_scan]  done")
    print(result.summary())
    return result
