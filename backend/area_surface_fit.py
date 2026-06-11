"""
backend/area_surface_fit.py
===========================
Fits a 2D surface model to a loaded AreaScan using
scipy.interpolate.RectBivariateSpline.

The spline is fitted over the full grid and queried at arbitrary (x, y)
positions in scan-space.  Queries outside the grid are clamped to the
nearest edge rather than extrapolated.

Public API
----------
    surf = fit_area(area_scan)

    surf.h_at(x, y)          → float   single-point height query
    surf.h_at_array(xy)      → ndarray  (N,) for (N, 2) input
    surf.gradient_at(x, y)   → (dh/dx, dh/dy)  for Z-rate enforcement
    surf.x_min / x_max       clamping bounds
    surf.y_min / y_max       clamping bounds
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RectBivariateSpline

if TYPE_CHECKING:
    from area_scan_loader import AreaScan


# ── Fitted surface ─────────────────────────────────────────────────────────────

@dataclass
class AreaFittedSurface:
    """
    2D surface h(x, y) fitted from an AreaScan.

    The spline is defined over [x_min, x_max] × [y_min, y_max].
    Queries outside this domain are clamped to the boundary.

    Attributes
    ----------
    spline   : RectBivariateSpline over the full grid
    x_min / x_max : X domain of the fitted grid
    y_min / y_max : Y domain of the fitted grid
    """
    spline:  RectBivariateSpline
    x_min:   float
    x_max:   float
    y_min:   float
    y_max:   float

    # ── Clamping ───────────────────────────────────────────────────────────

    def _clamp_x(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.x_min, self.x_max)

    def _clamp_y(self, y: np.ndarray) -> np.ndarray:
        return np.clip(y, self.y_min, self.y_max)

    # ── Query methods ──────────────────────────────────────────────────────

    def h_at(self, x: float, y: float) -> float:
        """
        Return normalised surface height h (mm) at scan-space (x, y).
        Clamped to grid boundary — no extrapolation.
        """
        xc = float(np.clip(x, self.x_min, self.x_max))
        yc = float(np.clip(y, self.y_min, self.y_max))
        return float(self.spline(yc, xc, grid=False))

    def h_at_array(self, xy: np.ndarray) -> np.ndarray:
        """
        Vectorised height query.

        Parameters
        ----------
        xy : (N, 2) array of scan-space [x, y] positions

        Returns
        -------
        (N,) array of normalised heights h (mm)
        """
        xy = np.asarray(xy, dtype=float)
        xc = np.clip(xy[:, 0], self.x_min, self.x_max)
        yc = np.clip(xy[:, 1], self.y_min, self.y_max)
        # RectBivariateSpline.ev evaluates at scattered (not grid) points
        return self.spline.ev(yc, xc)

    def gradient_at(self, x: float, y: float) -> tuple[float, float]:
        """
        Return (∂h/∂x, ∂h/∂y) at scan-space (x, y).
        Used by area_conformer for Z-rate enforcement.

        The gradient magnitude along a move direction v̂ = (vx, vy) is:
            |dh/ds| = |∂h/∂x · vx + ∂h/∂y · vy|
        where vx, vy are the unit-vector components.
        """
        xc = float(np.clip(x, self.x_min, self.x_max))
        yc = float(np.clip(y, self.y_min, self.y_max))
        # dx=1 means first derivative wrt x (second argument of spline(y, x))
        dh_dx = float(self.spline(yc, xc, dx=0, dy=1, grid=False))
        dh_dy = float(self.spline(yc, xc, dx=1, dy=0, grid=False))
        return dh_dx, dh_dy

    def gradient_at_array(self, xy: np.ndarray) -> np.ndarray:
        """
        Vectorised gradient query.

        Parameters
        ----------
        xy : (N, 2) array of scan-space [x, y] positions

        Returns
        -------
        (N, 2) array of [∂h/∂x, ∂h/∂y] values
        """
        xy = np.asarray(xy, dtype=float)
        xc = np.clip(xy[:, 0], self.x_min, self.x_max)
        yc = np.clip(xy[:, 1], self.y_min, self.y_max)
        dh_dx = self.spline.ev(yc, xc, dy=1)   # dy in ev = deriv wrt x-axis arg
        dh_dy = self.spline.ev(yc, xc, dx=1)   # dx in ev = deriv wrt y-axis arg
        return np.column_stack([dh_dx, dh_dy])

    def summary(self) -> str:
        return (
            f"AreaFittedSurface  "
            f"X:[{self.x_min:.3f}, {self.x_max:.3f}]  "
            f"Y:[{self.y_min:.3f}, {self.y_max:.3f}]"
        )


# ── Public entry point ─────────────────────────────────────────────────────────

def fit_area(area_scan: "AreaScan", kx: int = 3, ky: int = 3) -> AreaFittedSurface:
    """
    Fit a RectBivariateSpline to the H_grid of an AreaScan.

    Parameters
    ----------
    area_scan : loaded AreaScan (from area_scan_loader.load_area_scan)
    kx, ky    : spline degrees in X and Y (default 3 = cubic)

    Returns
    -------
    AreaFittedSurface ready for h_at / gradient_at queries.
    """
    x = area_scan.x_unique   # (Nx,)
    y = area_scan.y_unique   # (Ny,)
    H = area_scan.H_grid     # (Ny, Nx)

    # RectBivariateSpline(x, y, z) expects z shape (Nx, Ny) with x as first axis
    # But H_grid is (Ny, Nx). We fitted as H[iy, ix], so spline is called as
    # spline(y_query, x_query) → i.e. first axis = Y, second = X.
    # RectBivariateSpline signature: RectBivariateSpline(x1, x2, z) where
    # z has shape (len(x1), len(x2)).
    # We pass (y, x, H) so spline(y_q, x_q) is natural.

    print(f"[area_fit]  fitting RectBivariateSpline  "
          f"grid={len(x)}×{len(y)}  kx={kx}  ky={ky}")

    spline = RectBivariateSpline(y, x, H, kx=ky, ky=kx)

    surf = AreaFittedSurface(
        spline = spline,
        x_min  = float(x[0]),
        x_max  = float(x[-1]),
        y_min  = float(y[0]),
        y_max  = float(y[-1]),
    )

    print(f"[area_fit]  done  {surf.summary()}")
    return surf
