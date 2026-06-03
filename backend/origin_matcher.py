"""
backend/origin_matcher.py
=========================
Computes the rigid XY offset between code-space (from code_parser) and
scan-space (from scan_loader) so that every print waypoint can be mapped
to its corresponding surface height.

Background
----------
The laser scanner and the print tool are driven by the same machine using
the same coordinate system.  They physically travel the same path, separated
only by a fixed (dx, dy) offset in the machine frame.  No rotation or scaling
is involved.

The offset is found from a single point correspondence:

    code_origin  =  first_print_abs_xy  from ParsedCode
    scan_origin  =  first_point_xy      from LoadedScan

    dx = scan_origin.x - code_origin.x
    dy = scan_origin.y - code_origin.y

All subsequent code-XY positions are mapped to scan-space with:

    scan_x = code_x + dx
    scan_y = code_y + dy

The Z axis is handled separately through datum normalisation in scan_loader —
no Z offset is needed here.

Public API
----------
    match = compute_offset(parsed_code, loaded_scan)

    match.dx, match.dy           XY offset (mm)
    match.to_scan(cx, cy)        map code → scan  (x, y)
    match.to_code(sx, sy)        map scan → code  (x, y)
    match.coverage_ok            True if all code extents lie inside scan
    match.warnings               list of human-readable coverage warnings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from code_parser import ParsedCode
    from scan_loader import LoadedScan


# ── Tolerance ─────────────────────────────────────────────────────────────────

COVERAGE_TOL_MM = 0.5   # allow code extents up to this far outside scan (mm)


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class OriginMatch:
    """
    Result of origin matching between a ParsedCode and a LoadedScan.

    Attributes
    ----------
    dx, dy        : scan_space - code_space offset (mm)
    code_origin   : (x, y) first print waypoint in code-space
    scan_origin   : (x, y) first scan point in scan-space
    coverage_ok   : True if all code print extents map inside scan coverage
    warnings      : human-readable list of any coverage issues
    """
    dx:           float
    dy:           float
    code_origin:  tuple[float, float]
    scan_origin:  tuple[float, float]
    coverage_ok:  bool               = True
    warnings:     list[str]          = field(default_factory=list)

    # ── Coordinate transforms ──────────────────────────────────────────────

    def to_scan(self, code_x: float, code_y: float) -> tuple[float, float]:
        """Map a code-space (x, y) position to scan-space."""
        return code_x + self.dx, code_y + self.dy

    def to_code(self, scan_x: float, scan_y: float) -> tuple[float, float]:
        """Map a scan-space (x, y) position back to code-space."""
        return scan_x - self.dx, scan_y - self.dy

    def to_scan_array(self, code_xy: np.ndarray) -> np.ndarray:
        """
        Vectorised version for (N, 2) arrays.

        Parameters
        ----------
        code_xy : (N, 2) array of code-space [x, y] positions

        Returns
        -------
        (N, 2) array of scan-space [x, y] positions
        """
        offset = np.array([self.dx, self.dy], dtype=float)
        return code_xy + offset

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"OriginMatch",
            f"  code origin : ({self.code_origin[0]:.4f}, {self.code_origin[1]:.4f}) mm",
            f"  scan origin : ({self.scan_origin[0]:.4f}, {self.scan_origin[1]:.4f}) mm",
            f"  offset (dx) : {self.dx:+.4f} mm",
            f"  offset (dy) : {self.dy:+.4f} mm",
            f"  coverage    : {'OK' if self.coverage_ok else 'WARNING'}",
        ]
        for w in self.warnings:
            lines.append(f"  !! {w}")
        return "\n".join(lines)


# ── Coverage check helpers ────────────────────────────────────────────────────

def _scan_extents(scan: "LoadedScan") -> dict:
    """
    Return the XY bounding box of the scan in scan-space.

    Keys: sweep_min, sweep_max, step_min, step_max
    (in the scan's own sweep/step axis convention)
    """
    sweep_axis = scan.sweep_axis
    step_axis  = scan.step_axis

    sweep_vals = np.concatenate([ln.sweep_coords for ln in scan.lines])
    step_vals  = [ln.step_val for ln in scan.lines]

    return {
        "sweep_min": float(sweep_vals.min()),
        "sweep_max": float(sweep_vals.max()),
        "step_min":  float(min(step_vals)),
        "step_max":  float(max(step_vals)),
        "sweep_axis": sweep_axis,
        "step_axis":  step_axis,
    }


def _check_coverage(
    parsed_code: "ParsedCode",
    scan: "LoadedScan",
    match: OriginMatch,
    tol: float = COVERAGE_TOL_MM,
) -> tuple[bool, list[str]]:
    """
    Verify that every code print waypoint, once mapped to scan-space,
    falls within the scan's XY extent (plus tolerance).

    Returns (coverage_ok, warning_messages).
    """
    ext    = _scan_extents(scan)
    sa     = ext["sweep_axis"]
    ta     = ext["step_axis"]
    warns: list[str] = []

    # Collect all mapped print start/end positions
    mapped_sweep: list[float] = []
    mapped_step:  list[float] = []

    for block in parsed_code.blocks:
        sx, sy = match.to_scan(*block.abs_start[:2])
        ex, ey = match.to_scan(*block.abs_end[:2])
        for x, y in [(sx, sy), (ex, ey)]:
            mapped_sweep.append(x if sa == 0 else y)
            mapped_step.append( x if ta == 0 else y)

    # Also check all individual print move positions
    for block in parsed_code.blocks:
        for mv in block.moves:
            sx, sy = match.to_scan(mv.abs_x, mv.abs_y)
            mapped_sweep.append(sx if sa == 0 else sy)
            mapped_step.append( sx if ta == 0 else sy)

    ms_arr = np.array(mapped_sweep)
    mt_arr = np.array(mapped_step)

    sweep_name = scan.sweep_name
    step_name  = scan.step_name

    if ms_arr.min() < ext["sweep_min"] - tol:
        warns.append(
            f"Code extends {abs(ms_arr.min()-ext['sweep_min']):.3f} mm "
            f"below scan sweep ({sweep_name}) minimum "
            f"({ext['sweep_min']:.3f} mm)"
        )
    if ms_arr.max() > ext["sweep_max"] + tol:
        warns.append(
            f"Code extends {abs(ms_arr.max()-ext['sweep_max']):.3f} mm "
            f"beyond scan sweep ({sweep_name}) maximum "
            f"({ext['sweep_max']:.3f} mm)"
        )
    if mt_arr.min() < ext["step_min"] - tol:
        warns.append(
            f"Code extends {abs(mt_arr.min()-ext['step_min']):.3f} mm "
            f"below scan step ({step_name}) minimum "
            f"({ext['step_min']:.3f} mm)"
        )
    if mt_arr.max() > ext["step_max"] + tol:
        warns.append(
            f"Code extends {abs(mt_arr.max()-ext['step_max']):.3f} mm "
            f"beyond scan step ({step_name}) maximum "
            f"({ext['step_max']:.3f} mm)"
        )

    return len(warns) == 0, warns


# ── Public entry point ────────────────────────────────────────────────────────

def compute_offset(
    parsed_code: "ParsedCode",
    loaded_scan: "LoadedScan",
) -> OriginMatch:
    """
    Compute the XY offset between code-space and scan-space.

    Parameters
    ----------
    parsed_code  : result of code_parser.parse()
    loaded_scan  : result of scan_loader.load_scan()

    Returns
    -------
    OriginMatch with dx, dy and coverage validation results.

    Raises
    ------
    ValueError : if either input has no data to match from
    """
    code_origin = parsed_code.first_print_abs_xy
    scan_origin = loaded_scan.first_point_xy

    dx = scan_origin[0] - code_origin[0]
    dy = scan_origin[1] - code_origin[1]

    match = OriginMatch(
        dx          = dx,
        dy          = dy,
        code_origin = code_origin,
        scan_origin = scan_origin,
    )

    # Coverage validation
    ok, warnings = _check_coverage(parsed_code, loaded_scan, match)
    match.coverage_ok = ok
    match.warnings    = warnings

    # Console output
    print(f"[origin]  code origin  = ({code_origin[0]:.4f}, {code_origin[1]:.4f}) mm")
    print(f"[origin]  scan origin  = ({scan_origin[0]:.4f}, {scan_origin[1]:.4f}) mm")
    print(f"[origin]  offset dx={dx:+.4f} mm  dy={dy:+.4f} mm")
    if ok:
        print(f"[origin]  coverage OK — all code extents inside scan")
    else:
        for w in warnings:
            print(f"[origin]  WARNING: {w}")

    return match


# ── CLI self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    import tempfile
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from code_parser import parse
    from scan_loader import load_scan

    # ── Use toolpath.txt as the code file ─────────────────────────────────
    code_path = sys.argv[1] if len(sys.argv) > 1 else "toolpath.txt"
    pc = parse(code_path)

    # ── Build a synthetic scan offset by (+5, -3) mm from code-space ─────
    rng     = np.random.default_rng(0)
    N       = 200
    OFFSET  = (5.0, -3.0)   # known offset to recover
    rows    = []
    for step_i in range(pc.n_blocks):
        # Code print lines are at X = 0,1,2,3,4; sweep Y 0→-40
        x_code = float(step_i)
        x_scan = x_code + OFFSET[0]
        y_base = pc.blocks[step_i].abs_start[1]
        y_end  = pc.blocks[step_i].abs_end[1]
        y_scan = np.linspace(y_base + OFFSET[1], y_end + OFFSET[1], N)
        z_raw  = -10.0 + 0.001 * np.sin(y_scan / 5) + rng.normal(0, 1e-4, N)
        rows.append(np.column_stack([np.full(N, x_scan), y_scan, z_raw]))

    scan_data = np.vstack(rows)
    tmp = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode="w")
    np.savetxt(tmp.name, scan_data, delimiter=",", fmt="%.6f")
    tmp.close()

    sc = load_scan(tmp.name, scan_type="path")

    # ── Compute offset ────────────────────────────────────────────────────
    match = compute_offset(pc, sc)
    print()
    print(match.summary())
    print()

    # ── Verify ────────────────────────────────────────────────────────────
    assert abs(match.dx - OFFSET[0]) < 1e-3, \
        f"dx mismatch: got {match.dx:.6f}, expected {OFFSET[0]}"
    assert abs(match.dy - OFFSET[1]) < 1e-3, \
        f"dy mismatch: got {match.dy:.6f}, expected {OFFSET[1]}"
    assert match.coverage_ok, \
        f"coverage should be OK: {match.warnings}"

    # ── Round-trip ────────────────────────────────────────────────────────
    cx, cy = pc.first_print_abs_xy
    sx, sy = match.to_scan(cx, cy)
    cx2, cy2 = match.to_code(sx, sy)
    assert abs(cx2 - cx) < 1e-9 and abs(cy2 - cy) < 1e-9, "round-trip failed"

    # ── Vectorised transform ──────────────────────────────────────────────
    pts = np.array([[0.0, 0.0], [1.0, -20.0], [4.0, -39.0]])
    mapped = match.to_scan_array(pts)
    assert mapped.shape == pts.shape
    np.testing.assert_allclose(mapped[:, 0], pts[:, 0] + match.dx)
    np.testing.assert_allclose(mapped[:, 1], pts[:, 1] + match.dy)

    print("PASS  dx recovered correctly:", round(match.dx, 4), "==", OFFSET[0])
    print("PASS  dy recovered correctly:", round(match.dy, 4), "==", OFFSET[1])
    print("PASS  coverage_ok =", match.coverage_ok)
    print("PASS  round-trip to_scan / to_code")
    print("PASS  vectorised to_scan_array")
    print()
    print("All tests passed.")

    os.unlink(tmp.name)
