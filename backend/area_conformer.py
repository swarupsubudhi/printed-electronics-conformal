"""
backend/area_conformer.py
=========================
Applies surface conforming to a parsed nScrypt toolpath using a 2D area scan
surface model (AreaFittedSurface).

This is the area-scan counterpart of z_conformer.conform().
It takes the same inputs minus LoadedScan/FittedSurface (replaced by
AreaFittedSurface) and produces an identical ConformResult.

Algorithm per print block
-------------------------
1.  Map each code print waypoint to scan-space via OriginMatch.
2.  Query AreaFittedSurface.h_at(x_scan, y_scan) at every waypoint.
    No per-line grouping; the 2D surface handles all directions.
3.  Z-rate enforcement: for each segment, compute
        dh/ds = |Δh| / |Δxy|
        rate  = dh/ds * print_speed   [mm/s]
    Bisect any segment where rate > max_z_rate.
4.  Inject correctional δZ before TrigValveRel (h_start − h_prev_end
    − prev_travel_net_z) — identical accounting to z_conformer.
5.  Emit header + conformed blocks + verbatim travel segments + footer.

Public API
----------
    result = conform_area(parsed_code, area_surface, origin_match, cfg)

    result.lines          list[str]  — output file lines (no trailing newlines)
    result.block_stats    list[BlockStat]
    result.warnings       list[str]
    result.write(path)    write to disk (same as ConformResult.write)
"""

from __future__ import annotations

import warnings as _warnings
from typing import TYPE_CHECKING

import numpy as np

# Re-use ConformConfig, ConformResult, BlockStat, _fmt, Z_SUPPRESS_THRESH
# from z_conformer (same output contract, no duplication).
from z_conformer import (
    ConformConfig,
    ConformResult,
    BlockStat,
    _fmt,
    Z_SUPPRESS_THRESH,
    DEFAULT_CLEARANCE_MM,
)

if TYPE_CHECKING:
    from code_parser       import ParsedCode
    from origin_matcher    import OriginMatch
    from area_surface_fit  import AreaFittedSurface


# ── Z-rate enforcement (2D) ────────────────────────────────────────────────────

def _enforce_z_rate_2d(
    surface:     "AreaFittedSurface",
    s_xy:        list[tuple[float, float]],  # scan-space (x, y) per waypoint
    h_pts:       list[float],
    print_speed: float,
    max_z_rate:  float,
    max_bisect:  int = 2000,
) -> tuple[list[tuple[float, float]], list[float], int, bool]:
    """
    Bisect any segment whose Z-rate exceeds max_z_rate.

    Z-rate for segment i→i+1:
        ds    = sqrt((x[i+1]-x[i])² + (y[i+1]-y[i])²)
        dt    = ds / print_speed
        rate  = |h[i+1] - h[i]| / dt

    Mid-point h is queried from the surface.

    Returns (s_xy_final, h_final, n_bisected, constraint_satisfied).
    """
    s_xy  = list(s_xy)
    h_pts = list(h_pts)
    n_bisected = 0

    for _ in range(max_bisect):
        worst_rate = 0.0
        worst_i    = -1

        for i in range(len(s_xy) - 1):
            xi, yi   = s_xy[i]
            xj, yj   = s_xy[i + 1]
            ds       = np.hypot(xj - xi, yj - yi)
            if ds < 1e-9:
                continue
            dh   = abs(h_pts[i + 1] - h_pts[i])
            dt   = ds / print_speed
            rate = dh / dt
            if rate > worst_rate:
                worst_rate = rate
                worst_i    = i

        if worst_rate <= max_z_rate:
            break

        # Bisect worst segment
        xi, yi = s_xy[worst_i]
        xj, yj = s_xy[worst_i + 1]
        xm, ym = 0.5 * (xi + xj), 0.5 * (yi + yj)
        hm     = surface.h_at(xm, ym)

        s_xy.insert(worst_i + 1, (xm, ym))
        h_pts.insert(worst_i + 1, hm)
        n_bisected += 1
    else:
        return s_xy, h_pts, n_bisected, False

    return s_xy, h_pts, n_bisected, True


# ── Per-block emitter ──────────────────────────────────────────────────────────

def _emit_block_area(
    block:           "PrintBlock",
    surface:         "AreaFittedSurface",
    match:           "OriginMatch",
    h_prev_end:      float,
    prev_travel_net: float,
    cfg:             ConformConfig,
) -> tuple[list[str], float, float, dict, int, bool]:
    """
    Emit conformed nScrypt lines for one print block using 2D surface lookup.

    Returns
    -------
    (lines, h_block_end, dz_corr, error_stats, n_bisected, constraint_ok)
    """
    dp = cfg.decimals

    # ── 1. Collect all waypoints (block start + each move) in scan-space ─────
    abs_pts_code = [(block.abs_start[0], block.abs_start[1])]
    for mv in block.moves:
        abs_pts_code.append((mv.abs_x, mv.abs_y))

    s_xy   = [match.to_scan(cx, cy) for cx, cy in abs_pts_code]
    h_pts  = [surface.h_at(sx, sy)  for sx, sy in s_xy]

    # ── 2. Z-rate enforcement ─────────────────────────────────────────────────
    s_xy, h_pts, n_bisected, constraint_ok = _enforce_z_rate_2d(
        surface     = surface,
        s_xy        = s_xy,
        h_pts       = h_pts,
        print_speed = cfg.print_speed,
        max_z_rate  = cfg.max_z_rate,
    )

    # ── 3. Build output lines ─────────────────────────────────────────────────
    out = list(block.preamble_raw)

    # Replace trigwait value
    out = [
        (f"trigwait {_fmt(cfg.trigwait_time, dp)}"
         if ln.strip().lower().startswith("trigwait") else ln)
        for ln in out
    ]

    # Correctional δZ: accounts for surface height change AND preceding travel
    h_start  = float(h_pts[0])
    dz_corr  = h_start - h_prev_end - prev_travel_net

    if abs(dz_corr) > Z_SUPPRESS_THRESH:
        tv_idx = next(
            (i for i in reversed(range(len(out)))
             if out[i].strip().lower() == "trigvalverel"),
            None,
        )
        if tv_idx is not None:
            corr_line = (
                f"move  0  0  {_fmt(dz_corr, dp)}"
                f"  / Z correction ({dz_corr*1e3:+.2f} µm)"
            )
            out.insert(tv_idx, corr_line)

    # ── 4. Recover code-space XY for bisected segments ────────────────────────
    # Original code XY positions (N+1 points: start + moves)
    orig_code_x = np.array([block.abs_start[0]] + [m.abs_x for m in block.moves])
    orig_code_y = np.array([block.abs_start[1]] + [m.abs_y for m in block.moves])
    orig_s_xy   = [match.to_scan(cx, cy)
                   for cx, cy in zip(orig_code_x, orig_code_y)]

    # Parameterise original path by cumulative Euclidean distance
    orig_ds  = [0.0]
    for i in range(1, len(orig_s_xy)):
        d = np.hypot(orig_s_xy[i][0] - orig_s_xy[i-1][0],
                     orig_s_xy[i][1] - orig_s_xy[i-1][1])
        orig_ds.append(orig_ds[-1] + d)
    orig_t = np.array(orig_ds)

    # Final path parameter values
    final_ds = [0.0]
    for i in range(1, len(s_xy)):
        d = np.hypot(s_xy[i][0] - s_xy[i-1][0],
                     s_xy[i][1] - s_xy[i-1][1])
        final_ds.append(final_ds[-1] + d)
    final_t = np.array(final_ds)

    # Interpolate original code XY onto final parameter values
    if orig_t[-1] > 1e-9:
        t_norm      = orig_t / orig_t[-1]
        ft_norm     = final_t / final_t[-1] if final_t[-1] > 1e-9 else final_t
        code_x_new  = np.interp(ft_norm, t_norm, orig_code_x)
        code_y_new  = np.interp(ft_norm, t_norm, orig_code_y)
    else:
        code_x_new = np.full(len(s_xy), orig_code_x[0])
        code_y_new = np.full(len(s_xy), orig_code_y[0])

    # ── 5. Emit moves ─────────────────────────────────────────────────────────
    cur_z = cfg.clearance_mm + h_start

    for idx in range(len(s_xy)):
        hi       = float(h_pts[idx])
        z_target = cfg.clearance_mm + hi

        if idx == 0:
            dx, dy, dz = 0.0, 0.0, 0.0
        else:
            dx  = code_x_new[idx] - code_x_new[idx - 1]
            dy  = code_y_new[idx] - code_y_new[idx - 1]
            dz  = z_target - cur_z
            if abs(dz) < Z_SUPPRESS_THRESH:
                dz = 0.0

        out.append(f"move  {_fmt(dx, dp)}  {_fmt(dy, dp)}  {_fmt(dz, dp)}")
        cur_z = z_target

    h_block_end = float(h_pts[-1])

    # ── 6. Error statistics ───────────────────────────────────────────────────
    # Error = piecewise-linear interpolated h − true surface h, densely sampled
    all_err: list[float] = []
    for i in range(len(s_xy) - 1):
        xi, yi = s_xy[i]
        xj, yj = s_xy[i + 1]
        for t in np.linspace(0, 1, 20, endpoint=False):
            xm = xi + t * (xj - xi)
            ym = yi + t * (yj - yi)
            h_lin  = h_pts[i] + t * (h_pts[i + 1] - h_pts[i])
            h_true = surface.h_at(xm, ym)
            all_err.append((h_lin - h_true) * 1e3)   # mm → µm

    err_arr = np.array(all_err) if all_err else np.array([0.0])
    error_stats = {
        "mean": float(np.mean(err_arr)),
        "std":  float(np.std(err_arr)),
        "min":  float(np.min(err_arr)),
        "max":  float(np.max(err_arr)),
        "rms":  float(np.sqrt(np.mean(err_arr ** 2))),
    }

    return out, h_block_end, dz_corr, error_stats, n_bisected, constraint_ok


# ── Public entry point ─────────────────────────────────────────────────────────

def conform_area(
    parsed_code:  "ParsedCode",
    area_surface: "AreaFittedSurface",
    origin_match: "OriginMatch",
    cfg:          ConformConfig | None = None,
) -> ConformResult:
    """
    Apply 2D surface conforming to all print blocks.

    Parameters
    ----------
    parsed_code  : from code_parser.parse()
    area_surface : from area_surface_fit.fit_area()
    origin_match : OriginMatch with dx/dy mapping code-space → scan-space
                   (compute with origin_matcher.compute_offset_area())
    cfg          : ConformConfig; uses defaults if None

    Returns
    -------
    ConformResult — identical contract to z_conformer.conform().
    """
    if cfg is None:
        cfg = ConformConfig()

    result = ConformResult()
    warns  = result.warnings

    # ── 1. Header ─────────────────────────────────────────────────────────────
    out: list[str] = list(parsed_code.header_raw)

    clearance_delta = cfg.clearance_mm - DEFAULT_CLEARANCE_MM
    if abs(clearance_delta) > Z_SUPPRESS_THRESH:
        out.append(
            f"move  0  0  {_fmt(clearance_delta, cfg.decimals)}"
            f"  / clearance offset ({clearance_delta*1e3:+.1f} µm)"
        )

    # ── 2. Process each print block ───────────────────────────────────────────
    h_prev_end      = 0.0
    prev_travel_net = 0.0

    for i, block in enumerate(parsed_code.blocks):
        try:
            block_lines, h_block_end, dz_corr, err_stats, n_bis, ok = \
                _emit_block_area(
                    block           = block,
                    surface         = area_surface,
                    match           = origin_match,
                    h_prev_end      = h_prev_end,
                    prev_travel_net = prev_travel_net,
                    cfg             = cfg,
                )
        except Exception as exc:
            msg = f"Block '{block.label}': area conform failed — {exc}"
            warns.append(msg)
            _warnings.warn(msg, RuntimeWarning)
            # Fallback: pass through original block unchanged
            out.extend(block.preamble_raw)
            for mv in block.moves:
                out.append(mv.raw)
            if i < len(parsed_code.travels):
                out.extend(parsed_code.travels[i].raw_lines)
            h_prev_end      = 0.0
            prev_travel_net = 0.0
            continue

        if not ok:
            warns.append(
                f"Block '{block.label}': Z-rate constraint not fully satisfied. "
                f"Consider reducing print_speed."
            )

        out.extend(block_lines)

        result.block_stats.append(BlockStat(
            label                = block.label,
            n_moves_original     = block.n_moves,
            n_moves_final        = len(block_lines) - len(block.preamble_raw) - 1,
            n_bisected           = n_bis,
            constraint_satisfied = ok,
            dz_correction_mm     = dz_corr,
            error_stats          = err_stats,
        ))

        h_prev_end = h_block_end

        # Travel segment: pass through verbatim, track net Z for next block
        if i < len(parsed_code.travels):
            travel = parsed_code.travels[i]
            out.extend(travel.raw_lines)
            prev_travel_net = sum(mv.dz for mv in travel.moves)
        else:
            prev_travel_net = 0.0

    # ── 3. Footer ─────────────────────────────────────────────────────────────
    out.extend(parsed_code.footer_raw)
    out.append("/ end of toolpath")

    result.lines = out
    return result


# ── Origin matcher for area scans ──────────────────────────────────────────────

def compute_offset_area(
    parsed_code: "ParsedCode",
    area_scan:   "AreaScan",
) -> "OriginMatch":
    """
    Compute the XY offset between code-space and area-scan-space.

    The scan origin is the inset corner of the grid:
        scan_origin = (x_grid_min + edge_margin, y_grid_min + edge_margin)

    This is mapped to the first print waypoint in code-space, exactly as
    origin_matcher.compute_offset does for path scans.

    Parameters
    ----------
    parsed_code : from code_parser.parse()
    area_scan   : from area_scan_loader.load_area_scan()

    Returns
    -------
    OriginMatch with dx, dy and 2D coverage validation.
    """
    from origin_matcher import OriginMatch

    code_origin  = parsed_code.first_print_abs_xy
    scan_origin  = area_scan.scan_origin_xy

    dx = scan_origin[0] - code_origin[0]
    dy = scan_origin[1] - code_origin[1]

    # Coverage check: all code print waypoints must map inside the grid
    warns: list[str] = []
    tol = 0.5   # mm

    for block in parsed_code.blocks:
        for pt in [(block.abs_start[0], block.abs_start[1]),
                   (block.abs_end[0],   block.abs_end[1])]:
            sx = pt[0] + dx
            sy = pt[1] + dy
            if sx < area_scan.x_min - tol or sx > area_scan.x_max + tol:
                warns.append(
                    f"Block '{block.label}': X={sx:.3f} outside scan "
                    f"[{area_scan.x_min:.3f}, {area_scan.x_max:.3f}]"
                )
            if sy < area_scan.y_min - tol or sy > area_scan.y_max + tol:
                warns.append(
                    f"Block '{block.label}': Y={sy:.3f} outside scan "
                    f"[{area_scan.y_min:.3f}, {area_scan.y_max:.3f}]"
                )

    match = OriginMatch(
        dx          = dx,
        dy          = dy,
        code_origin = code_origin,
        scan_origin = scan_origin,
        coverage_ok = len(warns) == 0,
        warnings    = warns,
    )

    print(f"[area_origin]  code origin  = ({code_origin[0]:.4f}, {code_origin[1]:.4f}) mm")
    print(f"[area_origin]  scan origin  = ({scan_origin[0]:.4f}, {scan_origin[1]:.4f}) mm")
    print(f"[area_origin]  offset dx={dx:+.4f} mm  dy={dy:+.4f} mm")
    if match.coverage_ok:
        print(f"[area_origin]  coverage OK")
    else:
        for w in warns:
            print(f"[area_origin]  WARNING: {w}")

    return match
