"""
backend/z_conformer.py
======================
Applies surface conforming to a parsed nScrypt toolpath.

This is the heart of the conformal toolpath pipeline.  It takes the four
upstream products — ParsedCode, LoadedScan, OriginMatch, list[FittedSurface]
— and produces a new nScrypt text file in which all in-print Z moves track
the real scanned surface at the user's chosen clearance height.

Algorithm per print block
-------------------------
1.  Map each code print waypoint to scan-space via OriginMatch.
2.  Look up the matching FittedSurface by step-axis position.
3.  Extract sweep-axis positions (s_code) for all print waypoints.
4.  Run the chosen interpolation algorithm → (s_final, h_final).
5.  For ADAPTIVE_CURVATURE (which may produce NEW positions):
      map s_final back to code-space moves by proportional XY interpolation.
    For LSQ_POLY_3 and LINEAR_10PT (which preserve s_code positions):
      update only the Z value of each existing move.
6.  Compute relative ΔZ for each move (code uses cumulative relative moves).
7.  Inject correctional δZ before TrigValveRel (h_start - h_prev_block_end).
8.  Inject TrigWait before TrigValveRel.
9.  Emit header (with optional clearance delta) + all blocks + travels + footer.

Z accounting
------------
After the header:
    move 0  0  10        ; safe lift
    move 0  0  -9.990    ; plunge → nozzle at DEFAULT_CLEARANCE = 0.010 mm above datum
    [move 0  0  Δ]       ; optional: if user_clearance != DEFAULT_CLEARANCE

All subsequent Z positions are expressed relative to datum.
The nozzle target at any print waypoint is:   h_surface + user_clearance
The ΔZ for move i is:                         target[i] - target[i-1]

Public API
----------
    result = conform(parsed_code, loaded_scan, origin_match, surfaces,
                     cfg=ConformConfig(...))

    result.lines          list[str]  — output file lines (no newlines)
    result.block_stats    list[BlockStat]
    result.warnings       list[str]
    result.write(path)    write to disk
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from interpolation import (
    Algorithm, InterpolationConfig, InterpolationResult, run as interp_run
)
from surface_fit import surface_for_step

if TYPE_CHECKING:
    from code_parser  import ParsedCode, PrintBlock, TravelSegment, CodeMove
    from scan_loader  import LoadedScan
    from origin_matcher import OriginMatch
    from surface_fit  import FittedSurface


# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_CLEARANCE_MM  = 0.010   # header baseline (10 µm)
Z_SUPPRESS_THRESH     = 1e-5    # suppress |δ| < 0.01 µm (floating-point noise)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ConformConfig:
    """
    All user-adjustable settings for the conforming operation.

    Attributes
    ----------
    algorithm    : interpolation strategy
    print_speed  : mm/s  (must match code.txt speed for correct rate calc)
    max_z_rate   : mm/s  Z-axis rate limit
    clearance_mm : desired clearance above surface (mm); default 0.010
    wait_time    : seconds for Wait command (if used; currently unused in emit)
    trigwait_time: seconds for TrigWait before every TrigValveRel
    decimals     : decimal places in output move values
    n_linear     : waypoints for LINEAR_10PT
    poly_degree  : polynomial degree for LSQ_POLY_3
    n_seed       : seed points for ADAPTIVE_CURVATURE
    """
    algorithm:    Algorithm = Algorithm.ADAPTIVE_CURVATURE
    print_speed:  float     = 5.0
    max_z_rate:   float     = 1.0
    clearance_mm: float     = DEFAULT_CLEARANCE_MM
    wait_time:    float     = 0.0
    trigwait_time: float    = 0.5
    decimals:     int       = 3
    n_linear:     int       = 10
    poly_degree:  int       = 3
    n_seed:       int       = 20

    def to_interp_cfg(self) -> InterpolationConfig:
        return InterpolationConfig(
            algorithm    = self.algorithm,
            print_speed  = self.print_speed,
            max_z_rate   = self.max_z_rate,
            n_linear     = self.n_linear,
            poly_degree  = self.poly_degree,
            n_seed       = self.n_seed,
        )


# ── Per-block diagnostics ─────────────────────────────────────────────────────

@dataclass
class BlockStat:
    """Diagnostics for one print block after conforming."""
    label:                str
    n_moves_original:     int
    n_moves_final:        int
    n_bisected:           int
    constraint_satisfied: bool
    dz_correction_mm:     float    # injected δZ before TrigValveRel
    error_stats:          dict     # mean/std/min/max/rms in µm


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class ConformResult:
    """
    Complete output of conform().

    Attributes
    ----------
    lines       : list of output file lines (write with '\\r\\n'.join(lines))
    block_stats : per-block diagnostics
    warnings    : human-readable issues encountered
    """
    lines:       list[str]       = field(default_factory=list)
    block_stats: list[BlockStat] = field(default_factory=list)
    warnings:    list[str]       = field(default_factory=list)

    def write(self, filepath: str | Path, encoding: str = "utf-8") -> None:
        # Normalise before writing:
        #   - rstrip() removes stray trailing \r (prevents \r\r\n on re-runs)
        #   - drop blank lines entirely (nScrypt ignores them; the / comments
        #     already delineate sections, and prior runs injected one phantom
        #     blank per line via \r\r\n + splitlines)
        # Then write with newline="" so the explicit \r\n is preserved exactly
        # and the OS does NOT translate \n -> \r\n on top of it.
        cleaned = [ln.rstrip() for ln in self.lines]
        cleaned = [ln for ln in cleaned if ln != ""]

        text = "\r\n".join(cleaned) + "\r\n"
        Path(filepath).write_text(text, encoding=encoding, newline="")
        print(f"[conform] wrote '{filepath}'  ({len(cleaned)} lines)")

    def summary(self) -> str:
        lines = [f"ConformResult  {len(self.block_stats)} block(s)"]
        for bs in self.block_stats:
            ok = "OK" if bs.constraint_satisfied else "WARN"
            lines.append(
                f"  [{bs.label:>8}]  "
                f"moves {bs.n_moves_original}→{bs.n_moves_final}  "
                f"bisected={bs.n_bisected}  "
                f"dZ_corr={bs.dz_correction_mm*1e3:+.2f}µm  "
                f"err_mean={bs.error_stats.get('mean',0):+.3f}µm  [{ok}]"
            )
        for w in self.warnings:
            lines.append(f"  !! {w}")
        return "\n".join(lines)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _fmt(value: float, decimals: int) -> str:
    """Format a float to the required decimal places."""
    return f"{round(value, decimals):.{decimals}f}"


def _sweep_positions_scan(
    block:  "PrintBlock",
    match:  "OriginMatch",
    sweep_axis: int,
) -> np.ndarray:
    """
    Extract the sweep-axis scan-space positions for all print waypoints
    in this block, including the block start.

    Returns (N+1,) array where index 0 = abs_start, indices 1..N = moves.
    """
    # Block start
    sx0, sy0 = match.to_scan(*block.abs_start[:2])
    s0 = sx0 if sweep_axis == 0 else sy0

    pts = [s0]
    for mv in block.moves:
        sx, sy = match.to_scan(mv.abs_x, mv.abs_y)
        pts.append(sx if sweep_axis == 0 else sy)

    return np.array(pts, dtype=float)


def _interpolate_xy_for_new_s(
    block:      "PrintBlock",
    s_original: np.ndarray,
    s_new:      np.ndarray,
    sweep_axis: int,
) -> list[tuple[float, float]]:
    """
    For ADAPTIVE_CURVATURE, which may insert NEW sweep positions,
    recover the code-space (x, y) at each new s by linear interpolation
    between the original waypoints.

    Parameters
    ----------
    block      : PrintBlock with absolute XY positions
    s_original : (N+1,) original sweep positions (index 0 = block start)
    s_new      : (M,)   new sweep positions from adaptive algorithm
    sweep_axis : 0=X, 1=Y

    Returns list of (x, y) in code-space for each s in s_new.
    """
    # Build arrays of original code-space XY at each s position
    orig_x = np.array([block.abs_start[0]] + [m.abs_x for m in block.moves])
    orig_y = np.array([block.abs_start[1]] + [m.abs_y for m in block.moves])

    # For monotone interpolation, sort by s
    sort_idx = np.argsort(s_original)
    s_sorted = s_original[sort_idx]
    x_sorted = orig_x[sort_idx]
    y_sorted = orig_y[sort_idx]

    x_new = np.interp(s_new, s_sorted, x_sorted)
    y_new = np.interp(s_new, s_sorted, y_sorted)

    return list(zip(x_new.tolist(), y_new.tolist()))


def _emit_print_block(
    block:         "PrintBlock",
    interp_result: InterpolationResult,
    s_original:    np.ndarray,
    h_prev_end:    float,
    match:         "OriginMatch",
    sweep_axis:    int,
    cfg:           ConformConfig,
) -> tuple[list[str], float, float]:
    """
    Emit the nScrypt lines for one conformed print block.

    Returns (lines, h_block_end, dz_correction_mm).

    h_prev_end  : normalised surface h at the end of the previous block
                  (or 0.0 for the first block = datum)
    h_block_end : normalised surface h at the end of THIS block (for next)
    """
    dp    = cfg.decimals
    out   = list(block.preamble_raw)   # / lineN, speed, trigwait, trigvalverel

    # ── Replace trigwait value with user-configured trigwait_time ────────────
    out = [
        (f"trigwait {_fmt(cfg.trigwait_time, dp)}"
         if ln.strip().lower().startswith("trigwait") else ln)
        for ln in out
    ]

    s_pts = interp_result.s_pts     # scan-space sweep positions
    h_pts = interp_result.h_pts     # normalised surface heights

    # ── Correctional δZ before TrigValveRel ─────────────────────────────────
    h_start  = float(h_pts[0])
    dz_corr  = h_start - h_prev_end

    if abs(dz_corr) > Z_SUPPRESS_THRESH:
        # Insert just before the trigvalverel (last line of preamble)
        # Replace the trigvalverel line in out with correction + trigvalverel
        tv_idx = next(
            (i for i in reversed(range(len(out)))
             if out[i].strip().lower() == "trigvalverel"),
            None
        )
        if tv_idx is not None:
            corr_line = (
                f"move  0  0  {_fmt(dz_corr, dp)}"
                f"  / Z correction ({dz_corr*1e3:+.2f} µm)"
            )
            out.insert(tv_idx, corr_line)

    # ── Build move list for ADAPTIVE (new positions) vs others ───────────────
    is_adaptive = (interp_result.algo == Algorithm.ADAPTIVE_CURVATURE.value)

    # Always interpolate XY from original waypoints.
    # For LSQ/LINEAR this recovers the original positions exactly.
    # For ADAPTIVE or any bisected case, this gives correct intermediate XY.
    xy_new = _interpolate_xy_for_new_s(block, s_original, s_pts, sweep_axis)

    # ── Accumulate absolute Z as we emit each move ───────────────────────────
    # Code Z uses cumulative relative moves.
    # After header + optional clearance delta + correction:
    #   current absolute Z = user_clearance + h_start
    cur_z = cfg.clearance_mm + h_start

    # Emit moves from index 1 onward (index 0 = block start / TrigValveRel pos)
    # The first actual move in code is 'move 0 0 0' (zero move at TrigValveRel)
    # We keep the zero move but then emit conformed moves for the rest.
    for idx in range(len(s_pts)):
        xi, yi = xy_new[idx]
        hi     = float(h_pts[idx])

        # Target absolute Z for this waypoint
        z_target = cfg.clearance_mm + hi

        if idx == 0:
            # First point: corresponds to TrigValveRel position
            # The original code emits 'move 0 0 0' here — preserve that
            # (the δZ correction already positioned us correctly)
            dx = 0.0
            dy = 0.0
            dz = 0.0
        else:
            # Previous absolute position
            xi_prev, yi_prev = xy_new[idx - 1]
            dx = xi    - xi_prev
            dy = yi    - yi_prev
            dz = z_target - cur_z

        # Suppress pure-noise Z jitter
        if abs(dz) < Z_SUPPRESS_THRESH and idx > 0:
            dz = 0.0

        out.append(f"move  {_fmt(dx, dp)}  {_fmt(dy, dp)}  {_fmt(dz, dp)}")
        cur_z = z_target

    # h at the END of this block (last waypoint)
    h_block_end = float(h_pts[-1])

    return out, h_block_end, dz_corr


# ── Main conforming function ───────────────────────────────────────────────────

def conform(
    parsed_code: "ParsedCode",
    loaded_scan: "LoadedScan",
    origin_match: "OriginMatch",
    surfaces:     list["FittedSurface"],
    cfg:          ConformConfig | None = None,
) -> ConformResult:
    """
    Apply surface conforming to all print blocks.

    Parameters
    ----------
    parsed_code  : from code_parser.parse()
    loaded_scan  : from scan_loader.load_scan()
    origin_match : from origin_matcher.compute_offset()
    surfaces     : from surface_fit.fit_all()
    cfg          : ConformConfig; uses defaults if None

    Returns
    -------
    ConformResult with output lines and per-block diagnostics.
    """
    if cfg is None:
        cfg = ConformConfig()

    interp_cfg  = cfg.to_interp_cfg()
    sweep_axis  = loaded_scan.sweep_axis
    result      = ConformResult()
    warns       = result.warnings

    # ── 1. Header ────────────────────────────────────────────────────────────
    out: list[str] = list(parsed_code.header_raw)

    clearance_delta = cfg.clearance_mm - DEFAULT_CLEARANCE_MM
    if abs(clearance_delta) > Z_SUPPRESS_THRESH:
        out.append(
            f"move  0  0  {_fmt(clearance_delta, cfg.decimals)}"
            f"  / clearance offset ({clearance_delta*1e3:+.1f} µm)"
        )

    # ── 2. Process each print block ──────────────────────────────────────────
    h_prev_end = 0.0   # datum for first block

    for i, block in enumerate(parsed_code.blocks):

        # ── Map code waypoints → scan-space sweep positions ──────────────────
        s_scan = _sweep_positions_scan(block, origin_match, sweep_axis)

        # s_code for interpolation = all positions from TrigValveRel onward
        # (index 0 = block start, used for correction; pass full array)
        s_code_interp = s_scan   # (N+1,): start + all moves

        # ── Look up the surface for this block's step position ────────────────
        # Step position in scan-space
        code_step = (block.abs_start[0] if loaded_scan.step_axis == 0
                     else block.abs_start[1])
        sx0, sy0 = origin_match.to_scan(block.abs_start[0], block.abs_start[1])
        step_scan = sx0 if loaded_scan.step_axis == 0 else sy0

        surf = surface_for_step(surfaces, step_scan)

        # ── Look up scan line for LSQ raw data ───────────────────────────────
        from scan_loader import nearest_line
        scan_ln = nearest_line(loaded_scan, step_scan)

        # ── Run interpolation ─────────────────────────────────────────────────
        try:
            interp = interp_run(surf, s_code_interp, scan_ln, interp_cfg)
        except Exception as exc:
            msg = f"Block '{block.label}': interpolation failed — {exc}"
            warns.append(msg)
            _warnings.warn(msg, RuntimeWarning)
            # Fall back: pass-through original block unchanged
            out.extend(block.preamble_raw)
            for mv in block.moves:
                out.append(mv.raw)
            # valverel comes from the travel/footer raw_lines, not appended here
            if i < len(parsed_code.travels):
                out.extend(parsed_code.travels[i].raw_lines)
            h_prev_end = 0.0
            continue

        if not interp.constraint_satisfied:
            msg = (f"Block '{block.label}': Z-rate constraint not fully "
                   f"satisfied. Consider reducing print_speed.")
            warns.append(msg)

        # ── Emit conformed block ──────────────────────────────────────────────
        block_lines, h_block_end, dz_corr = _emit_print_block(
            block         = block,
            interp_result = interp,
            s_original    = s_code_interp,
            h_prev_end    = h_prev_end,
            match         = origin_match,
            sweep_axis    = sweep_axis,
            cfg           = cfg,
        )
        out.extend(block_lines)
        # NOTE: no explicit valverel here — the following travel segment's
        # raw_lines (or footer_raw for the last block) already begins with it.

        # ── Diagnostics ───────────────────────────────────────────────────────
        result.block_stats.append(BlockStat(
            label                = block.label,
            n_moves_original     = block.n_moves,
            n_moves_final        = interp.n_total - 1,  # excl. start position
            n_bisected           = interp.n_bisected,
            constraint_satisfied = interp.constraint_satisfied,
            dz_correction_mm     = dz_corr,
            error_stats          = interp.error_stats,
        ))

        h_prev_end = h_block_end

        # ── Travel segment (pass through with plunge correction) ──────────────
        if i < len(parsed_code.travels):
            travel = parsed_code.travels[i]
            out.extend(travel.raw_lines)

    # ── 3. Footer ─────────────────────────────────────────────────────────────
    out.extend(parsed_code.footer_raw)
    out.append("/ end of toolpath")

    result.lines = out
    return result


# ── CLI self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from code_parser    import parse
    from scan_loader    import load_scan
    from origin_matcher import compute_offset
    from surface_fit    import fit_all, SurfaceFitConfig

    rng = np.random.default_rng(11)

    # ── Load real code file ──────────────────────────────────────────────────
    code_path = sys.argv[1] if len(sys.argv) > 1 else "toolpath.txt"
    pc = parse(code_path)

    # ── Build synthetic scan offset by (+2, -1) mm ───────────────────────────
    OFFSET = (2.0, -1.0)
    N = 300

    def true_h(y):
        """Realistic 5µm sinusoidal surface variation."""
        return 0.005 * np.sin(y / 4.0) + 0.002 * np.cos(y / 2.5)

    rows = []
    for blk in pc.blocks:
        x_scan  = blk.abs_start[0] + OFFSET[0]
        y_start = blk.abs_start[1] + OFFSET[1]
        y_end   = blk.abs_end[1]   + OFFSET[1]
        # Generate points in the same sweep direction as the code
        # so that scan_start_xy matches the code start after offset
        y = np.linspace(y_start, y_end, N)
        z = -10.0 + true_h(y) + rng.normal(0, 1e-5, N)
        rows.append(np.column_stack([np.full(N, x_scan), y, z]))

    data = np.vstack(rows)
    f = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False, mode="w")
    np.savetxt(f.name, data, delimiter=",", fmt="%.7f")
    f.close()

    sc    = load_scan(f.name, "path")
    match = compute_offset(pc, sc)
    surfs = fit_all(sc, SurfaceFitConfig(bin_size=0.3, smooth_window=9))

    print()
    print("=" * 64)
    print("Testing all three algorithms")
    print("=" * 64)

    for algo in Algorithm:
        cfg = ConformConfig(
            algorithm     = algo,
            print_speed   = 5.0,
            max_z_rate    = 1.0,
            clearance_mm  = 0.010,
            trigwait_time = 0.5,
            decimals      = 3,
        )
        result = conform(pc, sc, match, surfs, cfg)

        # ── Structural checks ─────────────────────────────────────────────
        # Count commands in output
        n_tv  = sum(1 for l in result.lines
                    if l.strip().lower() == "trigvalverel")
        n_vr  = sum(1 for l in result.lines
                    if l.strip().lower() == "valverel")
        n_tw  = sum(1 for l in result.lines
                    if l.strip().lower().startswith("trigwait"))
        n_spd = sum(1 for l in result.lines
                    if l.strip().lower().startswith("speed"))

        assert n_tv  == pc.n_blocks, f"{algo}: trigvalverel {n_tv} != {pc.n_blocks}"
        assert n_vr  >= pc.n_blocks, f"{algo}: valverel {n_vr} < {pc.n_blocks}"
        assert n_tw  == pc.n_blocks, f"{algo}: trigwait {n_tw} != {pc.n_blocks}"

        # Clearance delta: 10µm → 10µm, no delta expected
        delta_lines = [l for l in result.lines if "clearance offset" in l]
        assert len(delta_lines) == 0, f"{algo}: unexpected clearance delta line"

        print(f"\nPASS  [{algo.value}]  "
              f"trigvalverel={n_tv}  valverel={n_vr}  "
              f"trigwait={n_tw}  speed={n_spd}")
        print(result.summary())

    # ── Test clearance delta (5µm) ────────────────────────────────────────
    cfg_5um = ConformConfig(clearance_mm=0.005, trigwait_time=0.3, decimals=3)
    res_5um = conform(pc, sc, match, surfs, cfg_5um)
    delta_l = [l for l in res_5um.lines if "clearance offset" in l]
    assert len(delta_l) == 1, f"Expected 1 clearance delta line, got {len(delta_l)}"
    assert "-0.005" in delta_l[0], f"Wrong delta: {delta_l[0]}"
    print(f"\nPASS  clearance_mm=0.005 → delta line: '{delta_l[0].strip()}'")

    # ── Test trigwait replacement ─────────────────────────────────────────
    tw_lines = [l for l in res_5um.lines
                if l.strip().lower().startswith("trigwait")]
    for tl in tw_lines:
        val = float(tl.strip().split()[-1])
        assert abs(val - 0.3) < 1e-6, f"Wrong trigwait value: {val}"
    print(f"PASS  trigwait_time=0.3 applied to all {len(tw_lines)} trigwait lines")

    # ── Write to temp file and verify it can be read back ────────────────
    out_path = tempfile.mktemp(suffix="_conformed.txt")
    res_5um.write(out_path)
    pc2 = parse(out_path)
    assert pc2.n_blocks == pc.n_blocks, \
        f"Round-trip block count mismatch: {pc2.n_blocks} vs {pc.n_blocks}"
    print(f"PASS  write + re-parse round-trip: {pc2.n_blocks} blocks OK")

    print()
    print("All tests passed.")
    os.unlink(f.name)
    os.unlink(out_path)