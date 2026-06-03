"""
backend/code_parser.py
======================
Parses nScrypt .txt toolpath files into a structured, inspectable
representation suitable for Z-conforming and lossless reconstruction.

Public API
----------
    result = parse("toolpath.txt")

    result.n_blocks                  # number of print passes
    result.first_print_abs_xy        # (x, y) of first print waypoint in code-space
    result.all_print_abs_xy()        # [[( x,y), ...], ...]  per block
    result.default_clearance_mm      # always 0.010 mm (10 µm) — user sets plunge manually
    result.blocks[i]                 # PrintBlock with .moves, .abs_start, .abs_end
    result.travels[i]                # TravelSegment after blocks[i]
    result.header_raw                # raw header lines (list of str)
    result.footer_raw                # raw footer lines

File format assumed
-------------------
    Relative moves by default (dx dy dz all cumulative).
    Header:   speed [travel]  /  move 0 0 10  /  move 0 0 -9.990
    Block:    [/ lineN]  speed [print]  trigwait [t]  trigvalverel
              move ...  (N moves)
              valverel
    Travel:   speed [travel]  move(retract)  move(XY)  move(plunge)
    Footer:   speed [travel]  move(retract)  / end of toolpath
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_CLEARANCE_MM = 0.010    # 10 µm — standard header plunge
Z_CORRECTION_THRESH  = 1e-5     # suppress |δZ| < 0.01 µm (floating-point noise)
_LABEL_RE            = re.compile(r"^/\s*line\d+", re.IGNORECASE)


# ── Token types ────────────────────────────────────────────────────────────────

class T:
    MOVE      = "move"
    SPEED     = "speed"
    TRIGWAIT  = "trigwait"
    TRIGVALVE = "trigvalverel"
    VALVE     = "valverel"
    WAIT      = "wait"
    ABSOLUTE  = "absolute"
    RELATIVE  = "relative"
    TOOL      = "tool"
    COMMENT   = "comment"
    BLANK     = "blank"
    UNKNOWN   = "unknown"


# ── Tokeniser ──────────────────────────────────────────────────────────────────

def _strip_comment(raw: str) -> str:
    """Remove inline ; comment and strip surrounding whitespace."""
    return raw.split(";")[0].strip()


def _tokenize(raw: str) -> tuple[str, list[str]]:
    """Return (token_type, args) for one raw source line."""
    clean = _strip_comment(raw)
    if not clean:
        return T.BLANK, []
    if clean.startswith("/"):
        return T.COMMENT, [clean]
    parts = clean.lower().split()
    kw, args = parts[0], parts[1:]
    return {
        "move":         T.MOVE,
        "speed":        T.SPEED,
        "trigwait":     T.TRIGWAIT,
        "trigvalverel": T.TRIGVALVE,
        "valverel":     T.VALVE,
        "wait":         T.WAIT,
        "absolute":     T.ABSOLUTE,
        "relative":     T.RELATIVE,
        "tool":         T.TOOL,
    }.get(kw, T.UNKNOWN), args


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class CodeMove:
    """
    One parsed move command.

    dx/dy/dz   : relative delta values (as written or converted from absolute)
    abs_x/y/z  : cumulative absolute position in code-space after this move
    raw        : original stripped, comment-free line text (for reconstruction)
    """
    dx:    float
    dy:    float
    dz:    float
    abs_x: float
    abs_y: float
    abs_z: float
    raw:   str


@dataclass
class PrintBlock:
    """
    One dispensing pass: the region from TrigValveRel through ValveRel.

    index         : 0-based block index
    label         : label from '/ lineN' comment, or auto-generated
    speed_print   : mm/s for this block
    trigwait_time : seconds for TrigWait command
    moves         : all CodeMove objects within the print pass
    abs_start     : absolute (x, y, z) at the moment TrigValveRel fires
    abs_end       : absolute (x, y, z) after the last print move
    preamble_raw  : raw lines from label/speed/trigwait through trigvalverel (inclusive)
    """
    index:         int
    label:         str
    speed_print:   float
    trigwait_time: float
    moves:         list[CodeMove]           = field(default_factory=list)
    abs_start:     tuple[float, float, float] = (0.0, 0.0, 0.0)
    abs_end:       tuple[float, float, float] = (0.0, 0.0, 0.0)
    preamble_raw:  list[str]                = field(default_factory=list)

    @property
    def sweep_xy(self) -> list[tuple[float, float]]:
        """(x, y) of every print waypoint including abs_start."""
        pts = [(self.abs_start[0], self.abs_start[1])]
        pts.extend((m.abs_x, m.abs_y) for m in self.moves)
        return pts

    @property
    def n_moves(self) -> int:
        return len(self.moves)


@dataclass
class TravelSegment:
    """
    Inter-line travel: ValveRel + speed + retract-Z + XY-step + plunge-Z.

    raw_lines  : all source lines in this segment (for pass-through reconstruction)
    moves      : parsed CodeMove objects (for position tracking)
    plunge_idx : index into moves[] of the plunge (-Z) move; -1 if absent

    The z_conformer inserts a correctional δZ move immediately AFTER the plunge
    and BEFORE the next TrigWait/TrigValveRel.
    """
    raw_lines:  list[str]       = field(default_factory=list)
    moves:      list[CodeMove]  = field(default_factory=list)
    plunge_idx: int             = -1

    @property
    def plunge_move(self) -> CodeMove | None:
        if 0 <= self.plunge_idx < len(self.moves):
            return self.moves[self.plunge_idx]
        return None

    @property
    def retract_dz(self) -> float | None:
        """Z delta of the retract move (positive, lifts nozzle)."""
        for m in self.moves:
            if m.dz > 0:
                return m.dz
        return None

    @property
    def plunge_dz(self) -> float | None:
        """Z delta of the plunge move (negative, lowers nozzle)."""
        if self.plunge_move:
            return self.plunge_move.dz
        return None


@dataclass
class ParsedCode:
    """
    Complete structured representation of a parsed nScrypt .txt file.

    Attributes
    ----------
    header_raw           : raw lines before the first block preamble
    default_clearance_mm : always DEFAULT_CLEARANCE_MM (0.010 mm = 10 µm)
    travel_speed         : initial travel speed from header (mm/s)
    blocks               : PrintBlock list in file order
    travels              : TravelSegment list; travels[i] follows blocks[i]
                           len(travels) == len(blocks) - 1
    footer_raw           : raw lines after the last ValveRel

    Properties
    ----------
    first_print_abs_xy   : (x, y) of first print waypoint — used by origin_matcher
    all_print_abs_xy()   : [[( x,y), ...], ...]  per block — used by top-view panel
    """
    header_raw:           list[str]           = field(default_factory=list)
    default_clearance_mm: float               = DEFAULT_CLEARANCE_MM
    travel_speed:         float               = 10.0
    blocks:               list[PrintBlock]    = field(default_factory=list)
    travels:              list[TravelSegment] = field(default_factory=list)
    footer_raw:           list[str]           = field(default_factory=list)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def first_print_abs_xy(self) -> tuple[float, float]:
        """
        Code-space XY of the very first print waypoint.
        Passed to origin_matcher.compute_offset(scan_first_xy, this).
        """
        if not self.blocks:
            raise ValueError("No print blocks found — cannot determine origin.")
        x, y, _ = self.blocks[0].abs_start
        return x, y

    def all_print_abs_xy(self) -> list[list[tuple[float, float]]]:
        """
        Per-block list of (x, y) absolute positions for all print waypoints.
        Includes abs_start as the first point of each block.
        Used by the 2D top-view panel to draw the code footprint.
        """
        return [block.sweep_xy for block in self.blocks]

    def summary(self) -> str:
        lines = [
            f"ParsedCode  n_blocks={self.n_blocks}  "
            f"clearance={self.default_clearance_mm*1e3:.1f}µm  "
            f"travel_speed={self.travel_speed}mm/s",
        ]
        for b in self.blocks:
            sx, sy, sz = b.abs_start
            ex, ey, ez = b.abs_end
            lines.append(
                f"  [{b.label:>8}]  {b.n_moves:3d} moves  "
                f"start=({sx:8.3f},{sy:8.3f},{sz:+.4f})  "
                f"end=({ex:8.3f},{ey:8.3f},{ez:+.4f})"
            )
        return "\n".join(lines)


# ── Parser ─────────────────────────────────────────────────────────────────────

# State constants
_S_HEADER   = 0
_S_PREAMBLE = 1
_S_PRINT    = 2
_S_TRAVEL   = 3


def parse(filepath: str | Path) -> ParsedCode:
    """
    Parse an nScrypt .txt toolpath file into a ParsedCode structure.

    Parameters
    ----------
    filepath : path to the .txt file

    Returns
    -------
    ParsedCode

    Raises
    ------
    FileNotFoundError : file does not exist
    ValueError        : no TrigValveRel/ValveRel pairs found, or mismatched count
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Code file not found: {path}")

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    result    = ParsedCode()

    # ── Cumulative position tracker ──────────────────────────────────────────
    pos    = [0.0, 0.0, 0.0]   # [x, y, z] in code-space
    is_rel = True               # nScrypt default: relative moves

    def apply_move(args: list[str], raw_s: str) -> CodeMove:
        """Consume a move command: update pos, return a CodeMove."""
        nonlocal is_rel
        dx, dy, dz = float(args[0]), float(args[1]), float(args[2])
        if is_rel:
            nx, ny, nz = pos[0]+dx, pos[1]+dy, pos[2]+dz
        else:                          # absolute mode
            nx, ny, nz = dx, dy, dz
            dx = nx - pos[0]
            dy = ny - pos[1]
            dz = nz - pos[2]
        pos[0], pos[1], pos[2] = nx, ny, nz
        return CodeMove(dx=dx, dy=dy, dz=dz,
                        abs_x=nx, abs_y=ny, abs_z=nz, raw=raw_s)

    # ── State machine ────────────────────────────────────────────────────────
    state = _S_HEADER

    # Working accumulators
    current_block:    PrintBlock    | None = None
    current_travel:   TravelSegment | None = None
    pending_label:    str                  = ""
    pending_speed:    float                = 10.0
    pending_trigwait: float                = 0.5
    pending_preamble: list[str]            = []

    header_move_count: int   = 0   # moves seen in header section
    header_z_track:    list  = []  # abs_z after each header move
    travel_move_count: int   = 0   # moves seen in current travel segment

    cur_speed = 10.0

    # ── Main loop ────────────────────────────────────────────────────────────
    for line in raw_lines:
        tok, args = _tokenize(line)
        s         = _strip_comment(line)   # cleaned text for storage

        # ────────────────────────── HEADER ───────────────────────────────
        if state == _S_HEADER:
            result.header_raw.append(s)

            if tok == T.SPEED:
                new_spd = float(args[0])
                if header_move_count >= 2:
                    # Speed after header moves = print speed for first block
                    # Pop this line from header — it belongs to the preamble
                    result.header_raw.pop()
                    cur_speed        = new_spd
                    pending_speed    = new_spd
                    pending_preamble = [s]
                    state            = _S_PREAMBLE
                else:
                    cur_speed            = new_spd
                    result.travel_speed  = new_spd

            elif tok == T.MOVE:
                mv = apply_move(args, s)
                header_z_track.append(mv.abs_z)
                header_move_count += 1

            elif tok == T.ABSOLUTE:
                is_rel = False
            elif tok == T.RELATIVE:
                is_rel = True

            elif tok == T.COMMENT and _LABEL_RE.match(s):
                # "/ line1" → preamble of first block begins
                result.header_raw.pop()          # remove from header
                pending_label    = s.lstrip("/").strip()
                pending_preamble = [s]
                pending_speed    = cur_speed
                state            = _S_PREAMBLE

            elif tok == T.TRIGWAIT:
                # TrigWait with no preceding label — still a preamble line
                result.header_raw.pop()
                pending_trigwait = float(args[0])
                pending_preamble = [s]
                state            = _S_PREAMBLE

        # ────────────────────────── PREAMBLE ─────────────────────────────
        elif state == _S_PREAMBLE:
            pending_preamble.append(s)

            if tok == T.SPEED:
                cur_speed     = float(args[0])
                pending_speed = cur_speed

            elif tok == T.TRIGWAIT:
                pending_trigwait = float(args[0])

            elif tok == T.COMMENT and _LABEL_RE.match(s):
                pending_label = s.lstrip("/").strip()

            elif tok == T.MOVE:
                apply_move(args, s)   # keep position current (e.g. 0,0,0 first move)

            elif tok == T.ABSOLUTE:
                is_rel = False
            elif tok == T.RELATIVE:
                is_rel = True

            elif tok == T.TRIGVALVE:
                # Commit the PrintBlock
                idx = len(result.blocks)
                current_block = PrintBlock(
                    index         = idx,
                    label         = pending_label or f"line{idx + 1}",
                    speed_print   = pending_speed,
                    trigwait_time = pending_trigwait,
                    abs_start     = tuple(pos),       # type: ignore[arg-type]
                    preamble_raw  = list(pending_preamble),
                )
                result.blocks.append(current_block)
                # Reset pending state
                pending_label    = ""
                pending_preamble = []
                state = _S_PRINT

        # ────────────────────────── PRINT ────────────────────────────────
        elif state == _S_PRINT:
            if tok == T.MOVE:
                mv = apply_move(args, s)
                current_block.moves.append(mv)     # type: ignore[union-attr]

            elif tok == T.VALVE:
                current_block.abs_end = tuple(pos) # type: ignore[union-attr,assignment]
                # Begin travel segment; include the valverel line
                current_travel    = TravelSegment(raw_lines=[s])
                travel_move_count = 0
                state = _S_TRAVEL

            elif tok == T.ABSOLUTE:
                is_rel = False
            elif tok == T.RELATIVE:
                is_rel = True

        # ────────────────────────── TRAVEL ───────────────────────────────
        elif state == _S_TRAVEL:

            # ── Transition checks (BEFORE consuming the line) ─────────────
            # A '/ lineN' comment or a TrigWait after ≥1 travel move
            # signals end of travel and start of next preamble.
            if travel_move_count >= 1:
                if tok == T.COMMENT and _LABEL_RE.match(s):
                    result.travels.append(current_travel)  # type: ignore[arg-type]
                    current_travel    = None
                    pending_label     = s.lstrip("/").strip()
                    pending_preamble  = [s]
                    pending_speed     = cur_speed
                    state = _S_PREAMBLE
                    continue   # do NOT add this line to travel

                if tok == T.TRIGWAIT:
                    result.travels.append(current_travel)  # type: ignore[arg-type]
                    current_travel    = None
                    pending_trigwait  = float(args[0])
                    pending_preamble  = [s]
                    state = _S_PREAMBLE
                    continue

            # ── Consume the line ──────────────────────────────────────────
            current_travel.raw_lines.append(s)     # type: ignore[union-attr]

            if tok == T.SPEED:
                cur_speed = float(args[0])

            elif tok == T.MOVE:
                mv = apply_move(args, s)
                current_travel.moves.append(mv)    # type: ignore[union-attr]
                travel_move_count += 1
                # Track last -Z move as the plunge
                if mv.dz < 0:
                    current_travel.plunge_idx = len(current_travel.moves) - 1  # type: ignore

            elif tok == T.ABSOLUTE:
                is_rel = False
            elif tok == T.RELATIVE:
                is_rel = True
            # Blank lines and comments pass through to raw_lines unchanged

    # ── End-of-file cleanup ──────────────────────────────────────────────────
    # If still in TRAVEL, the remaining lines form the footer retract
    if state == _S_TRAVEL and current_travel is not None:
        result.footer_raw = current_travel.raw_lines

    # NOTE: default_clearance_mm is always DEFAULT_CLEARANCE_MM (0.010 mm = 10 µm).
    # The header plunge (e.g. 'move 0 0 -9.990') is manually set by the user
    # to place the nozzle at 10 µm above the surface. It is NOT used to infer
    # clearance — clearance is always 10 µm unless the user changes it in the UI.

    # ── Validation ───────────────────────────────────────────────────────────
    if not result.blocks:
        raise ValueError(
            f"No print blocks (TrigValveRel/ValveRel pairs) found in '{path.name}'. "
            "Verify the file uses nScrypt format."
        )

    n_expected_travels = result.n_blocks - 1
    if len(result.travels) != n_expected_travels:
        import warnings
        warnings.warn(
            f"Expected {n_expected_travels} travel segment(s), "
            f"found {len(result.travels)}. "
            "File may have non-standard structure between print blocks."
        )

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"[parser]  '{path.name}'")
    print(result.summary())

    return result


# ── Reconstruction helper ─────────────────────────────────────────────────────

def reconstruct_lines(result: ParsedCode,
                      clearance_delta_mm: float = 0.0,
                      decimals: int = 3) -> list[str]:
    """
    Reconstruct the raw lines of a ParsedCode as-is (no Z conforming applied).
    Optionally inserts the clearance delta line after the header plunge.

    This is the baseline for z_conformer, which calls this and then
    substitutes the Z-modified print block lines.

    Parameters
    ----------
    result             : ParsedCode from parse()
    clearance_delta_mm : user_clearance - default_clearance (may be 0)
    decimals           : decimal places for the delta move line

    Returns
    -------
    list[str] of lines (no newlines)
    """
    out: list[str] = []

    # Header
    out.extend(result.header_raw)

    # Optional clearance adjustment line after the header plunge
    if abs(clearance_delta_mm) > Z_CORRECTION_THRESH:
        fmt = f"{{:.{decimals}f}}"
        out.append(
            f"move  0  0  {fmt.format(clearance_delta_mm)}"
            f"  / clearance offset ({clearance_delta_mm*1e3:+.1f} µm)"
        )

    # Blocks and travels
    for i, block in enumerate(result.blocks):
        out.extend(block.preamble_raw)
        for mv in block.moves:
            out.append(mv.raw)
        out.append("valverel")
        if i < len(result.travels):
            out.extend(result.travels[i].raw_lines)

    # Footer
    out.extend(result.footer_raw)

    return out


# ── CLI self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "toolpath.txt"
    pc = parse(target)
    print()
    print(pc.summary())
    print()
    print(f"first_print_abs_xy = {pc.first_print_abs_xy}")
    print()
    print("All print XY extents per block:")
    for i, xy_list in enumerate(pc.all_print_abs_xy()):
        xs = [p[0] for p in xy_list]
        ys = [p[1] for p in xy_list]
        print(f"  block {i}: X=[{min(xs):.3f}, {max(xs):.3f}]  "
              f"Y=[{min(ys):.3f}, {max(ys):.3f}]  ({len(xy_list)} pts)")
    print()
    print("Travel plunge moves:")
    for i, t in enumerate(pc.travels):
        pm = t.plunge_move
        if pm:
            print(f"  travel {i}: plunge dz={pm.dz:+.4f}  "
                  f"abs_z_after={pm.abs_z:+.4f}")
        else:
            print(f"  travel {i}: no plunge detected")
    print()

    # Round-trip test: reconstruct and verify line count
    rebuilt = reconstruct_lines(pc)
    print(f"Reconstruction: {len(rebuilt)} lines output")
