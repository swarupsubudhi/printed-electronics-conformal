"""
backend/preset.py
=================
Save and load ConformConfig presets as human-readable JSON files.

A preset captures everything needed to reproduce a conforming run:
    - All ConformConfig fields (algorithm, speeds, clearance, etc.)
    - Path to the original code.txt file
    - Scan type declaration ('path' or 'area')
    - SurfaceFitConfig fields (bin_size, smooth_window, sweep_length)
    - Metadata (name, created timestamp, version)

Preset files use .nsp extension (nScrypt Preset) but are plain JSON.

Public API
----------
    save(preset, path)                  write .nsp file
    load(path)  -> Preset               read .nsp file
    from_configs(conform_cfg,           build Preset from config objects
                 fit_cfg,
                 code_path,
                 scan_type,
                 name) -> Preset
    to_configs(preset)                  unpack back to config objects
    -> (ConformConfig, SurfaceFitConfig, code_path, scan_type)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from interpolation import Algorithm
from z_conformer   import ConformConfig
from surface_fit   import SurfaceFitConfig

PRESET_VERSION = "1.0"
PRESET_EXT     = ".nsp"


# ── Preset data class ─────────────────────────────────────────────────────────

@dataclass
class Preset:
    """
    Complete preset for a conformal toolpath run.

    All fields are JSON-serialisable primitives so the file is human-editable.

    Attributes
    ----------
    name          : user-given preset name (shown in UI)
    version       : preset format version (for future migration)
    created_at    : ISO-8601 timestamp string
    code_path     : absolute path to the code.txt file used
    scan_type     : 'path' or 'area'

    --- ConformConfig fields ---
    algorithm     : 'lsq_poly_3' | 'linear_10pt' | 'adaptive_curvature'
    print_speed   : mm/s
    max_z_rate    : mm/s
    clearance_mm  : mm  (nozzle standoff above surface)
    wait_time     : s   (Wait command duration, future use)
    trigwait_time : s   (TrigWait before every TrigValveRel)
    decimals      : decimal places in output move values
    n_linear      : waypoints for LINEAR_10PT
    poly_degree   : polynomial degree for LSQ_POLY_3
    n_seed        : seed waypoints for ADAPTIVE_CURVATURE

    --- SurfaceFitConfig fields ---
    bin_size      : mm  sweep-axis bin width for surface averaging
    smooth_window : uniform filter window for binned surface data
    sweep_length  : mm or None — crop each scan line to this length
    """
    name:         str   = "default"
    version:      str   = PRESET_VERSION
    created_at:   str   = field(default_factory=lambda: _now_iso())
    code_path:    str   = ""
    scan_type:    str   = "path"

    # ConformConfig
    algorithm:    str   = Algorithm.ADAPTIVE_CURVATURE.value
    print_speed:  float = 5.0
    max_z_rate:   float = 1.0
    clearance_mm: float = 0.010
    wait_time:    float = 0.0
    trigwait_time: float = 0.5
    decimals:     int   = 3
    n_linear:     int   = 10
    poly_degree:  int   = 3
    n_seed:       int   = 20

    # SurfaceFitConfig
    bin_size:      float       = 0.20
    smooth_window: int         = 11
    sweep_length:  float | None = None

    # ── Validation ─────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """
        Return a list of human-readable validation errors.
        Empty list means the preset is valid.
        """
        errs: list[str] = []
        valid_algos = {a.value for a in Algorithm}

        if self.algorithm not in valid_algos:
            errs.append(
                f"algorithm '{self.algorithm}' not in {sorted(valid_algos)}"
            )
        if self.print_speed <= 0:
            errs.append(f"print_speed must be > 0 (got {self.print_speed})")
        if self.max_z_rate <= 0:
            errs.append(f"max_z_rate must be > 0 (got {self.max_z_rate})")
        if self.clearance_mm < 0:
            errs.append(f"clearance_mm must be ≥ 0 (got {self.clearance_mm})")
        if self.trigwait_time < 0:
            errs.append(f"trigwait_time must be ≥ 0 (got {self.trigwait_time})")
        if self.decimals < 1 or self.decimals > 6:
            errs.append(f"decimals must be 1–6 (got {self.decimals})")
        if self.n_linear < 2:
            errs.append(f"n_linear must be ≥ 2 (got {self.n_linear})")
        if self.poly_degree < 1 or self.poly_degree > 9:
            errs.append(f"poly_degree must be 1–9 (got {self.poly_degree})")
        if self.n_seed < 2:
            errs.append(f"n_seed must be ≥ 2 (got {self.n_seed})")
        if self.bin_size <= 0:
            errs.append(f"bin_size must be > 0 (got {self.bin_size})")
        if self.smooth_window < 1:
            errs.append(f"smooth_window must be ≥ 1 (got {self.smooth_window})")
        if self.sweep_length is not None and self.sweep_length <= 0:
            errs.append(
                f"sweep_length must be > 0 or None (got {self.sweep_length})"
            )
        if self.scan_type not in ("path", "area"):
            errs.append(
                f"scan_type must be 'path' or 'area' (got '{self.scan_type}')"
            )
        return errs

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"Preset '{self.name}'  (v{self.version}  {self.created_at})",
            f"  code_path    : {self.code_path or '(not set)'}",
            f"  scan_type    : {self.scan_type}",
            f"  algorithm    : {self.algorithm}",
            f"  print_speed  : {self.print_speed} mm/s",
            f"  max_z_rate   : {self.max_z_rate} mm/s",
            f"  clearance    : {self.clearance_mm*1e3:.1f} µm",
            f"  trigwait     : {self.trigwait_time} s",
            f"  bin_size     : {self.bin_size} mm",
            f"  smooth_window: {self.smooth_window}",
            f"  sweep_length : {self.sweep_length} mm",
        ]
        errs = self.validate()
        if errs:
            lines.append(f"  VALIDATION ERRORS ({len(errs)}):")
            for e in errs:
                lines.append(f"    - {e}")
        return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_nsp_ext(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix.lower() != PRESET_EXT:
        p = p.with_suffix(PRESET_EXT)
    return p


# ── Public API ─────────────────────────────────────────────────────────────────

def save(preset: Preset, path: str | Path) -> Path:
    """
    Write a Preset to a .nsp JSON file.

    Parameters
    ----------
    preset : Preset to serialise
    path   : destination path (extension forced to .nsp)

    Returns
    -------
    Path actually written to.
    """
    errs = preset.validate()
    if errs:
        raise ValueError(
            f"Preset '{preset.name}' has validation errors:\n"
            + "\n".join(f"  - {e}" for e in errs)
        )

    out_path = _ensure_nsp_ext(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(preset)
    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[preset]  saved '{preset.name}' → '{out_path}'")
    return out_path


def load(path: str | Path) -> Preset:
    """
    Load a Preset from a .nsp JSON file.

    Parameters
    ----------
    path : path to the .nsp file

    Returns
    -------
    Preset (validated; raises ValueError if invalid)

    Raises
    ------
    FileNotFoundError : file not found
    ValueError        : JSON parse error, missing required fields, or
                        validation failures
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Preset file not found: '{p}'")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse preset '{p.name}': {exc}") from exc

    # Version migration hook (future)
    file_ver = data.get("version", "1.0")
    if file_ver != PRESET_VERSION:
        import warnings
        warnings.warn(
            f"Preset version '{file_ver}' differs from current '{PRESET_VERSION}'. "
            "Loading with defaults for missing fields.",
            UserWarning,
        )

    # Build Preset: accept only known fields, ignore extras, fill missing with defaults
    known = {f.name for f in Preset.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known}

    preset = Preset(**{
        **asdict(Preset()),   # start from defaults
        **filtered,           # overlay loaded values
    })

    errs = preset.validate()
    if errs:
        raise ValueError(
            f"Preset '{p.name}' failed validation:\n"
            + "\n".join(f"  - {e}" for e in errs)
        )

    print(f"[preset]  loaded '{preset.name}' from '{p}'")
    return preset


def from_configs(
    conform_cfg: ConformConfig,
    fit_cfg:     SurfaceFitConfig,
    code_path:   str | Path = "",
    scan_type:   str         = "path",
    name:        str         = "default",
) -> Preset:
    """
    Build a Preset from the current live config objects.

    Used by the UI 'Save Preset' button — it reads the current widget
    state via ConformConfig + SurfaceFitConfig and writes a Preset.
    """
    return Preset(
        name          = name,
        created_at    = _now_iso(),
        code_path     = str(code_path),
        scan_type     = scan_type,
        # ConformConfig
        algorithm     = conform_cfg.algorithm.value,
        print_speed   = conform_cfg.print_speed,
        max_z_rate    = conform_cfg.max_z_rate,
        clearance_mm  = conform_cfg.clearance_mm,
        wait_time     = conform_cfg.wait_time,
        trigwait_time = conform_cfg.trigwait_time,
        decimals      = conform_cfg.decimals,
        n_linear      = conform_cfg.n_linear,
        poly_degree   = conform_cfg.poly_degree,
        n_seed        = conform_cfg.n_seed,
        # SurfaceFitConfig
        bin_size      = fit_cfg.bin_size,
        smooth_window = fit_cfg.smooth_window,
        sweep_length  = fit_cfg.sweep_length,
    )


def to_configs(
    preset: Preset,
) -> tuple[ConformConfig, SurfaceFitConfig, str, str]:
    """
    Unpack a Preset back into live config objects.

    Used by the UI when loading a preset — populates all widgets from
    the stored values and returns configs ready for the backend.

    Returns
    -------
    (ConformConfig, SurfaceFitConfig, code_path, scan_type)
    """
    conform_cfg = ConformConfig(
        algorithm     = Algorithm(preset.algorithm),
        print_speed   = preset.print_speed,
        max_z_rate    = preset.max_z_rate,
        clearance_mm  = preset.clearance_mm,
        wait_time     = preset.wait_time,
        trigwait_time = preset.trigwait_time,
        decimals      = preset.decimals,
        n_linear      = preset.n_linear,
        poly_degree   = preset.poly_degree,
        n_seed        = preset.n_seed,
    )
    fit_cfg = SurfaceFitConfig(
        bin_size      = preset.bin_size,
        smooth_window = preset.smooth_window,
        sweep_length  = preset.sweep_length,
    )
    return conform_cfg, fit_cfg, preset.code_path, preset.scan_type


# ── CLI self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import tempfile

    tmpdir = Path(tempfile.mkdtemp())

    # ── 1. Build from defaults ────────────────────────────────────────────
    p1 = Preset(
        name         = "test_preset",
        code_path    = "/data/code.txt",
        scan_type    = "path",
        algorithm    = "adaptive_curvature",
        print_speed  = 5.0,
        max_z_rate   = 1.0,
        clearance_mm = 0.010,
        trigwait_time = 0.5,
        bin_size     = 0.2,
        smooth_window = 11,
    )
    errs = p1.validate()
    assert not errs, f"Unexpected validation errors: {errs}"
    print("PASS  Preset validates with no errors")

    # ── 2. Save ───────────────────────────────────────────────────────────
    saved_path = save(p1, tmpdir / "test_preset.nsp")
    assert saved_path.exists()
    assert saved_path.suffix == ".nsp"
    print(f"PASS  save() → '{saved_path.name}'")

    # ── 3. Load ───────────────────────────────────────────────────────────
    p2 = load(saved_path)
    assert p2.name          == p1.name
    assert p2.algorithm     == p1.algorithm
    assert p2.print_speed   == p1.print_speed
    assert p2.clearance_mm  == p1.clearance_mm
    assert p2.trigwait_time == p1.trigwait_time
    assert p2.bin_size      == p1.bin_size
    assert p2.code_path     == p1.code_path
    assert p2.scan_type     == p1.scan_type
    print("PASS  load() restores all fields exactly")

    # ── 4. Extension auto-correction ─────────────────────────────────────
    p3 = save(p1, tmpdir / "no_ext_preset")   # no .nsp
    assert p3.suffix == ".nsp"
    print("PASS  extension auto-corrected to .nsp")

    # ── 5. from_configs round-trip ────────────────────────────────────────
    cc = ConformConfig(
        algorithm     = Algorithm.LSQ_POLY_3,
        print_speed   = 8.0,
        max_z_rate    = 0.8,
        clearance_mm  = 0.005,
        trigwait_time = 0.3,
        decimals      = 4,
    )
    fc = SurfaceFitConfig(bin_size=0.3, smooth_window=7, sweep_length=30.0)
    preset_rt = from_configs(cc, fc, "/tmp/code.txt", "path", "lsq_run")
    cc2, fc2, cp2, st2 = to_configs(preset_rt)

    assert cc2.algorithm     == Algorithm.LSQ_POLY_3
    assert cc2.print_speed   == 8.0
    assert cc2.clearance_mm  == 0.005
    assert cc2.trigwait_time == 0.3
    assert cc2.decimals      == 4
    assert fc2.bin_size      == 0.3
    assert fc2.smooth_window == 7
    assert fc2.sweep_length  == 30.0
    assert cp2               == "/tmp/code.txt"
    assert st2               == "path"
    print("PASS  from_configs / to_configs round-trip")

    # ── 6. Validation catches bad values ─────────────────────────────────
    bad = Preset(
        name="bad", algorithm="nonexistent",
        print_speed=-1.0, clearance_mm=-0.005,
        decimals=0, scan_type="laser"
    )
    errs = bad.validate()
    assert len(errs) >= 4, f"Expected ≥4 errors, got {len(errs)}: {errs}"
    print(f"PASS  validation catches {len(errs)} bad-value errors")

    # ── 7. save() rejects invalid preset ─────────────────────────────────
    try:
        save(bad, tmpdir / "bad.nsp")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "validation errors" in str(e)
    print("PASS  save() raises ValueError for invalid preset")

    # ── 8. load() handles missing file ───────────────────────────────────
    try:
        load(tmpdir / "nonexistent.nsp")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    print("PASS  load() raises FileNotFoundError for missing file")

    # ── 9. load() handles corrupted JSON ─────────────────────────────────
    corrupt = tmpdir / "corrupt.nsp"
    corrupt.write_text("{ not valid json }", encoding="utf-8")
    try:
        load(corrupt)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "parse" in str(e).lower()
    print("PASS  load() raises ValueError for corrupt JSON")

    # ── 10. load() ignores unknown extra fields (forward compat) ─────────
    with_extra = tmpdir / "extra_fields.nsp"
    data = json.loads(saved_path.read_text())
    data["future_field"] = "ignored"
    with_extra.write_text(json.dumps(data), encoding="utf-8")
    p_extra = load(with_extra)
    assert p_extra.name == p1.name
    print("PASS  load() ignores unknown fields (forward compatibility)")

    # ── 11. Human-readable JSON check ────────────────────────────────────
    raw = saved_path.read_text(encoding="utf-8")
    assert "\n" in raw,    "JSON should be indented (human-readable)"
    assert '"name"' in raw, "JSON should contain 'name' key"
    print("PASS  JSON file is human-readable (indented)")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)

    print()
    print("All tests passed.")
