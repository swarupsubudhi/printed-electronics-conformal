"""
windows/batch_conformal.py
==========================
Batch Conformal Toolpath window.

Workflow:
    1. Load a saved preset (.nsp) — this supplies ConformConfig + code.txt path
    2. Select a folder of scan.xyz files
    3. Process each file: run full pipeline, display surface + toolpath
    4. User confirms (OK) or rejects each file
    5. Generate All — write *_conformed.txt for every approved file
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

import numpy as np
import customtkinter as ctk

from ui.theme import (
    APP_TITLE, BATCH_SIZE,
    COL_AMBER, COL_TEAL,
    FONT_TITLE, FONT_LABEL, FONT_SMALL, FONT_MONO_S,
    PAD, PAD_S, CORNER,
    primary_button, secondary_button, success_button,
    section_frame, label, status_bar,
)
from ui.surface_plot_panel import SurfacePlotPanel

from code_parser    import parse
from scan_loader    import load_scan
from origin_matcher import compute_offset
from surface_fit    import fit_all, SurfaceFitConfig
from z_conformer    import conform
from preset         import load as load_preset, to_configs


class BatchConformalWindow(ctk.CTkToplevel):

    def __init__(self, parent):
        super().__init__(parent)
        self.title(f"{APP_TITLE} — Batch conformal")
        self.geometry(BATCH_SIZE)
        self.minsize(700, 500)

        self._preset        = None
        self._conform_cfg   = None
        self._fit_cfg       = None
        self._code_path_str = ""
        self._scan_folder   = tk.StringVar()
        self._preset_name   = tk.StringVar(value="(none loaded)")

        # {filename: dict(status, result, scan_path)}
        self._file_states: dict[str, dict] = {}

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        hdr = ctk.CTkFrame(self, corner_radius=0,
                           fg_color=COL_AMBER, height=40)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            hdr, text="Batch conformal toolpath",
            font=FONT_TITLE, text_color="white",
        ).pack(side="left", padx=PAD)

        # Main body
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=PAD, pady=PAD)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)

        # ── Left panel: controls ──────────────────────────────────────────
        left = section_frame(body, width=260)
        left.grid(row=0, column=0, rowspan=2, sticky="ns",
                  padx=(0, PAD_S))
        left.grid_propagate(False)

        label(left, "Batch settings", style="title").pack(
            padx=PAD, pady=(PAD, PAD_S), anchor="w")

        # Preset
        label(left, "Preset (.nsp)", style="small").pack(
            padx=PAD, anchor="w")
        ctk.CTkLabel(
            left, textvariable=self._preset_name,
            font=FONT_SMALL,
            text_color=("gray50", "gray60"),
        ).pack(padx=PAD, pady=(0, PAD_S), anchor="w")
        secondary_button(left, "Load preset…",
                         command=self._load_preset,
                         width=220).pack(padx=PAD, pady=(0, PAD_S))

        ctk.CTkSeparator(left, orientation="horizontal").pack(
            fill="x", padx=PAD, pady=PAD_S)

        # Scan folder
        label(left, "Scan folder", style="small").pack(
            padx=PAD, anchor="w")
        ctk.CTkEntry(
            left, textvariable=self._scan_folder,
            font=FONT_SMALL, width=220,
        ).pack(padx=PAD, pady=(0, PAD_S))
        secondary_button(left, "Browse folder…",
                         command=self._browse_folder,
                         width=220).pack(padx=PAD, pady=(0, PAD_L))

        primary_button(left, "Scan & preview all",
                       command=self._do_scan_all,
                       width=220).pack(padx=PAD, pady=PAD_S)

        ctk.CTkSeparator(left, orientation="horizontal").pack(
            fill="x", padx=PAD, pady=PAD_S)

        success_button(left, "Generate All approved",
                       command=self._do_generate_all,
                       width=220).pack(padx=PAD, pady=PAD_S)

        # ── Top-right: file list ──────────────────────────────────────────
        list_frame = section_frame(body)
        list_frame.grid(row=0, column=1, sticky="ew",
                        pady=(0, PAD_S))
        list_frame.grid_columnconfigure(0, weight=1)

        label(list_frame, "Scan files", style="title").grid(
            row=0, column=0, padx=PAD, pady=(PAD, PAD_S), sticky="w")

        self._file_list = ctk.CTkScrollableFrame(
            list_frame, height=160,
        )
        self._file_list.grid(row=1, column=0, sticky="ew",
                             padx=PAD, pady=(0, PAD))
        self._file_list.grid_columnconfigure(0, weight=1)

        # ── Bottom-right: surface plot ────────────────────────────────────
        self._surf_plot = SurfacePlotPanel(body)
        self._surf_plot.grid(row=1, column=1, sticky="nsew")

        # Status bar
        self._status = status_bar(self)
        self._status.grid(row=2, column=0, sticky="ew",
                          padx=PAD, pady=(0, PAD_S))

    # ── Preset loading ────────────────────────────────────────────────────────

    def _load_preset(self):
        path = filedialog.askopenfilename(
            title="Load conformal preset",
            filetypes=[("nScrypt preset", "*.nsp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self._preset = load_preset(path)
            self._conform_cfg, self._fit_cfg, cp, _ = to_configs(self._preset)
            self._code_path_str = cp
            self._preset_name.set(self._preset.name)
            self._set_status(f"Preset loaded: {self._preset.name}")
        except Exception as exc:
            messagebox.showerror("Preset error", str(exc))

    # ── Folder browser ────────────────────────────────────────────────────────

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select scan folder")
        if folder:
            self._scan_folder.set(folder)
            self._populate_file_list(folder)

    def _populate_file_list(self, folder: str):
        # Clear existing rows
        for w in self._file_list.winfo_children():
            w.destroy()
        self._file_states.clear()

        xyz_files = sorted(Path(folder).glob("*.xyz"))
        if not xyz_files:
            ctk.CTkLabel(self._file_list,
                         text="No .xyz files found.",
                         font=FONT_SMALL).pack(padx=PAD_S, pady=PAD_S)
            return

        for i, fp in enumerate(xyz_files):
            row = ctk.CTkFrame(self._file_list, fg_color="transparent")
            row.pack(fill="x", pady=2)
            row.grid_columnconfigure(1, weight=1)

            # Status icon (will be updated)
            icon = ctk.CTkLabel(row, text="○", font=FONT_LABEL, width=20)
            icon.grid(row=0, column=0, padx=(PAD_S, 4))

            ctk.CTkLabel(row, text=fp.name,
                         font=FONT_SMALL, anchor="w").grid(
                row=0, column=1, sticky="w")

            # Approve / Reject buttons
            def make_approve(name=fp.name, ic=icon):
                return lambda: self._approve(name, ic, True)
            def make_reject(name=fp.name, ic=icon):
                return lambda: self._approve(name, ic, False)

            ctk.CTkButton(
                row, text="✔", width=30, height=24,
                fg_color=COL_TEAL, hover_color="#0F6E56",
                font=FONT_SMALL,
                command=make_approve(),
            ).grid(row=0, column=2, padx=2)
            ctk.CTkButton(
                row, text="✘", width=30, height=24,
                fg_color="#A32D2D", hover_color="#791F1F",
                font=FONT_SMALL,
                command=make_reject(),
            ).grid(row=0, column=3, padx=(2, PAD_S))

            self._file_states[fp.name] = {
                "path":     str(fp),
                "approved": True,
                "icon":     icon,
                "result":   None,
            }

        self._set_status(f"{len(xyz_files)} scan file(s) found.")

    def _approve(self, name: str, icon: ctk.CTkLabel, approved: bool):
        self._file_states[name]["approved"] = approved
        icon.configure(
            text="✔" if approved else "✘",
            text_color=COL_TEAL if approved else "#E24B4A",
        )

    # ── Scan & preview all ────────────────────────────────────────────────────

    def _do_scan_all(self):
        if not self._conform_cfg:
            messagebox.showerror("Error", "Load a preset first.")
            return
        if not self._code_path_str or not Path(self._code_path_str).exists():
            messagebox.showerror("Error",
                                 "Preset does not contain a valid code.txt path.")
            return
        if not self._file_states:
            messagebox.showerror("Error",
                                 "Select a scan folder with .xyz files first.")
            return

        self._set_status("Processing…")
        self.update_idletasks()

        def worker():
            try:
                parsed = parse(self._code_path_str)
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror(
                    "Code file error", str(exc)))
                return

            all_bd = []
            for name, state in self._file_states.items():
                try:
                    sc    = load_scan(state["path"],
                                      scan_type=self._preset.scan_type)
                    match = compute_offset(parsed, sc)
                    surfs = fit_all(sc, self._fit_cfg)
                    result = conform(parsed, sc, match, surfs,
                                     self._conform_cfg)
                    state["result"] = result

                    # Build plot data for first block of this file
                    from scan_loader   import nearest_line
                    from surface_fit   import surface_for_step
                    from interpolation import run as interp_run
                    from z_conformer   import _sweep_positions_scan

                    blk      = parsed.blocks[0]
                    sx0, sy0 = match.to_scan(*blk.abs_start[:2])
                    step_s   = sx0 if sc.step_axis == 0 else sy0
                    surf     = surface_for_step(surfs, step_s)
                    scan_ln  = nearest_line(sc, step_s)
                    s_code   = _sweep_positions_scan(
                        blk, match, sc.sweep_axis)
                    interp   = interp_run(surf, s_code, scan_ln,
                                         self._conform_cfg.to_interp_cfg())

                    s_fine   = np.linspace(
                        float(s_code.min()), float(s_code.max()), 300)
                    all_bd.append({
                        "label":       name,
                        "step_val":    step_s,
                        "s_fine":      s_fine,
                        "h_spline":    surf.h_at_array(s_fine),
                        "clearance_mm":self._conform_cfg.clearance_mm,
                        "s_tool":      interp.s_pts,
                        "h_tool":      interp.h_pts,
                        "s_raw":       scan_ln.sweep_coords,
                        "h_raw":       scan_ln.h,
                        "error_stats": interp.error_stats,
                        "n_bisected":  interp.n_bisected,
                        "constraint_satisfied": interp.constraint_satisfied,
                    })

                    state["icon"].configure(text="◉",
                                            text_color=COL_TEAL)
                except Exception as exc:
                    state["icon"].configure(text="!", text_color="#E24B4A")
                    state["result"] = None

            self.after(0, lambda bd=all_bd: self._surf_plot.update(bd))
            n = len([s for s in self._file_states.values() if s["result"]])
            self.after(0, lambda: self._set_status(
                f"Preview ready — {n}/{len(self._file_states)} files processed."))

        threading.Thread(target=worker, daemon=True).start()

    # ── Generate All ─────────────────────────────────────────────────────────

    def _do_generate_all(self):
        approved = [
            (name, s) for name, s in self._file_states.items()
            if s["approved"] and s["result"] is not None
        ]
        if not approved:
            messagebox.showinfo("", "No approved files with results to generate.")
            return

        out_dir = Path(self._scan_folder.get())
        written = []
        errors  = []

        for name, state in approved:
            try:
                out_path = (out_dir / name).with_stem(
                    Path(name).stem + "_conformed"
                ).with_suffix(".txt")
                state["result"].write(out_path)
                written.append(str(out_path.name))
                state["icon"].configure(text="✔✔",
                                        text_color=COL_TEAL)
            except Exception as exc:
                errors.append(f"{name}: {exc}")

        msg = f"Generated {len(written)} file(s) in:\n{out_dir}"
        if errors:
            msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors)
        messagebox.showinfo("Batch complete", msg)
        self._set_status(f"Done — {len(written)} file(s) written.")

    def _set_status(self, msg: str):
        self._status.configure(text=msg)
        self.update_idletasks()
