"""
windows/conformal.py
====================
The Conformal Toolpath window.

Workflow (three panels, shown sequentially):
    Panel 1 — Load Files
        Pick scan.xyz and code.txt, declare scan type, click Load.
    Panel 2 — 2D Top View
        See scan coverage + code paths overlaid.
        Click a path to select it, then OK to proceed.
    Panel 3 — Configure & Generate
        Left  : parameter controls + algorithm selector
        Centre: embedded top-view thumbnail + surface/toolpath plot
        Bottom: Generate (selected) and Generate All buttons

On Generate All the conformed file is written as <stem>_conformed.txt
next to the original code.txt, and the preset is saved automatically.
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
import customtkinter as ctk

from ui.theme import (
    APP_TITLE, CONFORMAL_SIZE,
    COL_ACCENT, COL_TEAL, FONT_TITLE, FONT_LABEL, FONT_SMALL,
    PAD, PAD_S, PAD_L, CORNER,
    primary_button, secondary_button, success_button,
    section_frame, label, status_bar,
)
from ui.top_view_panel    import TopViewPanel
from ui.surface_plot_panel import SurfacePlotPanel

# Backend imports
from code_parser    import parse,  ParsedCode
from scan_loader    import load_scan, LoadedScan
from origin_matcher import compute_offset, OriginMatch
from surface_fit    import fit_all, SurfaceFitConfig, FittedSurface
from interpolation  import Algorithm
from z_conformer    import conform, ConformConfig
from preset         import Preset, from_configs, save as save_preset, to_configs

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ConformalWindow(ctk.CTkToplevel):
    """Conformal toolpath window — full three-panel workflow."""

    def __init__(self, parent):
        super().__init__(parent)
        self.title(f"{APP_TITLE} — Conformal toolpath")
        self.geometry(CONFORMAL_SIZE)
        self.minsize(900, 600)

        # ── State ─────────────────────────────────────────────────────────
        self._scan_path  = tk.StringVar()
        self._code_path  = tk.StringVar()
        self._scan_type  = tk.StringVar(value="path")
        self._selected_block = -1

        self._parsed_code: ParsedCode   | None = None
        self._loaded_scan: LoadedScan   | None = None
        self._origin:      OriginMatch  | None = None
        self._surfaces:    list[FittedSurface]  = []

        # ConformConfig controls (bound to tk variables)
        self._v_algo       = tk.StringVar(value=Algorithm.ADAPTIVE_CURVATURE.value)
        self._v_speed      = tk.StringVar(value="5.0")
        self._v_zrate      = tk.StringVar(value="1.0")
        self._v_clearance  = tk.StringVar(value="10")     # µm in UI
        self._v_trigwait   = tk.StringVar(value="0.5")
        self._v_decimals   = tk.StringVar(value="3")
        self._v_bin_size   = tk.StringVar(value="0.20")
        self._v_smooth     = tk.StringVar(value="11")

        self._build_ui()
        self._show_panel("load")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Title bar ─────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, corner_radius=0, fg_color=COL_ACCENT, height=40)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            hdr, text="Conformal toolpath",
            font=FONT_TITLE, text_color="white",
        ).pack(side="left", padx=PAD)
        self._step_lbl = ctk.CTkLabel(
            hdr, text="Step 1 of 3 — Load files",
            font=FONT_SMALL, text_color="#CCCCFF",
        )
        self._step_lbl.pack(side="right", padx=PAD)

        # ── Panel container ───────────────────────────────────────────────
        self._container = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self._container.grid(row=1, column=0, sticky="nsew")
        self._container.grid_columnconfigure(0, weight=1)
        self._container.grid_rowconfigure(0, weight=1)

        self._panel_load    = self._build_panel_load(self._container)
        self._panel_topview = self._build_panel_topview(self._container)
        self._panel_config  = self._build_panel_config(self._container)

        for p in (self._panel_load, self._panel_topview, self._panel_config):
            p.grid(row=0, column=0, sticky="nsew")

        # ── Status bar ────────────────────────────────────────────────────
        self._status = status_bar(self)
        self._status.grid(row=2, column=0, sticky="ew", padx=PAD, pady=(0, PAD_S))

    # ── Panel 1: Load ─────────────────────────────────────────────────────────

    def _build_panel_load(self, parent) -> ctk.CTkFrame:
        p = ctk.CTkFrame(parent, fg_color="transparent")

        inner = section_frame(p)
        inner.pack(expand=True, anchor="center", padx=PAD_L, pady=PAD_L,
                   fill="both")
        inner.grid_columnconfigure(1, weight=1)

        label(inner, "Load scan and code files", style="title").grid(
            row=0, column=0, columnspan=3, padx=PAD, pady=(PAD, PAD_L),
            sticky="w")

        # scan.xyz
        label(inner, "Scan file (.xyz)", style="normal").grid(
            row=1, column=0, sticky="w", padx=PAD, pady=PAD_S)
        ctk.CTkEntry(inner, textvariable=self._scan_path,
                     font=FONT_SMALL, width=420).grid(
            row=1, column=1, sticky="ew", padx=PAD_S, pady=PAD_S)
        secondary_button(inner, "Browse…",
                         command=self._browse_scan).grid(
            row=1, column=2, padx=(0, PAD), pady=PAD_S)

        # code.txt
        label(inner, "Code file (.txt)", style="normal").grid(
            row=2, column=0, sticky="w", padx=PAD, pady=PAD_S)
        ctk.CTkEntry(inner, textvariable=self._code_path,
                     font=FONT_SMALL, width=420).grid(
            row=2, column=1, sticky="ew", padx=PAD_S, pady=PAD_S)
        secondary_button(inner, "Browse…",
                         command=self._browse_code).grid(
            row=2, column=2, padx=(0, PAD), pady=PAD_S)

        # Scan type
        label(inner, "Scan type", style="normal").grid(
            row=3, column=0, sticky="w", padx=PAD, pady=PAD_S)
        type_frame = ctk.CTkFrame(inner, fg_color="transparent")
        type_frame.grid(row=3, column=1, sticky="w", padx=PAD_S, pady=PAD_S)
        for val, txt in [("path", "Path scan"), ("area", "Area scan (future)")]:
            ctk.CTkRadioButton(
                type_frame, text=txt, variable=self._scan_type,
                value=val, font=FONT_LABEL,
                state="normal" if val == "path" else "disabled",
            ).pack(side="left", padx=(0, PAD))

        # Load from preset
        label(inner, "Or load preset", style="muted").grid(
            row=4, column=0, sticky="w", padx=PAD, pady=(PAD_L, PAD_S))
        secondary_button(inner, "Load preset (.nsp)…",
                         command=self._load_preset).grid(
            row=4, column=1, sticky="w", padx=PAD_S, pady=(PAD_L, PAD_S))

        # Load button
        primary_button(inner, "Load  →", command=self._do_load,
                       width=160).grid(
            row=5, column=0, columnspan=3, pady=PAD_L)

        return p

    # ── Panel 2: 2D Top View ──────────────────────────────────────────────────

    def _build_panel_topview(self, parent) -> ctk.CTkFrame:
        p = ctk.CTkFrame(parent, fg_color="transparent")
        p.grid_columnconfigure(0, weight=1)
        p.grid_rowconfigure(0, weight=1)

        self._top_view = TopViewPanel(
            p, on_path_selected=self._on_path_clicked,
        )
        self._top_view.grid(row=0, column=0, sticky="nsew",
                            padx=PAD, pady=PAD)

        nav = ctk.CTkFrame(p, fg_color="transparent")
        nav.grid(row=1, column=0, sticky="ew", padx=PAD, pady=(0, PAD))

        secondary_button(nav, "← Back", command=lambda: self._show_panel("load")
                         ).pack(side="left")
        self._topview_sel_lbl = ctk.CTkLabel(
            nav, text="Click a path to select it",
            font=FONT_SMALL, text_color=("gray50", "gray60"),
        )
        self._topview_sel_lbl.pack(side="left", padx=PAD)
        primary_button(nav, "OK — Configure  →",
                       command=self._go_configure).pack(side="right")

        return p

    # ── Panel 3: Configure & Generate ────────────────────────────────────────

    def _build_panel_config(self, parent) -> ctk.CTkFrame:
        p = ctk.CTkFrame(parent, fg_color="transparent")
        p.grid_columnconfigure(1, weight=1)
        p.grid_rowconfigure(0, weight=1)

        # ── Left: parameters ──────────────────────────────────────────────
        left = section_frame(p, width=240)
        left.grid(row=0, column=0, sticky="ns", padx=(PAD, PAD_S),
                  pady=PAD)
        left.grid_propagate(False)
        left.grid_columnconfigure(1, weight=1)

        label(left, "Parameters", style="title").grid(
            row=0, column=0, columnspan=2, padx=PAD,
            pady=(PAD, PAD_S), sticky="w")

        _rows = [
            ("Algorithm",      self._v_algo,      "optionmenu",
             [a.value for a in Algorithm]),
            ("Print speed mm/s", self._v_speed,   "entry",    None),
            ("Max Z rate mm/s",  self._v_zrate,   "entry",    None),
            ("Clearance (µm)",   self._v_clearance,"entry",   None),
            ("TrigWait (s)",     self._v_trigwait, "entry",   None),
            ("Decimals",         self._v_decimals, "entry",   None),
            ("Bin size (mm)",    self._v_bin_size, "entry",   None),
            ("Smooth window",    self._v_smooth,   "entry",   None),
        ]
        for i, (lbl_txt, var, wtype, opts) in enumerate(_rows, start=1):
            label(left, lbl_txt, style="small").grid(
                row=i, column=0, sticky="w", padx=PAD, pady=3)
            if wtype == "optionmenu":
                ctk.CTkOptionMenu(
                    left, variable=var, values=opts,
                    width=130, font=FONT_SMALL,
                    command=lambda _: self._preview_selected(),
                ).grid(row=i, column=1, sticky="w", padx=PAD_S, pady=3)
            else:
                ctk.CTkEntry(
                    left, textvariable=var, width=90, font=FONT_SMALL,
                ).grid(row=i, column=1, sticky="w", padx=PAD_S, pady=3)

        row_after = len(_rows) + 1
        secondary_button(
            left, "Save preset", command=self._save_preset, width=210,
        ).grid(row=row_after, column=0, columnspan=2,
               padx=PAD, pady=(PAD_L, PAD_S))
        secondary_button(
            left, "Load preset", command=self._load_preset, width=210,
        ).grid(row=row_after+1, column=0, columnspan=2,
               padx=PAD, pady=PAD_S)

        # ── Centre: plots ─────────────────────────────────────────────────
        centre = ctk.CTkFrame(p, fg_color="transparent")
        centre.grid(row=0, column=1, sticky="nsew",
                    padx=PAD_S, pady=PAD)
        centre.grid_columnconfigure(0, weight=1)
        centre.grid_rowconfigure(1, weight=1)

        # Mini top-view thumbnail
        thumb_frame = section_frame(centre, height=180)
        thumb_frame.grid(row=0, column=0, sticky="ew")
        thumb_frame.grid_propagate(False)

        self._thumb_view = TopViewPanel(
            thumb_frame,
            on_path_selected=self._on_path_clicked,
        )
        self._thumb_view.pack(fill="both", expand=True)

        # Surface plot panel
        self._surf_plot = SurfacePlotPanel(centre)
        self._surf_plot.grid(row=1, column=0, sticky="nsew", pady=(PAD_S, 0))

        # ── Bottom navigation ─────────────────────────────────────────────
        nav = ctk.CTkFrame(p, fg_color="transparent", height=50)
        nav.grid(row=1, column=0, columnspan=2, sticky="ew",
                 padx=PAD, pady=(0, PAD))

        secondary_button(nav, "← Top view",
                         command=lambda: self._show_panel("topview")
                         ).pack(side="left")

        self._path_lbl = ctk.CTkLabel(
            nav, text="No path selected",
            font=FONT_SMALL, text_color=("gray50", "gray60"),
        )
        self._path_lbl.pack(side="left", padx=PAD)

        success_button(nav, "Generate All",
                       command=self._do_generate_all,
                       width=160).pack(side="right")
        primary_button(nav, "Generate selected",
                       command=self._do_generate_selected,
                       width=160).pack(side="right", padx=(0, PAD_S))

        return p

    # ── Panel switching ───────────────────────────────────────────────────────

    def _show_panel(self, name: str):
        panels = {
            "load":    (self._panel_load,    "Step 1 of 3 — Load files"),
            "topview": (self._panel_topview, "Step 2 of 3 — 2D top view"),
            "config":  (self._panel_config,  "Step 3 of 3 — Configure & Generate"),
        }
        p, step_text = panels[name]
        p.tkraise()
        self._step_lbl.configure(text=step_text)

    # ── File browsers ─────────────────────────────────────────────────────────

    def _browse_scan(self):
        path = filedialog.askopenfilename(
            title="Select scan file",
            filetypes=[("XYZ files", "*.xyz"), ("All files", "*.*")],
        )
        if path:
            self._scan_path.set(path)

    def _browse_code(self):
        path = filedialog.askopenfilename(
            title="Select code file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self._code_path.set(path)

    # ── Load ─────────────────────────────────────────────────────────────────

    def _do_load(self):
        scan_p = self._scan_path.get().strip()
        code_p = self._code_path.get().strip()

        if not scan_p or not os.path.isfile(scan_p):
            messagebox.showerror("Error", "Select a valid scan.xyz file.")
            return
        if not code_p or not os.path.isfile(code_p):
            messagebox.showerror("Error", "Select a valid code.txt file.")
            return

        self._set_status("Loading files…")
        self.update_idletasks()

        try:
            self._parsed_code = parse(code_p)
            self._loaded_scan = load_scan(
                scan_p, scan_type=self._scan_type.get()
            )
            self._origin  = compute_offset(self._parsed_code, self._loaded_scan)
            fit_cfg       = self._read_fit_cfg()
            self._surfaces = fit_all(self._loaded_scan, fit_cfg)
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            self._set_status("Load failed.")
            return

        # Coverage warning
        if not self._origin.coverage_ok:
            msg = "Coverage warning:\n" + "\n".join(self._origin.warnings)
            messagebox.showwarning("Coverage", msg)

        # Populate top-view panels
        scan_xy = self._loaded_scan.all_surface_xy()
        code_xy = [np.array(xy) for xy in self._parsed_code.all_print_abs_xy()]

        # Apply origin offset to code XY for overlay
        code_xy_scan = [
            np.column_stack([
                xy[:, 0] + self._origin.dx,
                xy[:, 1] + self._origin.dy,
            ])
            for xy in code_xy
        ]
        scan_h = [ln.h for ln in self._loaded_scan.lines]

        for panel in (self._top_view, self._thumb_view):
            panel.load(
                scan_xy    = [xy.astype(float) for xy in scan_xy],
                code_xy    = [xy.astype(float) for xy in code_xy_scan],
                scan_h     = [h.astype(float) for h in scan_h],
            )

        self._set_status(
            f"Loaded: {self._loaded_scan.n_lines} scan lines, "
            f"{self._parsed_code.n_blocks} code blocks."
        )
        self._show_panel("topview")

    # ── Path selection callback ───────────────────────────────────────────────

    def _on_path_clicked(self, block_index: int):
        self._selected_block = block_index
        lbl = self._parsed_code.blocks[block_index].label \
              if self._parsed_code else str(block_index)

        self._topview_sel_lbl.configure(
            text=f"Selected: {lbl}",
            text_color=(COL_ACCENT, "#9999FF"),
        )
        self._path_lbl.configure(
            text=f"Selected: {lbl}",
            text_color=(COL_ACCENT, "#9999FF"),
        )
        self._thumb_view.select_path(block_index)
        self._top_view.select_path(block_index)

    # ── Configure panel entry ─────────────────────────────────────────────────

    def _go_configure(self):
        self._show_panel("config")
        if self._selected_block >= 0:
            self._preview_selected()

    # ── Preview single block ──────────────────────────────────────────────────

    def _preview_selected(self):
        if self._selected_block < 0 or not self._surfaces:
            return
        try:
            bd = self._build_block_data(self._selected_block)
        except Exception as exc:
            self._set_status(f"Preview error: {exc}")
            return
        self._surf_plot.update_single(bd)
        self._set_status(
            f"Preview: {bd['label']}  "
            f"err mean={bd['error_stats'].get('mean',0):+.3f} µm"
        )

    # ── Generate selected (preview only) ─────────────────────────────────────

    def _do_generate_selected(self):
        if self._selected_block < 0:
            messagebox.showinfo("", "Click a path in the top view first.")
            return
        self._preview_selected()

    # ── Generate All ─────────────────────────────────────────────────────────

    def _do_generate_all(self):
        if not self._parsed_code or not self._surfaces:
            messagebox.showerror("Error", "Load files first.")
            return

        out_path = Path(self._code_path.get())
        out_path = out_path.with_stem(out_path.stem + "_conformed")

        cfg = self._read_conform_cfg()
        self._set_status("Generating…")
        self.update_idletasks()

        def worker():
            try:
                result = conform(
                    self._parsed_code, self._loaded_scan,
                    self._origin, self._surfaces, cfg,
                )
                result.write(out_path)

                # Build plot data for all blocks
                all_bd = [self._build_block_data(i)
                          for i in range(self._parsed_code.n_blocks)]

                self.after(0, self._on_generate_done, result, all_bd,
                           str(out_path))
            except Exception as exc:
                self.after(0, lambda e=exc:
                    messagebox.showerror("Generate error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_generate_done(self, result, all_bd, out_path):
        self._surf_plot.update(all_bd)
        warns = result.warnings
        msg = f"Saved: {out_path}"
        if warns:
            msg += f"\n{len(warns)} warning(s)"
        self._set_status(msg)
        if warns:
            messagebox.showwarning("Warnings", "\n".join(warns))

    # ── Block data builder (for plots) ────────────────────────────────────────

    def _build_block_data(self, block_index: int) -> dict:
        from scan_loader    import nearest_line
        from surface_fit    import surface_for_step
        from interpolation  import run as interp_run

        blk  = self._parsed_code.blocks[block_index]
        cfg  = self._read_conform_cfg()

        sx0, sy0 = self._origin.to_scan(*blk.abs_start[:2])
        step_scan = (sx0 if self._loaded_scan.step_axis == 0 else sy0)
        surf      = surface_for_step(self._surfaces, step_scan)
        scan_ln   = nearest_line(self._loaded_scan, step_scan)

        from z_conformer import _sweep_positions_scan
        s_code = _sweep_positions_scan(
            blk, self._origin, self._loaded_scan.sweep_axis
        )

        interp = interp_run(surf, s_code, scan_ln, cfg.to_interp_cfg())

        s_fine = np.linspace(
            float(s_code.min()), float(s_code.max()), 300
        )
        h_spline = surf.h_at_array(s_fine)

        return {
            "label":               blk.label,
            "step_val":            step_scan,
            "s_fine":              s_fine,
            "h_spline":            h_spline,
            "clearance_mm":        cfg.clearance_mm,
            "s_tool":              interp.s_pts,
            "h_tool":              interp.h_pts,
            "s_raw":               scan_ln.sweep_coords,
            "h_raw":               scan_ln.h,
            "error_stats":         interp.error_stats,
            "n_bisected":          interp.n_bisected,
            "constraint_satisfied":interp.constraint_satisfied,
        }

    # ── Config readers ────────────────────────────────────────────────────────

    def _read_conform_cfg(self) -> ConformConfig:
        return ConformConfig(
            algorithm     = Algorithm(self._v_algo.get()),
            print_speed   = float(self._v_speed.get()),
            max_z_rate    = float(self._v_zrate.get()),
            clearance_mm  = float(self._v_clearance.get()) * 1e-3,  # µm→mm
            trigwait_time = float(self._v_trigwait.get()),
            decimals      = int(self._v_decimals.get()),
        )

    def _read_fit_cfg(self) -> SurfaceFitConfig:
        return SurfaceFitConfig(
            bin_size      = float(self._v_bin_size.get()),
            smooth_window = int(self._v_smooth.get()),
        )

    # ── Preset save / load ────────────────────────────────────────────────────

    def _save_preset(self):
        path = filedialog.asksaveasfilename(
            title="Save preset",
            defaultextension=".nsp",
            filetypes=[("nScrypt preset", "*.nsp")],
        )
        if not path:
            return
        try:
            p = from_configs(
                self._read_conform_cfg(),
                self._read_fit_cfg(),
                code_path = self._code_path.get(),
                scan_type = self._scan_type.get(),
                name      = Path(path).stem,
            )
            save_preset(p, path)
            self._set_status(f"Preset saved: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Preset error", str(exc))

    def _load_preset(self):
        from preset import load as load_preset
        path = filedialog.askopenfilename(
            title="Load preset",
            filetypes=[("nScrypt preset", "*.nsp"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            p = load_preset(path)
            cc, fc, cp, st = to_configs(p)
            self._v_algo.set(cc.algorithm.value)
            self._v_speed.set(str(cc.print_speed))
            self._v_zrate.set(str(cc.max_z_rate))
            self._v_clearance.set(str(cc.clearance_mm * 1e3))
            self._v_trigwait.set(str(cc.trigwait_time))
            self._v_decimals.set(str(cc.decimals))
            self._v_bin_size.set(str(fc.bin_size))
            self._v_smooth.set(str(fc.smooth_window))
            if cp:
                self._code_path.set(cp)
            self._scan_type.set(st)
            self._set_status(f"Preset loaded: {p.name}")
        except Exception as exc:
            messagebox.showerror("Preset error", str(exc))

    # ── Status ────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._status.configure(text=msg)
        self.update_idletasks()
