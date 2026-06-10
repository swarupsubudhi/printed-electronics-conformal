"""
ui/surface_plot_panel.py
========================
Tabbed panel showing the conformed surface vs toolpath for all print lines.

Tab 1 — "Lines"  : stacked per-line subplots (surface spline, clearance band,
                    conformed toolpath, raw scan h values)
Tab 2 — "3D view": 3D surface mesh with toolpath overlay
Tab 3 — "Stats"  : per-block error stat table

The panel can be updated incrementally (one block at a time during Generate)
or all at once after Generate All.
"""

from __future__ import annotations

import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — registers projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import customtkinter as ctk

from ui.theme import (
    PLOT_SCAN_LINE, PLOT_SPLINE, PLOT_TOOLPATH, PLOT_TOL_BAND,
    PAD, PAD_S, FONT_SMALL, FONT_MONO_S,
)


class SurfacePlotPanel(ctk.CTkFrame):
    """
    Tabbed plot panel.  Call update() to populate with new data.
    """

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._data: list[dict] = []   # list of block-data dicts
        self._current_idx = 0         # index of the line shown on the Lines tab
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._tabs = ctk.CTkTabview(self, corner_radius=6)
        self._tabs.grid(row=0, column=0, sticky="nsew", padx=PAD_S, pady=PAD_S)

        self._tab_lines = self._tabs.add("Lines")
        self._tab_3d    = self._tabs.add("3D view")
        self._tab_stats = self._tabs.add("Stats")

        for tab in (self._tab_lines, self._tab_3d, self._tab_stats):
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)

        # ── Lines tab ─────────────────────────────────────────────────────
        # One line shown at a time; navigate via dropdown + ◄ ► arrows.
        self._tab_lines.grid_rowconfigure(0, weight=0)   # control bar
        self._tab_lines.grid_rowconfigure(1, weight=1)   # canvas (expands)
        self._tab_lines.grid_rowconfigure(2, weight=0)   # matplotlib toolbar

        ctrl = ctk.CTkFrame(self._tab_lines, fg_color="transparent")
        ctrl.grid(row=0, column=0, sticky="ew", padx=PAD_S, pady=(PAD_S, 0))

        self._line_var = tk.StringVar(value="—")
        self._prev_btn = ctk.CTkButton(
            ctrl, text="◄", width=34, command=self._prev_line)
        self._prev_btn.pack(side="left")
        self._line_menu = ctk.CTkOptionMenu(
            ctrl, variable=self._line_var, values=["—"], width=170,
            font=FONT_SMALL, command=self._on_line_selected)
        self._line_menu.pack(side="left", padx=PAD_S)
        self._next_btn = ctk.CTkButton(
            ctrl, text="►", width=34, command=self._next_line)
        self._next_btn.pack(side="left")
        self._line_count_lbl = ctk.CTkLabel(
            ctrl, text="", font=FONT_SMALL, text_color=("gray50", "gray60"))
        self._line_count_lbl.pack(side="left", padx=PAD)

        self._lines_fig = plt.figure(figsize=(7, 3), tight_layout=True)
        self._lines_canvas = FigureCanvasTkAgg(self._lines_fig,
                                               master=self._tab_lines)
        self._lines_canvas.get_tk_widget().grid(row=1, column=0,
                                                sticky="nsew")
        tb_frame = ctk.CTkFrame(self._tab_lines, fg_color="transparent", height=28)
        tb_frame.grid(row=2, column=0, sticky="ew")
        NavigationToolbar2Tk(self._lines_canvas, tb_frame).update()

        # ── 3D tab ────────────────────────────────────────────────────────
        self._3d_fig = plt.figure(figsize=(7, 5), tight_layout=True)
        self._3d_ax  = self._3d_fig.add_subplot(111, projection="3d")
        self._3d_canvas = FigureCanvasTkAgg(self._3d_fig,
                                            master=self._tab_3d)
        self._3d_canvas.get_tk_widget().grid(row=0, column=0,
                                             sticky="nsew")
        tb3_frame = ctk.CTkFrame(self._tab_3d, fg_color="transparent", height=28)
        tb3_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self._3d_canvas, tb3_frame).update()

        # ── Stats tab ─────────────────────────────────────────────────────
        self._stats_text = ctk.CTkTextbox(
            self._tab_stats, font=FONT_MONO_S,
            wrap="none", state="disabled",
        )
        self._stats_text.grid(row=0, column=0, sticky="nsew",
                               padx=PAD_S, pady=PAD_S)

        # Set the selector to its empty/disabled state initially
        self._refresh_line_selector()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, block_data: list[dict]):
        """
        Refresh all three tabs with new data.

        Each dict in block_data must contain:
            label        str
            s_fine       (K,) sweep positions for spline curve
            h_spline     (K,) spline h values
            clearance_mm float
            s_tool       (N,) waypoint sweep positions
            h_tool       (N,) waypoint h values (before clearance)
            s_raw        (M,) raw scan sweep positions
            h_raw        (M,) raw scan h values
            error_stats  dict with mean/std/min/max/rms
            n_bisected   int
        """
        self._data = block_data
        self._current_idx = 0
        self._refresh_line_selector()
        self._draw_lines()
        self._draw_3d()
        self._draw_stats()

    def update_single(self, block_dict: dict):
        """Add/replace one block's data and show it on the Lines tab."""
        label = block_dict["label"]
        for i, d in enumerate(self._data):
            if d["label"] == label:
                self._data[i] = block_dict
                idx = i
                break
        else:
            self._data.append(block_dict)
            idx = len(self._data) - 1
        self._current_idx = idx          # jump to the line just updated
        self._refresh_line_selector()
        self._draw_lines()

    def clear(self):
        self._data = []
        self._current_idx = 0
        for fig in (self._lines_fig, self._3d_fig):
            fig.clf()
        for canvas in (self._lines_canvas, self._3d_canvas):
            canvas.draw_idle()
        self._stats_text.configure(state="normal")
        self._stats_text.delete("1.0", "end")
        self._stats_text.configure(state="disabled")
        self._refresh_line_selector()

    # ── Line navigation (one at a time) ───────────────────────────────────────

    def _refresh_line_selector(self):
        """Sync dropdown values, count label, and arrow states to the data."""
        n = len(self._data)
        if n == 0:
            self._line_menu.configure(values=["—"])
            self._line_var.set("—")
            self._line_count_lbl.configure(text="")
            self._prev_btn.configure(state="disabled")
            self._next_btn.configure(state="disabled")
            return
        self._current_idx = max(0, min(self._current_idx, n - 1))
        labels = [d["label"] for d in self._data]
        self._line_menu.configure(values=labels)
        self._line_var.set(labels[self._current_idx])
        self._line_count_lbl.configure(
            text=f"Line {self._current_idx + 1} of {n}")
        self._prev_btn.configure(
            state="normal" if self._current_idx > 0 else "disabled")
        self._next_btn.configure(
            state="normal" if self._current_idx < n - 1 else "disabled")

    def _on_line_selected(self, choice: str):
        for i, d in enumerate(self._data):
            if d["label"] == choice:
                self._current_idx = i
                break
        self._refresh_line_selector()
        self._draw_lines()

    def _prev_line(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._refresh_line_selector()
            self._draw_lines()

    def _next_line(self):
        if self._current_idx < len(self._data) - 1:
            self._current_idx += 1
            self._refresh_line_selector()
            self._draw_lines()

    # ── Lines tab ─────────────────────────────────────────────────────────────

    def _draw_lines(self):
        """Draw ONLY the currently-selected line (one at a time)."""
        fig = self._lines_fig
        fig.clf()
        n = len(self._data)
        if n == 0:
            self._lines_canvas.draw_idle()
            return

        idx = max(0, min(self._current_idx, n - 1))
        d   = self._data[idx]

        ax = fig.add_subplot(111)
        cl = d["clearance_mm"]
        s_f, h_sp = d["s_fine"], d["h_spline"]
        s_t, h_t  = d["s_tool"], d["h_tool"]
        s_r, h_r  = d.get("s_raw"), d.get("h_raw")

        # Raw scan points (if provided)
        if s_r is not None and len(s_r):
            ax.scatter(s_r, h_r * 1e3, s=1, color="#CCCCCC",
                       alpha=0.5, zorder=1, label="Raw scan")

        # Fitted surface spline
        ax.plot(s_f, h_sp * 1e3, color=PLOT_SPLINE,
                linewidth=1.5, zorder=2, label="Surface spline")

        # Clearance band around surface
        ax.fill_between(
            s_f, h_sp * 1e3, (h_sp + cl) * 1e3,
            color=PLOT_TOL_BAND, alpha=0.10, zorder=2,
            label=f"{cl*1e3:.0f} µm clearance",
        )

        # Conformed toolpath (h + clearance displayed)
        z_tool = (h_t + cl) * 1e3
        ax.plot(s_t, z_tool, color=PLOT_TOOLPATH,
                linewidth=1.4, marker="o", markersize=3,
                zorder=4, label=f"Toolpath ({len(s_t)} pts)")

        # Annotation
        st  = d.get("error_stats", {})
        bis = d.get("n_bisected", 0)
        ax.set_title(
            f"{d['label']}  |  "
            f"err mean={st.get('mean',0):+.3f} µm  "
            f"std={st.get('std',0):.3f} µm  "
            f"bisected={bis}",
            fontsize=9,
        )
        ax.set_xlabel("Sweep (mm)", fontsize=9)
        ax.set_ylabel("h (µm)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        ax.grid(True, linestyle="--", alpha=0.3)

        self._lines_canvas.draw_idle()

    # ── 3D tab ────────────────────────────────────────────────────────────────

    def _draw_3d(self):
        fig = self._3d_fig
        fig.clf()
        ax  = fig.add_subplot(111, projection="3d")
        self._3d_ax = ax

        if not self._data:
            self._3d_canvas.draw_idle()
            return

        for d in self._data:
            step = d.get("step_val", 0.0)
            s_f  = d["s_fine"]
            h_sp = d["h_spline"]
            cl   = d["clearance_mm"]
            s_t  = d["s_tool"]
            h_t  = d["h_tool"]

            # Surface ribbon at this step position
            ax.plot(s_f, np.full_like(s_f, step), h_sp * 1e3,
                    color=PLOT_SPLINE, linewidth=1.2, alpha=0.6)

            # Toolpath
            ax.plot(s_t, np.full_like(s_t, step), (h_t + cl) * 1e3,
                    color=PLOT_TOOLPATH, linewidth=1.5,
                    marker="o", markersize=2.5, alpha=0.9)

        ax.set_xlabel("Sweep (mm)", fontsize=8, labelpad=4)
        ax.set_ylabel("Step (mm)",  fontsize=8, labelpad=4)
        ax.set_zlabel("h (µm)",     fontsize=8, labelpad=4)
        ax.set_title("3D surface + conformed toolpath", fontsize=9)
        ax.tick_params(labelsize=7)

        self._3d_canvas.draw_idle()

    # ── Stats tab ─────────────────────────────────────────────────────────────

    def _draw_stats(self):
        if not self._data:
            return

        col_w = [10, 9, 9, 9, 9, 9, 9]
        hdr = (
            f"{'Block':<{col_w[0]}}"
            f"{'mean(µm)':>{col_w[1]}}"
            f"{'std':>{col_w[2]}}"
            f"{'min':>{col_w[3]}}"
            f"{'max':>{col_w[4]}}"
            f"{'rms':>{col_w[5]}}"
            f"{'bisected':>{col_w[6]}}"
        )
        sep  = "─" * sum(col_w)
        rows = [hdr, sep]

        for d in self._data:
            st  = d.get("error_stats", {})
            bis = d.get("n_bisected", 0)
            ok  = "" if d.get("constraint_satisfied", True) else " !"
            rows.append(
                f"{d['label']:<{col_w[0]}}"
                f"{st.get('mean',0):>{col_w[1]}.3f}"
                f"{st.get('std', 0):>{col_w[2]}.3f}"
                f"{st.get('min', 0):>{col_w[3]}.3f}"
                f"{st.get('max', 0):>{col_w[4]}.3f}"
                f"{st.get('rms', 0):>{col_w[5]}.3f}"
                f"{bis:>{col_w[6]}}{ok}"
            )

        text = "\n".join(rows)
        self._stats_text.configure(state="normal")
        self._stats_text.delete("1.0", "end")
        self._stats_text.insert("1.0", text)
        self._stats_text.configure(state="disabled")