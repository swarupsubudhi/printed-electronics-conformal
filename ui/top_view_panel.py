"""
ui/top_view_panel.py
====================
Embedded 2D matplotlib canvas showing:
  - Scan surface polylines (one per forward scan line, blue)
  - Code.txt XY path footprint (orange)
  - Selected path highlighted in amber
  - Optional raw scan point cloud (toggled off by default)
  - Optional Z colormap on scan lines (toggled off by default)

The panel fires a callback when the user clicks a path:
    on_path_selected(block_index: int)
"""

from __future__ import annotations

import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
import customtkinter as ctk

from ui.theme import (
    PLOT_SCAN_LINE, PLOT_CODE_PATH, PLOT_SELECTED,
    PAD, PAD_S, FONT_SMALL,
)


class TopViewPanel(ctk.CTkFrame):
    """
    2D XY top-view canvas embedded in a CTk frame.

    Parameters
    ----------
    parent          : CTk parent widget
    on_path_selected: callback(block_index: int) fired on path click
    """

    # Click tolerance in data units (mm) for path selection
    _PICK_TOL_MM = 0.5

    def __init__(self, parent, on_path_selected=None, **kw):
        super().__init__(parent, **kw)

        self._cb         = on_path_selected
        self._scan_xy    = []   # list of (N,2) arrays — scan polylines
        self._code_xy    = []   # list of (N,2) arrays — code path footprints
        self._scan_h     = []   # list of (N,) h arrays for colormap
        self._selected   = -1   # currently selected block index

        self._show_pts   = tk.BooleanVar(value=False)
        self._show_cmap  = tk.BooleanVar(value=False)

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Matplotlib figure
        self._fig, self._ax = plt.subplots(figsize=(6, 5), tight_layout=True)
        self._fig.patch.set_facecolor("none")
        self._ax.set_aspect("equal")
        self._ax.set_xlabel("X (mm)", fontsize=9)
        self._ax.set_ylabel("Y (mm)", fontsize=9)
        self._ax.set_title("Top view — scan coverage + code paths", fontsize=9)
        self._ax.grid(True, linestyle="--", alpha=0.3)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew",
                                          padx=PAD_S, pady=PAD_S)

        # Navigation toolbar
        toolbar_frame = ctk.CTkFrame(self, fg_color="transparent", height=30)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        self._toolbar = NavigationToolbar2Tk(self._canvas, toolbar_frame)
        self._toolbar.update()

        # Toggle controls
        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.grid(row=2, column=0, sticky="ew", padx=PAD_S, pady=(0, PAD_S))

        ctk.CTkCheckBox(
            ctrl, text="Show scan point cloud", variable=self._show_pts,
            font=FONT_SMALL, command=self._redraw,
        ).pack(side="left", padx=PAD_S)

        ctk.CTkCheckBox(
            ctrl, text="Z colormap", variable=self._show_cmap,
            font=FONT_SMALL, command=self._redraw,
        ).pack(side="left", padx=PAD_S)

        # Connect click event
        self._canvas.mpl_connect("button_press_event", self._on_click)

    # ── Public interface ──────────────────────────────────────────────────────

    def load(
        self,
        scan_xy:  list[np.ndarray],
        code_xy:  list[np.ndarray],
        scan_raw: list[np.ndarray] | None = None,
        scan_h:   list[np.ndarray] | None = None,
    ):
        """
        Populate the panel with scan + code data and redraw.

        Parameters
        ----------
        scan_xy  : list of (N,2) XY arrays for each forward scan line
        code_xy  : list of (M,2) XY arrays for each code print block
        scan_raw : list of (N,3) XYZ raw point cloud per line (optional)
        scan_h   : list of (N,) normalised h arrays per line (for colormap)
        """
        self._scan_xy  = scan_xy
        self._code_xy  = code_xy
        self._scan_raw = scan_raw or []
        self._scan_h   = scan_h or [np.zeros(len(xy)) for xy in scan_xy]
        self._selected = -1
        self._redraw()

    def select_path(self, block_index: int):
        """Programmatically select a path by block index."""
        self._selected = block_index
        self._redraw()

    def clear(self):
        self._scan_xy = []
        self._code_xy = []
        self._scan_raw = []
        self._scan_h = []
        self._selected = -1
        self._redraw()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _redraw(self):
        ax = self._ax
        ax.cla()
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)", fontsize=9)
        ax.set_ylabel("Y (mm)", fontsize=9)
        ax.set_title("Top view — scan coverage + code paths", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)

        # Optional raw point cloud
        if self._show_pts.get() and self._scan_raw:
            for pts in self._scan_raw:
                ax.scatter(pts[:, 0], pts[:, 1], s=0.5,
                           color="#BBBBBB", alpha=0.4, zorder=1)

        # Scan surface polylines
        if self._show_cmap.get() and self._scan_h:
            all_h = np.concatenate(self._scan_h)
            h_min, h_max = float(all_h.min()), float(all_h.max())
            cmap = plt.cm.coolwarm

            for xy, h in zip(self._scan_xy, self._scan_h):
                pts   = xy.reshape(-1, 1, 2)
                segs  = np.concatenate([pts[:-1], pts[1:]], axis=1)
                # Normalise h to [0,1] for colormap
                norm_h = (0.5 * np.ones(len(segs)) if h_max == h_min
                          else (h[:-1] - h_min) / (h_max - h_min))
                lc = LineCollection(segs, cmap=cmap, linewidth=1.2,
                                    alpha=0.8, zorder=2)
                lc.set_array(norm_h)
                ax.add_collection(lc)
        else:
            for xy in self._scan_xy:
                ax.plot(xy[:, 0], xy[:, 1],
                        color=PLOT_SCAN_LINE, linewidth=1.0,
                        alpha=0.6, zorder=2)

        # Code path footprints
        for i, xy in enumerate(self._code_xy):
            is_sel   = (i == self._selected)
            colour   = PLOT_SELECTED if is_sel else PLOT_CODE_PATH
            lw       = 2.2 if is_sel else 1.2
            alpha    = 1.0 if is_sel else 0.7
            zorder   = 5 if is_sel else 3

            ax.plot(xy[:, 0], xy[:, 1],
                    color=colour, linewidth=lw,
                    alpha=alpha, zorder=zorder,
                    solid_capstyle="round")

            # Mark start point
            ax.plot(xy[0, 0], xy[0, 1], "o",
                    color=colour, markersize=4 if is_sel else 3,
                    zorder=zorder + 1)

        # Auto-fit axes if we have data
        all_xy = self._scan_xy + self._code_xy
        if all_xy:
            stacked = np.vstack(all_xy)
            xs, ys  = stacked[:, 0], stacked[:, 1]
            margin  = max((xs.max()-xs.min()), (ys.max()-ys.min())) * 0.05 + 0.5
            ax.set_xlim(xs.min()-margin, xs.max()+margin)
            ax.set_ylim(ys.min()-margin, ys.max()+margin)

        # Legend
        from matplotlib.lines import Line2D
        handles = []
        if self._scan_xy:
            handles.append(Line2D([0],[0], color=PLOT_SCAN_LINE,
                                  lw=1.5, label="Scan coverage"))
        if self._code_xy:
            handles.append(Line2D([0],[0], color=PLOT_CODE_PATH,
                                  lw=1.5, label="Code paths"))
        if handles:
            ax.legend(handles=handles, fontsize=8, loc="upper right")

        self._canvas.draw_idle()

    # ── Click-to-select ───────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self._ax:
            return
        if event.button != 1:        # left click only
            return
        if self._toolbar.mode != "": # ignore if pan/zoom active
            return
        if not self._code_xy:
            return

        cx, cy    = event.xdata, event.ydata
        best_i    = -1
        best_dist = float("inf")

        for i, xy in enumerate(self._code_xy):
            # Minimum distance from click to any segment of this path
            for j in range(len(xy) - 1):
                d = _point_to_segment_dist(
                    cx, cy,
                    xy[j, 0], xy[j, 1],
                    xy[j+1, 0], xy[j+1, 1],
                )
                if d < best_dist:
                    best_dist = d
                    best_i    = i

        if best_dist <= self._PICK_TOL_MM:
            self._selected = best_i
            self._redraw()
            if self._cb is not None:
                self._cb(best_i)


# ── Geometry helper ───────────────────────────────────────────────────────────

def _point_to_segment_dist(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Perpendicular distance from point P to segment AB."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return np.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)))
    return np.hypot(px - (ax + t*dx), py - (ay + t*dy))
