"""
windows/toolpath_gen.py
=======================
Toolpath Generator window.

Layout:
    Left sidebar  : "New" button
                    PRESETS section — checkboxes for composite overlay
                    COMPOSITE FILES label + active list
                    Browse file… button
    Centre        : code editor with line numbers
    Right         : 3D/2D visualiser
    Bottom bar    : Refresh button + Export button

Refresh is ALWAYS manual — the plot never updates automatically.
Pressing Refresh replots whatever is currently in the editor,
plus any checked composite files, all overlaid on the same axes.

Composite files share [0,0,0] origin and are distinguished by
auto-assigned colours from COMPOSITE_COLOURS.

To add presets: place .txt files in the presets/ folder next to main.pyw.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import customtkinter as ctk

from ui.theme import (
    APP_TITLE, TOOLPATH_GEN_SIZE, COL_CORAL,
    FONT_TITLE, FONT_LABEL, FONT_SMALL, FONT_MONO, FONT_MONO_S,
    PAD, PAD_S, CORNER,
    success_button,
    section_frame, label, status_bar,
)

# ── Preset directory ───────────────────────────────────────────────────────────
# Place .txt toolpath files here — they appear in the sidebar automatically.
PRESETS_DIR: Path = Path(__file__).parent.parent / "preset_toolpath"

# ── Composite overlay colours (auto-assigned per file, cycling) ───────────────
COMPOSITE_COLOURS = [
    "#534AB7",   # purple
    "#1D9E75",   # teal
    "#E9C46A",   # amber
    "#2A9D8F",   # teal-green
    "#E76F51",   # coral-orange
    "#457B9D",   # steel blue
    "#8338EC",   # violet
    "#06D6A0",   # mint
]


# ── nScrypt XYZ reconstructor ─────────────────────────────────────────────────

def _reconstruct_xyz(code: str) -> np.ndarray:
    """
    Parse nScrypt text → (N, 4) float array [abs_x, abs_y, abs_z, is_print].
    is_print = 1 for moves inside TrigValveRel…ValveRel, 0 otherwise.
    Relative moves by default; 'Absolute' command switches mode.
    """
    rows: list[list[float]] = [[0.0, 0.0, 0.0, 0.0]]
    x = y = z = 0.0
    printing = False
    absolute = False

    for raw in code.splitlines():
        s = raw.split(";")[0].strip()
        if not s or s.startswith("/"):
            continue
        sl = s.lower()

        if sl == "absolute":
            absolute = True
        elif sl == "relative":
            absolute = False
        elif sl == "trigvalverel":
            printing = True
        elif sl == "valverel":
            printing = False
        elif sl.startswith("move"):
            parts = sl.split()
            if len(parts) < 4:
                continue
            try:
                dx, dy, dz = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            flag = 1.0 if printing else 0.0
            if absolute:
                x, y, z = dx, dy, dz
            else:
                x += dx; y += dy; z += dz
            rows.append([x, y, z, flag])

    return np.array(rows, dtype=float) if len(rows) > 1 else np.zeros((1, 4))


def _split_segments(data: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """
    Split waypoint array into consecutive segments of constant is_print.
    Adjacent segments share a boundary XYZ connector row so drawn lines
    connect end-to-end with no gap.
    """
    if len(data) < 2:
        return []

    flags = data[:, 3].astype(int)
    transitions = [0]
    for i in range(1, len(flags)):
        if flags[i] != flags[i - 1]:
            transitions.append(i)
    transitions.append(len(data))

    segments: list[tuple[np.ndarray, bool]] = []
    for k in range(len(transitions) - 1):
        i0, i1 = transitions[k], transitions[k + 1]
        seg = data[i0:i1]
        isp = bool(flags[i0])
        if segments:
            prev_end = segments[-1][0][-1:].copy()
            prev_end[:, 3] = float(isp)
            seg = np.vstack([prev_end, seg])
        if len(seg) >= 2:
            segments.append((seg, isp))

    return segments


# ── Blank template ─────────────────────────────────────────────────────────────

_BLANK_TEMPLATE = """\
/ nScryptConformal v2 — new toolpath
/
speed 10

/ Initial positioning
move 0  0  10
move 0  0  -9.990

/ line 1
speed 5
trigwait 0.5
trigvalverel
move  0.000  0.000  0.000
move  0.000  -40.000  0.000
valverel
speed 10
move  0.000  0.000  2.000
move  1.000  40.000  0.000
move  0.000  0.000  -2.000

/ end of toolpath
"""


# ── Main window ────────────────────────────────────────────────────────────────

class ToolpathGenWindow(ctk.CTkToplevel):

    _SIDEBAR_W = 230   # fixed sidebar width (px)

    def __init__(self, parent):
        super().__init__(parent)
        self.title(f"{APP_TITLE} — Toolpath Generator")
        self.geometry(TOOLPATH_GEN_SIZE)
        self.minsize(960, 560)

        self._xyz_data    = np.zeros((1, 4))
        self._view_mode   = tk.StringVar(value="3D")
        self._active_btn  = None   # highlighted sidebar item

        # Composite state: path → (BooleanVar, colour_str)
        self._composite: dict[Path, tuple[tk.BooleanVar, str]] = {}

        self._build_ui()
        self._load_new()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0, minsize=self._SIDEBAR_W)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=2)
        self.grid_rowconfigure(1, weight=1)

        # Header
        hdr = ctk.CTkFrame(self, corner_radius=0,
                           fg_color=COL_CORAL, height=60)
        hdr.grid(row=0, column=0, columnspan=3, sticky="ew")
        ctk.CTkLabel(hdr, text="Toolpath Generator",
                     font=FONT_TITLE, text_color="white").pack(
            side="left", padx=PAD)
        ctk.CTkLabel(hdr, text="Write · visualise · export",
                     font=FONT_SMALL, text_color="#FFDDCC").pack(
            side="left", padx=PAD_S)

        self._build_sidebar()
        self._build_editor()
        self._build_viewer()

        # Status bar
        self._status = status_bar(self)
        self._status.grid(row=2, column=0, columnspan=3,
                          sticky="ew", padx=PAD, pady=(0, PAD_S))

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        sb = section_frame(self, width=self._SIDEBAR_W)
        sb.grid(row=1, column=0, sticky="nsew",
                padx=(PAD, PAD_S), pady=PAD)
        sb.grid_propagate(False)
        sb.grid_columnconfigure(0, weight=1)
        # Row weights: New=0, Presets label=0, Presets list=3,
        #              Composite label=0, Composite list=1, Browse=0
        sb.grid_rowconfigure(2, weight=3)   # preset list expands most
        sb.grid_rowconfigure(4, weight=1)   # composite list expands some

        # ── New button ────────────────────────────────────────────────────
        ctk.CTkButton(
            sb, text="+ New",
            font=FONT_LABEL,
            fg_color=COL_CORAL, hover_color="#993C1D",
            corner_radius=CORNER, height=32,
            command=self._load_new,
        ).grid(row=0, column=0, padx=PAD_S, pady=(PAD, PAD_S), sticky="ew")

        # ── PRESETS label ─────────────────────────────────────────────────
        ctk.CTkLabel(
            sb, text="PRESETS",
            font=("Segoe UI", 9),
            text_color=("gray50", "gray60"),
            anchor="w",
        ).grid(row=1, column=0, padx=PAD_S + 2, pady=(2, 1), sticky="w")

        # Scrollable presets list
        self._preset_list = ctk.CTkScrollableFrame(
            sb, fg_color="transparent", corner_radius=0,
        )
        self._preset_list.grid(row=2, column=0, sticky="nsew",
                               padx=PAD_S, pady=(0, PAD_S))
        self._preset_list.grid_columnconfigure(0, weight=1)

        self._populate_presets()

        # ── COMPOSITE FILES label ─────────────────────────────────────────
        ctk.CTkLabel(
            sb, text="COMPOSITE FILES",
            font=("Segoe UI", 9),
            text_color=("gray50", "gray60"),
            anchor="w",
        ).grid(row=3, column=0, padx=PAD_S + 2, pady=(4, 1), sticky="w")

        # Scrollable composite active-list
        self._composite_list = ctk.CTkScrollableFrame(
            sb, fg_color="transparent", corner_radius=0,
        )
        self._composite_list.grid(row=4, column=0, sticky="nsew",
                                  padx=PAD_S, pady=(0, PAD_S))
        self._composite_list.grid_columnconfigure(0, weight=1)
        self._composite_empty_lbl = ctk.CTkLabel(
            self._composite_list,
            text="Tick presets above\nto overlay them",
            font=FONT_SMALL,
            text_color=("gray50", "gray60"),
            justify="left", anchor="w",
            wraplength=self._SIDEBAR_W - 28,
        )
        self._composite_empty_lbl.grid(row=0, column=0, sticky="w",
                                       padx=4, pady=PAD_S)

        # ── Browse file button ────────────────────────────────────────────
        ctk.CTkButton(
            sb, text="Browse file…",
            font=FONT_SMALL,
            fg_color="transparent", border_width=1,
            corner_radius=CORNER, height=28,
            command=self._open_file,
        ).grid(row=5, column=0, padx=PAD_S, pady=(0, PAD), sticky="ew")

    def _populate_presets(self):
        """Scan PRESETS_DIR and build one checkbox row per .txt file."""
        for w in self._preset_list.winfo_children():
            w.destroy()

        presets = sorted(PRESETS_DIR.glob("*.txt")) \
                  if PRESETS_DIR.is_dir() else []

        if not presets:
            ctk.CTkLabel(
                self._preset_list,
                text="No presets.\nAdd .txt files to:\npresets/",
                font=FONT_SMALL,
                text_color=("gray50", "gray60"),
                justify="left", anchor="w",
                wraplength=self._SIDEBAR_W - 28,
            ).grid(row=0, column=0, padx=4, pady=PAD_S, sticky="w")
            return

        # Assign a colour to each preset (cycling)
        for i, path in enumerate(presets):
            colour = COMPOSITE_COLOURS[i % len(COMPOSITE_COLOURS)]

            # Ensure composite state exists for this path
            if path not in self._composite:
                self._composite[path] = (tk.BooleanVar(value=False), colour)

            var, col = self._composite[path]

            row_frame = ctk.CTkFrame(
                self._preset_list, fg_color="transparent")
            row_frame.grid(row=i, column=0, sticky="ew", pady=1)
            row_frame.grid_columnconfigure(1, weight=1)

            # Colour swatch  (small square)
            swatch = tk.Canvas(
                row_frame, width=10, height=10,
                bg=col, highlightthickness=0, relief="flat",
            )
            swatch.grid(row=0, column=0, padx=(2, 4), pady=4)

            # File name button (loads into editor)
            name = path.stem.replace("_", " ")
            btn = ctk.CTkButton(
                row_frame, text=name,
                font=FONT_SMALL, anchor="w",
                fg_color="transparent",
                text_color=("gray20", "gray90"),
                hover_color=("gray88", "gray25"),
                corner_radius=6, height=26,
            )
            btn.configure(command=lambda p=path, b=btn: self._load_preset(p, b))
            btn.grid(row=0, column=1, sticky="ew")

            # Composite checkbox
            cb = ctk.CTkCheckBox(
                row_frame, text="", variable=var,
                width=20, height=20,
                checkbox_width=16, checkbox_height=16,
                corner_radius=3,
                command=self._update_composite_list,
            )
            cb.grid(row=0, column=2, padx=(2, 4))

    def _update_composite_list(self):
        """Refresh the composite active-files panel below the presets."""
        for w in self._composite_list.winfo_children():
            w.destroy()

        active = [(p, col) for p, (var, col) in self._composite.items()
                  if var.get()]

        if not active:
            self._composite_empty_lbl = ctk.CTkLabel(
                self._composite_list,
                text="Tick presets above\nto overlay them",
                font=FONT_SMALL,
                text_color=("gray50", "gray60"),
                justify="left", anchor="w",
                wraplength=self._SIDEBAR_W - 28,
            )
            self._composite_empty_lbl.grid(row=0, column=0,
                                           sticky="w", padx=4, pady=PAD_S)
            return

        for i, (path, col) in enumerate(active):
            row = ctk.CTkFrame(
                self._composite_list, fg_color="transparent")
            row.grid(row=i, column=0, sticky="ew", pady=1)
            row.grid_columnconfigure(1, weight=1)

            # Colour swatch
            tk.Canvas(
                row, width=10, height=10,
                bg=col, highlightthickness=0,
            ).grid(row=0, column=0, padx=(2, 4), pady=3)

            ctk.CTkLabel(
                row, text=path.stem.replace("_", " "),
                font=FONT_SMALL, anchor="w",
                text_color=("gray20", "gray90"),
            ).grid(row=0, column=1, sticky="w")

    def _highlight_item(self, btn: ctk.CTkButton | None):
        if self._active_btn and self._active_btn.winfo_exists():
            self._active_btn.configure(
                fg_color="transparent",
                text_color=("gray20", "gray90"),
            )
        self._active_btn = btn
        if btn:
            btn.configure(
                fg_color=("#FAECE7", "#3A1A10"),
                text_color=(COL_CORAL, "#FFAA88"),
            )

    # ── Editor ────────────────────────────────────────────────────────────────

    def _build_editor(self):
        ed = section_frame(self)
        ed.grid(row=1, column=1, sticky="nsew", padx=PAD_S, pady=PAD)
        ed.grid_columnconfigure(1, weight=1)
        ed.grid_rowconfigure(1, weight=1)

        # Title + filename
        top = ctk.CTkFrame(ed, fg_color="transparent")
        top.grid(row=0, column=0, columnspan=3,
                 sticky="ew", padx=PAD, pady=(PAD, PAD_S))
        label(top, "Code editor", style="title").pack(side="left")
        self._file_label = ctk.CTkLabel(
            top, text="new file",
            font=FONT_SMALL, text_color=("gray50", "gray60"),
        )
        self._file_label.pack(side="left", padx=(8, 0), anchor="s", pady=(0, 2))

        # Line numbers
        self._line_nums = tk.Text(
            ed, width=4, font=FONT_MONO_S,
            state="disabled", bg="#F0F0F0", fg="#888888",
            relief="flat", bd=0, highlightthickness=0,
        )
        self._line_nums.grid(row=1, column=0, sticky="ns",
                             padx=(PAD_S, 0), pady=(0, PAD_S))

        # Code editor
        self._editor = tk.Text(
            ed, font=FONT_MONO, wrap="none",
            undo=True, relief="flat",
            bg="white", fg="#1a1a1a",
            insertbackground="#534AB7",
            selectbackground="#CECBF6",
        )
        self._editor.grid(row=1, column=1, sticky="nsew",
                          pady=(0, PAD_S))
        self._editor.bind("<KeyRelease>", self._on_keyrelease)
        self._editor.bind("<MouseWheel>", self._on_scroll)

        vsb = ctk.CTkScrollbar(ed, command=self._sync_scroll)
        vsb.grid(row=1, column=2, sticky="ns", pady=(0, PAD_S))
        self._editor.configure(yscrollcommand=vsb.set)

        # Bottom toolbar (Clear only — Refresh + Export are in viewer bar)
        tb = ctk.CTkFrame(ed, fg_color="transparent")
        tb.grid(row=2, column=0, columnspan=3,
                sticky="ew", padx=PAD, pady=(0, PAD_S))
        ctk.CTkButton(
            tb, text="Clear", font=FONT_SMALL,
            fg_color="transparent", border_width=1,
            corner_radius=CORNER, height=28, width=60,
            command=self._clear_editor,
        ).pack(side="left")

    # ── Viewer ────────────────────────────────────────────────────────────────

    def _build_viewer(self):
        vf = section_frame(self)
        vf.grid(row=1, column=2, sticky="nsew",
                padx=(0, PAD), pady=PAD)
        vf.grid_columnconfigure(0, weight=1)
        vf.grid_rowconfigure(1, weight=1)

        # View controls
        ctrl = ctk.CTkFrame(vf, fg_color="transparent")
        ctrl.grid(row=0, column=0, sticky="ew",
                  padx=PAD, pady=(PAD, PAD_S))
        label(ctrl, "View:", style="small").pack(side="left")
        for v in ["3D", "XY", "XZ", "YZ"]:
            ctk.CTkRadioButton(
                ctrl, text=v, variable=self._view_mode,
                value=v, font=FONT_SMALL,
                command=None,   # no auto-refresh on view change either
            ).pack(side="left", padx=PAD_S)

        self._highlight_print = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            ctrl, text="Highlight print",
            variable=self._highlight_print,
            font=FONT_SMALL,
            command=None,       # manual refresh only
        ).pack(side="left", padx=(PAD, 0))

        # Matplotlib figure
        self._fig = plt.figure(figsize=(6, 5), tight_layout=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=vf)
        self._canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew",
                                          padx=PAD_S, pady=PAD_S)

        tb_mpl = ctk.CTkFrame(vf, fg_color="transparent", height=28)
        tb_mpl.grid(row=2, column=0, sticky="ew", padx=PAD_S)
        NavigationToolbar2Tk(self._canvas, tb_mpl).update()

        # Bottom bar: Refresh + Export
        bot = ctk.CTkFrame(vf, fg_color="transparent")
        bot.grid(row=3, column=0, sticky="ew",
                 padx=PAD, pady=(PAD_S, PAD))

        ctk.CTkButton(
            bot, text="⟳  Refresh",
            font=FONT_LABEL,
            fg_color="#534AB7", hover_color="#3C3489",
            corner_radius=CORNER, height=32, width=110,
            command=self._refresh,
        ).pack(side="left", padx=(0, PAD_S))

        success_button(
            bot, "Export .txt…",
            command=self._export, width=110,
        ).pack(side="left")

    # ── File loading ──────────────────────────────────────────────────────────

    def _load_new(self):
        self._editor.delete("1.0", "end")
        self._editor.insert("1.0", _BLANK_TEMPLATE)
        self._file_label.configure(text="new file")
        self._highlight_item(None)
        self._update_line_numbers()
        self._set_status("New file — press Refresh to visualise")

    def _load_preset(self, path: Path, btn: ctk.CTkButton | None = None):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            messagebox.showerror("Error", f"Cannot read preset:\n{e}")
            return
        self._editor.delete("1.0", "end")
        self._editor.insert("1.0", text)
        self._file_label.configure(text=path.name)
        self._highlight_item(btn)
        self._update_line_numbers()
        self._set_status(f"Loaded: {path.name} — press Refresh to visualise")

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open nScrypt file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            messagebox.showerror("Error", f"Cannot read file:\n{e}")
            return
        self._editor.delete("1.0", "end")
        self._editor.insert("1.0", text)
        self._file_label.configure(text=p.name)
        self._highlight_item(None)
        self._update_line_numbers()
        self._set_status(f"Opened: {p.name} — press Refresh to visualise")

    def _clear_editor(self):
        if messagebox.askyesno("Clear", "Clear the editor?"):
            self._editor.delete("1.0", "end")
            self._file_label.configure(text="new file")
            self._highlight_item(None)
            self._update_line_numbers()
            self._set_status("Editor cleared — press Refresh to update plot")

    def _export(self):
        path = filedialog.asksaveasfilename(
            title="Export toolpath",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            Path(path).write_text(
                self._editor.get("1.0", "end-1c"), encoding="utf-8")
            self._set_status(f"Exported: {Path(path).name}")

    # ── Editor helpers ────────────────────────────────────────────────────────

    def _on_keyrelease(self, event=None):
        # Only update line numbers on keystroke — no plot refresh
        self._update_line_numbers()

    def _on_scroll(self, event=None):
        self._update_line_numbers()

    def _sync_scroll(self, *args):
        self._editor.yview(*args)
        self._update_line_numbers()

    def _update_line_numbers(self):
        content = self._editor.get("1.0", "end-1c")
        n = content.count("\n") + 1
        nums = "\n".join(str(i) for i in range(1, n + 1))
        self._line_nums.configure(state="normal")
        self._line_nums.delete("1.0", "end")
        self._line_nums.insert("1.0", nums)
        self._line_nums.configure(state="disabled")
        self._line_nums.yview_moveto(self._editor.yview()[0])

    # ── Refresh & plot ────────────────────────────────────────────────────────

    def _refresh(self):
        """
        Re-parse the editor content and all checked composite files,
        then redraw the plot.  This is the ONLY path that updates the plot.
        """
        code = self._editor.get("1.0", "end-1c")
        self._xyz_data = _reconstruct_xyz(code)
        d = self._xyz_data

        n_moves  = len(d) - 1
        n_print  = int(d[:, 3].sum())
        n_travel = n_moves - n_print

        active_composite = [
            (p, col) for p, (var, col) in self._composite.items()
            if var.get()
        ]

        status_parts = []
        if n_moves > 0:
            status_parts.append(
                f"Editor: {n_moves} moves ({n_print} print, {n_travel} travel)  "
                f"X:[{d[:,0].min():.2f},{d[:,0].max():.2f}]  "
                f"Y:[{d[:,1].min():.2f},{d[:,1].max():.2f}]  "
                f"Z:[{d[:,2].min():.3f},{d[:,2].max():.3f}]"
            )
        if active_composite:
            status_parts.append(
                f"+ {len(active_composite)} composite file(s)"
            )
        self._set_status("  |  ".join(status_parts) if status_parts
                         else "No moves — nothing to plot")

        self._draw_plot(active_composite)

    def _draw_plot(self, composite: list[tuple[Path, str]]):
        self._fig.clf()
        vm = self._view_mode.get()
        hp = self._highlight_print.get()
        d  = self._xyz_data

        if vm == "3D":
            ax = self._fig.add_subplot(111, projection="3d")
            ax.set_xlabel("X (mm)", fontsize=8, labelpad=3)
            ax.set_ylabel("Y (mm)", fontsize=8, labelpad=3)
            ax.set_zlabel("Z (mm)", fontsize=8, labelpad=3)
        else:
            ax = self._fig.add_subplot(111)
            xi, yi, lx, ly = {
                "XY": (0, 1, "X (mm)", "Y (mm)"),
                "XZ": (0, 2, "X (mm)", "Z (mm)"),
                "YZ": (1, 2, "Y (mm)", "Z (mm)"),
            }[vm]
            ax.set_xlabel(lx, fontsize=8)
            ax.set_ylabel(ly, fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.3)

        legend_handles: list = []

        # ── Editor toolpath (primary, coral print colour) ─────────────────
        if len(d) >= 2:
            segs = _split_segments(d)
            print_col = "#D85A30" if hp else "#888888"
            self._draw_segments(ax, segs, print_col, vm,
                                xi if vm != "3D" else 0,
                                yi if vm != "3D" else 1,
                                label_prefix="Editor")
            legend_handles += [
                plt.Line2D([0],[0], color=print_col, lw=1.8,
                           label="Editor — print"),
                plt.Line2D([0],[0], color="#BBBBBB", lw=0.8,
                           linestyle=":", label="Travel"),
            ]

        # ── Composite files (one colour each) ─────────────────────────────
        for path, col in composite:
            try:
                code = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            dc = _reconstruct_xyz(code)
            if len(dc) < 2:
                continue
            segs_c = _split_segments(dc)
            self._draw_segments(ax, segs_c, col, vm,
                                xi if vm != "3D" else 0,
                                yi if vm != "3D" else 1,
                                label_prefix=path.stem[:12])
            legend_handles.append(
                plt.Line2D([0],[0], color=col, lw=1.8,
                           label=path.stem.replace("_", " ")[:18])
            )

        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=7,
                      loc="upper right")

        if vm == "3D":
            ax.set_title("3D toolpath", fontsize=9)
        else:
            ax.set_aspect("equal" if vm == "XY" else "auto")
            ax.tick_params(labelsize=7)

        self._canvas.draw_idle()

    def _draw_segments(self, ax, segments, print_col: str,
                       vm: str, xi: int, yi: int,
                       label_prefix: str = ""):
        """Draw one set of segments (editor or composite file) onto ax."""
        travel_col = "#CCCCCC"
        for seg, is_print in segments:
            col = print_col if is_print else travel_col
            lw  = 1.8       if is_print else 0.8
            ls  = "-"       if is_print else ":"
            al  = 1.0       if is_print else 0.6

            if vm == "3D":
                ax.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                        color=col, linewidth=lw, linestyle=ls,
                        alpha=al, zorder=3 if is_print else 1)
            else:
                ax.plot(seg[:, xi], seg[:, yi],
                        color=col, linewidth=lw, linestyle=ls,
                        alpha=al, zorder=3 if is_print else 1)

    # ── Status ────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._status.configure(text=msg)
        self.update_idletasks()
