"""
windows/hub.py
==============
The Hub window — the first thing the user sees after launching.
Four large buttons open the corresponding mode window as a Toplevel.
"""

from __future__ import annotations
import customtkinter as ctk
from ui.theme import (
    APP_TITLE, HUB_SIZE,
    COL_ACCENT, COL_TEAL, COL_AMBER, COL_CORAL,
    FONT_TITLE, FONT_LABEL, FONT_SMALL,
    PAD, PAD_S, CORNER,
)


class HubWindow(ctk.CTk):
    """Main application window — hosts the four mode buttons."""

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(HUB_SIZE)
        self.resizable(False, False)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=PAD, pady=(PAD, 0))

        ctk.CTkLabel(
            hdr, text="nScryptConformal", font=("Segoe UI", 22, "bold"),
        ).pack(side="left")
        ctk.CTkLabel(
            hdr, text="v2.0", font=("Segoe UI", 13),
            text_color=("gray50", "gray60"),
        ).pack(side="left", padx=(6, 0), anchor="s", pady=(0, 3))

        # ── Button grid ───────────────────────────────────────────────────────
        grid = ctk.CTkFrame(self, fg_color="transparent")
        grid.grid(row=1, column=0, sticky="nsew", padx=PAD, pady=PAD)
        grid.grid_columnconfigure((0, 1), weight=1)
        grid.grid_rowconfigure((0, 1), weight=1)

        BTN_H = 100

        # Button spec: (text, subtitle, colour, hover, row, col, handler)
        buttons = [
            (
                "Conformal toolpath",
                "Single scan · save preset",
                COL_ACCENT, "#3C3489",
                0, 0, self._open_conformal,
            ),
            (
                "Batch conformal",
                "Batch run from saved preset",
                COL_AMBER, "#854F0B",
                0, 1, self._open_batch_conformal,
            ),
            (
                "Curved toolpath",
                "Single scan · curved paths",
                COL_TEAL, "#0F6E56",
                1, 0, self._open_curved,
            ),
            (
                "Toolpath generator",
                "Write · visualise · export",
                COL_CORAL, "#993C1D",
                1, 1, self._open_toolpath_gen,
            ),
        ]

        for text, subtitle, fg, hover, row, col, cmd in buttons:
            frame = ctk.CTkFrame(
                grid, corner_radius=CORNER,
                fg_color=fg, cursor="hand2",
            )
            frame.grid(row=row, column=col, padx=PAD_S, pady=PAD_S,
                       sticky="nsew")
            frame.grid_propagate(False)
            frame.configure(height=BTN_H)

            ctk.CTkLabel(
                frame, text=text,
                font=("Segoe UI", 13, "bold"),
                text_color="white",
            ).pack(pady=(18, 2))
            ctk.CTkLabel(
                frame, text=subtitle,
                font=FONT_SMALL,
                text_color=("white", "#DDDDDD"),
            ).pack()

            # Bind click on both frame and labels
            for w in [frame] + list(frame.winfo_children()):
                w.bind("<Button-1>", lambda e, c=cmd: c())
                w.bind("<Enter>", lambda e, f=frame, h=hover: f.configure(fg_color=h))
                w.bind("<Leave>", lambda e, f=frame, c=fg:   f.configure(fg_color=c))

        # ── Footer ────────────────────────────────────────────────────────────
        ctk.CTkLabel(
            self,
            text="nScrypt conformal surface toolpath generator",
            font=FONT_SMALL,
            text_color=("gray50", "gray60"),
        ).grid(row=2, column=0, pady=(0, PAD_S))

    # ── Mode launchers ────────────────────────────────────────────────────────

    def _open_conformal(self):
        from windows.conformal import ConformalWindow
        ConformalWindow(self)

    def _open_batch_conformal(self):
        from windows.batch_conformal import BatchConformalWindow
        BatchConformalWindow(self)

    def _open_curved(self):
        from windows.curved import CurvedWindow
        CurvedWindow(self)

    def _open_toolpath_gen(self):
        from windows.toolpath_gen import ToolpathGenWindow
        ToolpathGenWindow(self)
