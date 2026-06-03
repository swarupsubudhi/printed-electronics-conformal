"""
windows/curved.py
=================
Curved Toolpath window — architecture identical to conformal.py but
configured for curved (non-flat) path geometries.

The curved toolpath algorithm is not yet implemented; this window is a
placeholder that exposes the same Load → Configure → Generate workflow
and will be populated in a future sprint.
"""

from __future__ import annotations
from tkinter import messagebox
import customtkinter as ctk

from ui.theme import (
    APP_TITLE, CURVED_SIZE, COL_TEAL,
    FONT_TITLE, FONT_LABEL, FONT_SMALL,
    PAD, PAD_S, CORNER,
    secondary_button, label, status_bar, section_frame,
)


class CurvedWindow(ctk.CTkToplevel):

    def __init__(self, parent):
        super().__init__(parent)
        self.title(f"{APP_TITLE} — Curved toolpath")
        self.geometry(CURVED_SIZE)
        self.minsize(600, 400)
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        hdr = ctk.CTkFrame(self, corner_radius=0,
                           fg_color=COL_TEAL, height=40)
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            hdr, text="Curved toolpath",
            font=FONT_TITLE, text_color="white",
        ).pack(side="left", padx=PAD)

        # Body
        body = section_frame(self)
        body.grid(row=1, column=0, sticky="nsew",
                  padx=PAD*2, pady=PAD*2)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=1)

        label(body, "Curved toolpath — coming soon",
              style="title").grid(row=0, column=0,
                                  padx=PAD, pady=PAD, sticky="w")

        ctk.CTkLabel(
            body,
            text=(
                "The curved toolpath algorithm will support non-planar print "
                "paths that follow surface curvature in both X and Y.\n\n"
                "This window is a placeholder.  The workflow will mirror the "
                "Conformal toolpath window:\n\n"
                "  1.  Load scan.xyz  +  code.txt\n"
                "  2.  2D top view with path selection\n"
                "  3.  Configure parameters + algorithm\n"
                "  4.  Generate  *_curved.txt"
            ),
            font=FONT_SMALL,
            justify="left",
            wraplength=520,
        ).grid(row=1, column=0, padx=PAD_S*3, pady=PAD_S,
               sticky="nw")

        secondary_button(
            body, "Close",
            command=self.destroy,
            width=120,
        ).grid(row=2, column=0, pady=PAD)

        status_bar(self).grid(row=2, column=0, sticky="ew",
                              padx=PAD, pady=(0, PAD_S))
