"""
ui/theme.py
===========
Shared visual constants, colour palette, and widget factory helpers
used across all nScryptConformal v2 windows.

All windows import from here so a single change propagates everywhere.
"""

from __future__ import annotations
import customtkinter as ctk

# ── App metadata ──────────────────────────────────────────────────────────────
APP_NAME    = "nScryptConformal"
APP_VERSION = "v2.0"
APP_TITLE   = f"{APP_NAME} {APP_VERSION}"

# ── Window geometry ───────────────────────────────────────────────────────────
HUB_SIZE         = "480x320"
CONFORMAL_SIZE   = "1280x820"
BATCH_SIZE       = "900x640"
CURVED_SIZE      = "1100x760"
TOOLPATH_GEN_SIZE= "1200x820"

# ── Colours (hex, works in both light and dark appearance modes) ──────────────
# These are used for matplotlib figures and custom canvas elements only.
# CTk widgets use the CTk theme system automatically.

COL_ACCENT       = "#534AB7"   # purple — primary action
COL_ACCENT_LIGHT = "#EEEDFE"
COL_TEAL         = "#1D9E75"   # success / generated
COL_TEAL_LIGHT   = "#E1F5EE"
COL_AMBER        = "#BA7517"   # warning / batch
COL_AMBER_LIGHT  = "#FAEEDA"
COL_CORAL        = "#D85A30"   # toolpath generator
COL_CORAL_LIGHT  = "#FAECE7"
COL_GRAY         = "#5F5E5A"
COL_GRAY_LIGHT   = "#F1EFE8"

# Plot colours
PLOT_SCAN_LINE   = "#378ADD"   # blue  — scan surface polylines
PLOT_CODE_PATH   = "#E24B4A"   # red   — code.txt path footprint
PLOT_SELECTED    = "#EF9F27"   # amber — selected/active path
PLOT_SPLINE      = "#1D9E75"   # teal  — fitted surface spline
PLOT_TOOLPATH    = "#D85A30"   # coral — conformed toolpath
PLOT_TOL_BAND    = "#E24B4A"   # red   — ±tolerance band (alpha=0.12)

# ── Typography ────────────────────────────────────────────────────────────────
FONT_TITLE  = ("Segoe UI", 14, "bold")
FONT_LABEL  = ("Segoe UI", 11)
FONT_SMALL  = ("Segoe UI", 10)
FONT_MONO   = ("Consolas",  11)
FONT_MONO_S = ("Consolas",  10)

# ── Padding / spacing ─────────────────────────────────────────────────────────
PAD        = 12    # standard outer pad
PAD_S      = 6     # small internal pad
PAD_L      = 20    # large section pad
CORNER     = 8     # CTk corner radius for frames/buttons

# ── CTk widget factories ──────────────────────────────────────────────────────

def section_frame(parent, **kw) -> ctk.CTkFrame:
    """Raised card-style container."""
    return ctk.CTkFrame(parent, corner_radius=CORNER, **kw)


def label(parent, text: str, style: str = "normal", **kw) -> ctk.CTkLabel:
    """
    Convenience label.  style: 'normal' | 'title' | 'small' | 'muted'
    """
    fonts = {
        "title":  FONT_TITLE,
        "normal": FONT_LABEL,
        "small":  FONT_SMALL,
        "muted":  FONT_SMALL,
    }
    colours = {
        "muted": ("gray50", "gray60"),
    }
    f = fonts.get(style, FONT_LABEL)
    tc = colours.get(style, None)
    kw.setdefault("font", f)
    if tc:
        kw.setdefault("text_color", tc)
    return ctk.CTkLabel(parent, text=text, **kw)


def primary_button(parent, text: str, command=None, **kw) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent, text=text, command=command,
        corner_radius=CORNER, font=FONT_LABEL,
        fg_color=COL_ACCENT, hover_color="#3C3489",
        **kw,
    )


def secondary_button(parent, text: str, command=None, **kw) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent, text=text, command=command,
        corner_radius=CORNER, font=FONT_LABEL,
        fg_color="transparent", border_width=1,
        **kw,
    )


def success_button(parent, text: str, command=None, **kw) -> ctk.CTkButton:
    return ctk.CTkButton(
        parent, text=text, command=command,
        corner_radius=CORNER, font=FONT_LABEL,
        fg_color=COL_TEAL, hover_color="#0F6E56",
        **kw,
    )


def param_row(parent, label_text: str, row: int,
              variable=None, width: int = 100,
              from_: float = 0, to: float = 100,
              widget: str = "entry") -> ctk.CTkEntry | ctk.CTkOptionMenu:
    """
    Place a label + input widget in a grid row.
    widget: 'entry' | 'optionmenu'
    """
    ctk.CTkLabel(parent, text=label_text, font=FONT_LABEL,
                 anchor="w").grid(row=row, column=0, sticky="w",
                                  padx=PAD_S, pady=3)
    if widget == "entry":
        w = ctk.CTkEntry(parent, width=width, font=FONT_MONO_S,
                         textvariable=variable)
        w.grid(row=row, column=1, sticky="w", padx=PAD_S, pady=3)
        return w
    if widget == "optionmenu":
        w = ctk.CTkOptionMenu(parent, variable=variable, width=width,
                              font=FONT_LABEL)
        w.grid(row=row, column=1, sticky="w", padx=PAD_S, pady=3)
        return w
    raise ValueError(f"Unknown widget type: {widget!r}")


def status_bar(parent) -> ctk.CTkLabel:
    """Footer status bar pinned to the bottom of a window."""
    bar = ctk.CTkLabel(
        parent, text="Ready", anchor="w",
        font=FONT_SMALL, text_color=("gray40", "gray60"),
        height=24,
    )
    return bar
