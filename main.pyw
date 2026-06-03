"""
main.pyw
========
Entry point for nScryptConformal v2.

Called by launch.bat (itself called silently by JobCommand.vbs).
launch.bat sets PYTHONPATH to include backend/, ui/, windows/ already.
The .pyw extension suppresses the Windows console window.
"""

import sys
import os

# ── Fallback path setup (if launched directly, not via launch.bat) ────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ("backend", "ui", "windows"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Dependency check ──────────────────────────────────────────────────────────
_REQUIRED = {
    "customtkinter": "pip install customtkinter",
    "numpy":         "pip install numpy",
    "scipy":         "pip install scipy",
    "matplotlib":    "pip install matplotlib",
    "pandas":        "pip install pandas",
    "openpyxl":      "pip install openpyxl",
}

_missing = []
for _pkg, _install in _REQUIRED.items():
    try:
        __import__(_pkg)
    except ImportError:
        _missing.append((_pkg, _install))

if _missing:
    import tkinter as tk
    from tkinter import messagebox
    _root = tk.Tk()
    _root.withdraw()
    _msg = (
        "Missing required packages:\n\n"
        + "\n".join(f"  {pkg}   ->   {cmd}" for pkg, cmd in _missing)
        + "\n\nOr activate the virtual environment and run:\n"
        "  .venv\\Scripts\\pip install "
        + " ".join(p for p, _ in _missing)
    )
    messagebox.showerror("nScryptConformal v2 - Missing Dependencies", _msg)
    sys.exit(1)

# ── Launch ────────────────────────────────────────────────────────────────────
import customtkinter as ctk
from hub import HubWindow

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

if __name__ == "__main__":
    app = HubWindow()
    app.mainloop()
