# run_boot.py
import os, sys, traceback, datetime

#----------- Don't import pyarrow ----------------
os.environ.setdefault("PYARROW_IGNORE_IMPORT_ERROR", "1")
os.environ.setdefault("PANDAS_IGNORE_PYARROW", "1")

# ---------- figure out base dir next to the EXE ----------
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ---------- open a log file and redirect stdout/stderr ----------
log_path = os.path.join(LOG_DIR, "boot.log")
log_f = open(log_path, "w", encoding="utf-8", buffering=1, errors="replace")
sys.stdout = log_f
sys.stderr = log_f

print(f"[BOOT] starting at {datetime.datetime.now().isoformat()}")
print(f"[BOOT] BASE_DIR={BASE_DIR}")

# ---------- catch hard crashes too ----------
try:
    import faulthandler
    faulthandler.enable(log_f)  # dumps fatal errors to boot.log
    print("[BOOT] faulthandler enabled")
except Exception as e:
    print(f"[BOOT] faulthandler failed: {e}")

# ---------- make Qt very chatty about plugins ----------
os.environ.setdefault("QT_DEBUG_PLUGINS", "1")

# Optional: if OpenGL causes trouble, force software OpenGL
# os.environ.setdefault("QT_OPENGL", "software")

# ---------- global excepthook -> file ----------
def _excepthook(exctype, value, tb):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    crash_file = os.path.join(LOG_DIR, f"crash_{ts}.log")
    with open(crash_file, "w", encoding="utf-8") as f:
        traceback.print_exception(exctype, value, tb, file=f)
    print(f"[BOOT] Uncaught exception logged to {crash_file}")

sys.excepthook = _excepthook

# ---------- START APP ----------
print("[BOOT] importing app and launching...")
try:
    # Light probe: import PyQt6 early to force plugin load messages to appear in boot.log
    import PyQt6  # noqa: F401
    from finplan_suite.app import launch_app
    launch_app()
    print("[BOOT] app exited normally")
except Exception:
    _excepthook(*sys.exc_info())
    # Best-effort popup so double-click users see something
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showerror(
            "FinPlan Suite",
            f"An error occurred.\nSee log:\n{log_path}"
        )
    except Exception as e:
        print(f"[BOOT] failed to show error popup: {e}")
    sys.exit(1)
finally:
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass
