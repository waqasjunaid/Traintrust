"""
Environment doctor — diagnose setup issues before you waste time training.

Usage:
    python -m train_trust.scripts.doctor

Checks:
    - Python version
    - All required packages importable
    - PyTorch version + CUDA availability + driver compatibility
    - Joern present (if USE_JOERN=1)
    - Java version (Joern needs JDK 11+)
    - Repo files are consistent (no stale train.py etc.)
    - HuggingFace cache reachable (optional)

Exits with code 0 if everything looks good, 1 if any check fails.
"""

from __future__ import annotations
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
RESET  = "\033[0m"

ok_count = 0
warn_count = 0
err_count = 0


def ok(msg):     global ok_count;   ok_count += 1;   print(f"{GREEN}✓{RESET} {msg}")
def warn(msg):   global warn_count; warn_count += 1; print(f"{YELLOW}!{RESET} {msg}")
def err(msg):    global err_count;  err_count += 1;  print(f"{RED}✗{RESET} {msg}")


# -----------------------------------------------------------------------------
# Python + packages
# -----------------------------------------------------------------------------

def check_python():
    v = sys.version_info
    if v.major == 3 and v.minor >= 9:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        err(f"Python {v.major}.{v.minor}.{v.micro} — need 3.9+")


REQUIRED_PACKAGES = [
    ("torch",        "torch"),
    ("transformers", "transformers"),
    ("numpy",        "numpy"),
    ("pandas",       "pandas"),
    ("yaml",         "pyyaml"),
    ("sklearn",      "scikit-learn"),
    ("networkx",     "networkx"),
    ("tqdm",         "tqdm"),
    ("pytest",       "pytest"),
]


def check_packages():
    missing = []
    for import_name, pkg_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            ok(f"package {pkg_name}")
        except ImportError:
            err(f"package {pkg_name} MISSING — run: pip install {pkg_name}")
            missing.append(pkg_name)
    if missing:
        print(f"\n  Fix all at once with:  pip install {' '.join(missing)}\n")


# -----------------------------------------------------------------------------
# PyTorch + CUDA
# -----------------------------------------------------------------------------

def check_pytorch():
    try:
        import torch
        ok(f"PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            ok(f"CUDA available — {torch.cuda.get_device_name(0)}")
            ok(f"CUDA build: {torch.version.cuda}")
        else:
            # Distinguish "no GPU on this machine" vs "GPU but driver too old".
            try:
                # Force a CUDA call to see the actual reason.
                torch.cuda.init()
                warn("CUDA reports unavailable but no error raised")
            except RuntimeError as e:
                msg = str(e)
                if "driver" in msg.lower() and "old" in msg.lower():
                    warn(
                        "CUDA driver is too old for this PyTorch build. "
                        "Either update your NVIDIA driver, or install PyTorch "
                        "for an older CUDA toolkit, e.g.:\n"
                        "    pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
                        "  Training will run on CPU until then (slow on BigVul)."
                    )
                else:
                    warn(f"No GPU detected — running on CPU. Detail: {msg[:120]}")
    except ImportError:
        err("PyTorch missing")


# -----------------------------------------------------------------------------
# Joern + Java
# -----------------------------------------------------------------------------

def check_joern():
    use_joern = os.environ.get("USE_JOERN", "0") == "1"
    if not use_joern:
        warn("USE_JOERN is not set — PDG will use the heuristic stub. "
             "For real experiments: export USE_JOERN=1 (after installing Joern).")
        return

    if not shutil.which("joern-parse"):
        err("USE_JOERN=1 but joern-parse not on PATH. Install: "
            "https://docs.joern.io/installation")
        return

    try:
        out = subprocess.check_output(["joern", "--version"],
                                      stderr=subprocess.STDOUT, timeout=10).decode()
        ok(f"Joern: {out.strip().splitlines()[0]}")
    except Exception as e:
        warn(f"Joern present but `joern --version` failed: {e!r}")

    # Java check
    try:
        out = subprocess.check_output(["java", "-version"],
                                      stderr=subprocess.STDOUT, timeout=10).decode()
        first = out.strip().splitlines()[0]
        ok(f"Java: {first}")
        # Try to extract major version
        import re
        m = re.search(r'"(\d+)', first)
        if m:
            major = int(m.group(1))
            if major < 11:
                err(f"Java {major} too old; Joern needs 11+")
    except FileNotFoundError:
        err("java not on PATH — Joern needs JDK 11+")
    except Exception as e:
        warn(f"java check failed: {e!r}")


# -----------------------------------------------------------------------------
# Repo file consistency
# -----------------------------------------------------------------------------

def check_repo_files():
    here = Path(__file__).resolve().parent.parent.parent  # repo root
    train_py = here / "train_trust" / "train.py"
    if not train_py.exists():
        err(f"missing {train_py}")
        return

    body = train_py.read_text()
    if "_build_datasets" in body:
        ok("train.py has BigVul dispatcher (current)")
    else:
        err(
            "train.py is STALE — missing _build_datasets dispatcher.\n"
            "  Re-extract the latest tar over your repo:\n"
            "      tar -xzf train-trust.tar.gz --overwrite -C ..\n"
            "  Or re-download and replace this file.")

    bigvul_py = here / "train_trust" / "data" / "bigvul.py"
    if not bigvul_py.exists() or bigvul_py.stat().st_size < 100:
        err(f"missing or tiny {bigvul_py} — extraction was incomplete")
    else:
        ok("data/bigvul.py present")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" Train-Trust environment doctor")
    print("=" * 60)
    check_python()
    check_packages()
    check_pytorch()
    check_joern()
    check_repo_files()
    print("=" * 60)
    print(f" {ok_count} ok   {warn_count} warn   {err_count} err")
    print("=" * 60)
    sys.exit(1 if err_count else 0)


if __name__ == "__main__":
    main()
