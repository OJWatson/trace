from __future__ import annotations

import sys
from pathlib import Path


# Ensure the project package in src/ is importable during tests.
# This avoids importing Python's stdlib module named "trace".
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Some environments may have already imported Python's stdlib module named
# "trace" before pytest collects our tests. If so, it will shadow the project
# package also named "trace". Purge that module (and any of its submodules)
# so imports resolve to our package.
_maybe_trace = sys.modules.get("trace")
if _maybe_trace is not None and not hasattr(_maybe_trace, "__path__"):
    for _k in list(sys.modules.keys()):
        if _k == "trace" or _k.startswith("trace."):
            sys.modules.pop(_k, None)
