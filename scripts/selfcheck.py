"""
Run the core math tests WITHOUT needing pytest, torch, or a GPU.

    python scripts/selfcheck.py

This exercises every pure-NumPy kernel (PCA confound removal, the emotion- and
deflection-vector recipes, the probe projection, Elo, the visualiser ranking,
and the exact token-span mapping). It is the in-repo demonstration that the math
is correct — the part of the pipeline you can verify on a laptop. The heavy
model/extraction code is validated separately on a GPU machine.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

# Make the repo importable when run as `python scripts/selfcheck.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _discover_tests() -> list[tuple[str, object]]:
    """Find every ``test_*`` function in every ``tests/test_*.py`` module."""
    found: list[tuple[str, object]] = []
    for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        module = importlib.import_module(f"tests.{path.stem}")
        for name, fn in vars(module).items():
            if name.startswith("test_") and callable(fn):
                found.append((f"{path.stem}.{name}", fn))
    return found


def main() -> int:
    tests = _discover_tests()
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001 - we want to report any failure
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"ok    {name}")
    print(f"\n{passed} passed, {failed} failed, out of {len(tests)} core tests.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
