"""Pre-commit guard: if model/feature-producing files are staged but the
generated plot PNGs are not, the PNGs are about to become stale.

Block the commit and tell the user to run `python -m eval.plot_metrics` and
re-stage the PNGs. Allow override with `--no-verify` per standard git.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLOT_FILES = ["docs/confusion.png", "docs/roc_pr.png"]
PLOT_INPUTS = [
    "app/features.py",
    "app/heuristic.py",
    "app/parser.py",
    "app/classifier_model.pkl",
]


def staged() -> set[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True, check=True, cwd=ROOT,
    )
    return {line.strip().replace("\\", "/") for line in out.stdout.splitlines() if line.strip()}


def main() -> int:
    s = staged()
    inputs_changed = s & set(PLOT_INPUTS)
    plots_changed = s & set(PLOT_FILES)

    if inputs_changed and not plots_changed:
        print("\033[33mWarning: model or feature code is being committed without "
              "regenerating the figures.\033[0m", file=sys.stderr)
        print(f"  Staged inputs: {sorted(inputs_changed)}", file=sys.stderr)
        print(f"  Run: python -m eval.plot_metrics && git add {' '.join(PLOT_FILES)}",
              file=sys.stderr)
        print("  (use `git commit --no-verify` to override)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
