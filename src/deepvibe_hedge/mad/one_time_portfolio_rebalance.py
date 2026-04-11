"""
One-shot MAD / MRAT reconcile: same as ``live_bot`` with ``--once``.

Use this when you want to enter or align positions **immediately** (for example you start the
stack mid-afternoon). The long-running ``live_bot`` only runs its scheduled rebalance **after**
that session’s official close (unless you use ``--once`` there too).
"""
from __future__ import annotations

import sys

from deepvibe_hedge.mad import live_bot


def main() -> None:
    prog = sys.argv[0]
    rest = [a for a in sys.argv[1:] if a != "--once"]
    sys.argv = [prog, "--once", *rest]
    live_bot.main()


if __name__ == "__main__":
    main()
