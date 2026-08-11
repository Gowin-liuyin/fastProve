"""Allow ``python -m evals`` as an alias for ``python -m evals.report``."""

from __future__ import annotations

import sys

from .report import main

if __name__ == "__main__":
    sys.exit(main())
