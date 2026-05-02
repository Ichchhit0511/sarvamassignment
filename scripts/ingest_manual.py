"""CLI helper: ingest a PDF manual without going through the web UI.

Usage:
    python -m scripts.ingest_manual path/to/manual.pdf royal_enfield_classic_350
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ingest import ingest_pdf  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python -m scripts.ingest_manual <pdf_path> <manual_id>")
        sys.exit(1)
    pdf_path = Path(sys.argv[1])
    manual_id = sys.argv[2]
    result = ingest_pdf(pdf_path, manual_id)
    print(result)


if __name__ == "__main__":
    main()
