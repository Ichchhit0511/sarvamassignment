"""Seed the index with the synthetic chunks used by the golden eval set."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.ingest import _save_bm25, _save_chroma
from backend.models import Chunk


CHUNKS = [
    Chunk(
        chunk_id="c001", manual_id="re_classic_350", page=42,
        section="Troubleshooting > Engine > White Smoke",
        text=("WHITE SMOKE FROM EXHAUST. Possible causes: (1) Coolant entering "
              "combustion chamber due to head gasket failure. (2) Condensation "
              "in exhaust during cold start (normal, dissipates in 2-3 minutes). "
              "Action: If smoke persists after engine warms up, stop riding and "
              "consult an authorized service center. Do not continue operation as "
              "this may damage the catalytic converter."),
        component_tags=["engine", "cooling", "exhaust"], symptom_tags=["smoke"],
    ),
    Chunk(
        chunk_id="c002", manual_id="re_classic_350", page=43,
        section="Troubleshooting > Engine > Black Smoke",
        text=("BLACK SMOKE FROM EXHAUST. Indicates rich fuel mixture. Check air "
              "filter for clogging. Replace if dirty. Verify fuel injector "
              "cleanliness. Persistent black smoke requires ECU diagnostic at "
              "service center."),
        component_tags=["engine", "fuel", "exhaust"], symptom_tags=["smoke"],
    ),
    Chunk(
        chunk_id="c003", manual_id="re_classic_350", page=18,
        section="Maintenance > Cooling System",
        text=("COOLANT LEVEL CHECK. Inspect coolant reservoir weekly. Top up "
              "with manufacturer-specified coolant only. Check radiator hoses "
              "for cracks or leaks. Capacity: 1.2 litres. Service interval: "
              "every 12,000 km."),
        component_tags=["cooling"], symptom_tags=["leak"],
    ),
    Chunk(
        chunk_id="c004", manual_id="re_classic_350", page=71,
        section="Brakes > Pad Replacement",
        text=("BRAKE PAD REPLACEMENT. Loosen caliper bolts. Remove old pads. "
              "Inspect rotor for scoring. Install new pads ensuring chamfered "
              "edge faces rotation. Torque caliper bolts to 28 Nm. Bed in pads "
              "with 10 moderate stops."),
        component_tags=["brakes"], symptom_tags=[],
    ),
]


def main() -> None:
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    if settings.bm25_index_path.exists():
        settings.bm25_index_path.unlink()
    _save_chroma(CHUNKS)
    _save_bm25(CHUNKS)
    print(f"✓ Seeded {len(CHUNKS)} chunks into ChromaDB + BM25.")


if __name__ == "__main__":
    main()
