"""Fig 1 / figM0 entry point: runs figM0_schematic.py (the real a-f SVG plate)."""
import runpy
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "figM0_schematic.py"
runpy.run_path(str(TARGET), run_name="__main__")
