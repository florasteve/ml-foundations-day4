#!/usr/bin/env bash
set -euo pipefail
export MPLBACKEND=Agg
PYTHONPATH=src python examples/quadratic_contours.py
PYTHONPATH=src python examples/lr_sweep.py
PYTHONPATH=src python examples/surface3d.py
PYTHONPATH=src python examples/logreg_surface.py
echo "All figures regenerated in ./figures"
