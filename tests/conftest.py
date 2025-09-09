import sys, pathlib
# Add the repo's ./src directory so "import mfd4" works in tests
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))
