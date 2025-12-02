import os
import sys

# Ensure the backend source directory is importable when running tests directly.
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("[conftest] added project root to sys.path", file=sys.stderr)
