import sys
import os

# Add sagesim to Python path
sagesim_path = "/home/co1/sagesim_github/SAGESim"
if sagesim_path not in sys.path:
    sys.path.insert(0, sagesim_path)

# Verify sagesim is available
try:
    import sagesim

    print(f"✓ sagesim successfully loaded from: {sagesim.__file__}")
except ImportError as e:
    print(f"✗ Failed to import sagesim: {e}")
    print(f"Current Python path: {sys.path}")
