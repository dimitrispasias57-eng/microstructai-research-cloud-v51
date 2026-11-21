"""Import exact vendored engines without changing the accepted product."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'vendor'
for path in (VENDOR/'accepted_beta6/research/product',VENDOR/'accepted_beta6/research/engine',VENDOR/'RBC_DEFORMABILITY_DEV',VENDOR/'RBC_MORPHOLOGY_DEV'):
    if str(path) not in sys.path:sys.path.insert(0,str(path))

# Configure the unchanged Windows compatibility dispatch BEFORE NumPy imports.
import runtime_profile
runtime_profile.configure()
