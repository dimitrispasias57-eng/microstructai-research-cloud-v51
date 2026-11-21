"""Local candidate imports only. No deployment or changes to frozen sources."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PRODUCT = HERE.parent
HOSTED = (HERE / 'vendor').is_dir()
CODES = HERE / 'vendor' if HOSTED else PRODUCT.parent
if HOSTED:
    import hashlib, json
    for name,expected in json.loads((HERE/'VENDOR_SHA256.json').read_text()).items():
        if hashlib.sha256((CODES/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Frozen review adapter/morphology source changed: '+name)
for path in (PRODUCT / 'product', CODES / 'RBC_MORPHOLOGY_DEV'):
    sys.path.insert(0, str(path))
# Configure the existing process-local numerical profile before importing NumPy.
from runtime_profile import configure
configure()

VERSION = 'mask-review-0.2.0-dev.1'
PRODUCT_VERSION = '4.1.0-beta.6'
DELIVERY = 'HOSTED STAGING candidate' if HOSTED else 'LOCAL DEVELOPMENT candidate'
PATH_SOURCE = CODES / 'RBC_PATHOLOGICS_TARGET_SEGMENTER_V01/frozen/target_segment.py'
PATH_HASH = 'ed80d5702c80251453139e0902907960d852db9ef480b1f1226612afc0fdad36'
