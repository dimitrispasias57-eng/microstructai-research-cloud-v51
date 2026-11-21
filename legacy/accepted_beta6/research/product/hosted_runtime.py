"""Host-runtime policy/provenance. Never computes or changes scientific metrics."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import re
import sys

REPO=Path(__file__).resolve().parents[2]
MANIFEST=REPO/'HOSTED_RUNTIME_MANIFEST.json'
RUNTIME_ID='microstructai-research-4.1-linux-cp31312-v1'
COMPAT_MANIFEST=REPO/'runtime/COMMUNITY_CLOUD_COMPATIBILITY.json'
COMPAT_SHA256='a4d1676dd7bb4846aa11e1b81e84401fe8984f6b56ec7eb09e1f2bbd54bddd9c'
SUPPLEMENT_LOCK=REPO/'runtime/requirements-community-cloud.lock'
# UI/product-only releases; immutable beta.3 numerical runtime/reference
# remains authoritative. Beta.5 adds reviewed-mask orchestration only;
# beta.6 adds experimental/trust wording and performance guidance only.
RESET_ONLY_RELEASES={'4.1.0-beta.4':'4.1.0-beta.3','4.1.0-beta.5':'4.1.0-beta.3','4.1.0-beta.6':'4.1.0-beta.3'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def compatibility_profile():
    """A separately recorded deployment profile, never replacement references."""
    if not COMPAT_MANIFEST.exists() or sha(COMPAT_MANIFEST)!=COMPAT_SHA256:
        raise ValueError('Community Cloud compatibility manifest is missing or changed.')
    profile=json.loads(COMPAT_MANIFEST.read_text())
    for path,key in ((MANIFEST,'base_manifest_sha256'),
                     (REPO/'requirements-linux.lock','base_lock_sha256'),
                     (REPO/'reference/LINUX_HOSTED_REFERENCE_RESULTS.json','reference_sha256'),
                     (SUPPLEMENT_LOCK,'supplement_lock_sha256')):
        if not path.exists() or sha(path)!=profile[key]:
            raise ValueError('Community Cloud compatibility integrity mismatch: '+path.name)
    return profile

def provenance():
    actual_python=platform.python_version()
    base={'runtime_id':RUNTIME_ID if actual_python=='3.13.12' else 'unapproved',
          'reference_runtime_id':RUNTIME_ID,'actual_python':actual_python,'actual_os':platform.platform(),
          'actual_machine':platform.machine(),'canonical_os':'Linux','release_manifest_available':MANIFEST.exists()}
    if MANIFEST.exists():
        manifest=json.loads(MANIFEST.read_text())
        base.update(dependency_lock_sha256=sha(REPO/'requirements-linux.lock'),
                    approved_lock_sha256=manifest['dependency_lock_sha256'],runtime_manifest_sha256=sha(MANIFEST),
                    release_date=manifest['release_date'])
        if actual_python=='3.13.15':
            try:
                profile=compatibility_profile()
                base.update(runtime_id=profile['runtime_id'],compatibility_manifest_sha256=sha(COMPAT_MANIFEST),
                            supplement_lock_sha256=sha(SUPPLEMENT_LOCK),
                            additional_packages=profile['additional_packages'],
                            compatibility_date=profile['date'])
            except (OSError,ValueError,KeyError) as exc:
                base['compatibility_error']=str(exc)
    return base

def verify():
    errors=[]
    if not MANIFEST.exists():return {'pass':False,'errors':['Approved runtime manifest is missing.']}
    manifest=json.loads(MANIFEST.read_text())
    packages=dict(manifest['packages'])
    expected_python=manifest['python']
    # Explicit support for the observed Community Cloud patch. Unknown future
    # patches remain blocked. All original references and 57 package pins stay.
    if platform.python_version()=='3.13.15':
        try:
            profile=compatibility_profile()
            if set(packages)&set(profile['additional_packages']):
                raise ValueError('Compatibility profile must not replace a frozen dependency.')
            packages.update(profile['additional_packages'])
            expected_python=profile['python']
        except (OSError,ValueError,KeyError) as exc:
            errors.append(str(exc))
    if sys.platform!='linux':errors.append('Canonical hosted runtime requires Linux.')
    if platform.python_version()!=expected_python:errors.append('Python patch version does not match an explicitly pinned runtime profile.')
    if platform.machine()!=manifest['machine']:errors.append('CPU architecture does not match the approved runtime.')
    if sha(REPO/'requirements-linux.lock')!=manifest['dependency_lock_sha256']:errors.append('Dependency lock hash mismatch.')
    if sha(REPO/'runtime/WHEEL_MANIFEST.json')!=manifest['wheel_manifest_sha256']:errors.append('Wheel manifest hash mismatch.')
    for name,version in packages.items():
        try: actual=md.version(name)
        except md.PackageNotFoundError: actual='missing'
        if actual!=version:errors.append(f'{name}: expected {version}; found {actual}.')
    installed={re.sub('[-_.]+','-',d.metadata['Name']).lower() for d in md.distributions() if d.metadata['Name']}
    extras=installed-set(packages)
    if extras:errors.append('Unapproved additional distributions: '+', '.join(sorted(extras)))
    bootstrap=sys.modules.get('_microstructai_hosted_startup')
    if not bootstrap or not bootstrap.BEFORE_NUMPY or bootstrap.VERSION!='1.0.0':errors.append('Startup hook did not precede NumPy import.')
    from runtime_profile import describe
    if not describe()['profile_active']:errors.append('Numerical startup profile is inactive.')
    if any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')):errors.append('Thread controls do not match the approved runtime.')
    from service import VERSION, CORE_VERSION
    if RESET_ONLY_RELEASES.get(VERSION,VERSION)!=manifest['app_version'] or CORE_VERSION!=manifest['structural_core_version']:errors.append('App/core version does not match the hosted release manifest.')
    for name in ('opencv-python','opencv-python-headless'):
        try:md.version(name);errors.append(f'Unexpected {name}; not in the approved runtime.')
        except md.PackageNotFoundError:pass
    identity=provenance()
    if extras:
        identity['unexpected_distributions']={n:md.version(n) for n in sorted(extras)}
    return {'pass':not errors,'errors':errors,'identity':identity}
