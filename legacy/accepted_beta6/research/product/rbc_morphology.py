"""Development-only labelled-instance measurements. Never binarize into cell IDs."""
import hashlib
import importlib.metadata as metadata
import math
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import regionprops
from service import Settings, decode, array_hash

VERSION = '0.1.0-dev.1-instance-input'
MAX_INSTANCES = 2000
NOTE = ('DEVELOPMENT: supplied instance identities are user/reference assertions, not independently validated cells. '
        'No automatic cell detector, segmentation repair, deformability inference or diagnostic interpretation is provided.')


def analyze_instances(labels, *, identity_confirmed=False, um_per_pixel=None, calibration_source='', sample_id='', donor_id='', source=None):
    labels = np.asarray(labels)
    if not identity_confirmed:
        raise ValueError('Explicitly confirm that each positive ID represents one individual-cell instance, not a binary foreground or semantic class.')
    if labels.ndim != 2 or labels.dtype.kind not in 'ui' or np.any(labels < 0):
        raise ValueError('Require a 2-D nonnegative integer instance mask: 0 background, one positive ID per instance. No RGB, floats or automatic relabelling.')
    if labels.size > 1_000_000: raise ValueError('Maximum 1 million pixels; no resizing.')
    ids = np.unique(labels); ids = ids[ids > 0]
    if not len(ids) or len(ids) > MAX_INSTANCES:
        raise ValueError('Require 1–2000 labelled instances. No labels were deleted.')
    if um_per_pixel is not None:
        if isinstance(um_per_pixel,bool) or not math.isfinite(um_per_pixel) or um_per_pixel <= 0 or not calibration_source.strip():
            raise ValueError('Positive finite isotropic calibration and its source are required.')
    # Dense internal index avoids huge allocations for sparse IDs. Original IDs
    # are exported unchanged and touching different IDs remain separate.
    dense = np.zeros(labels.shape, dtype=np.int32)
    positive = labels > 0
    dense[positive] = np.searchsorted(ids, labels[positive])+1
    rows = []
    properties = regionprops(dense)
    if sum((p.bbox[2]-p.bbox[0])*(p.bbox[3]-p.bbox[1]) for p in properties) > 4_000_000:
        raise ValueError('Instance bounding boxes exceed the hosted computation budget. Use the local workflow; no labels were removed.')
    for prop in properties:
        original_id = int(ids[prop.label-1])
        if ndimage.label(prop.image, structure=np.ones((3,3),dtype=np.uint8))[1] != 1:
            raise ValueError(f'ID {original_id} contains disconnected regions. Supply validated individual labels; no automatic split was performed.')
        a,p = float(prop.area),float(prop.perimeter)
        major,minor = float(prop.axis_major_length),float(prop.axis_minor_length)
        border = prop.bbox[0]==0 or prop.bbox[1]==0 or prop.bbox[2]==labels.shape[0] or prop.bbox[3]==labels.shape[1]
        isotropic = math.isclose(major,minor,rel_tol=1e-12,abs_tol=1e-12)
        row = dict(instance_id=original_id, sample_id=sample_id, donor_id=donor_id,
                   projected_area_px2=a, perimeter_px=p,
                   equivalent_circle_diameter_px=float(prop.equivalent_diameter_area),
                   major_axis_px=major, minor_axis_px=minor,
                   aspect_ratio=major/minor if minor>0 else None,
                   circularity=4*math.pi*a/(p*p) if p>0 else None,
                   orientation_rad=None if isotropic else float(prop.orientation),
                   solidity=float(prop.solidity), eccentricity=float(prop.eccentricity),
                   touches_border=border,
                   qc='; '.join(x for x,active in [('BORDER_TRUNCATION_POSSIBLE',border),('ZERO_PERIMETER',p==0),('ZERO_MINOR_AXIS',minor==0),('ORIENTATION_UNDEFINED_ISOTROPIC',isotropic),('HOLES_IN_INSTANCE',prop.euler_number<1)] if active) or 'No configured flag; not independent validation')
        if um_per_pixel is not None:
            row['projected_area_um2']=a*um_per_pixel**2
            for key in ('perimeter_px','equivalent_circle_diameter_px','major_axis_px','minor_axis_px'):
                row[key[:-3]+'_um']=row[key]*um_per_pixel
        rows.append(row)
    table = pd.DataFrame(rows)
    distribution=[]
    for key in ('projected_area_px2','perimeter_px','equivalent_circle_diameter_px','major_axis_px','minor_axis_px','aspect_ratio','circularity','solidity','eccentricity'):
        v=pd.to_numeric(table[key],errors='coerce').dropna()
        distribution.append(dict(metric=key,n_observed=int(len(v)),median=float(v.median()) if len(v) else None,
                                 q25=float(v.quantile(.25)) if len(v) else None,q75=float(v.quantile(.75)) if len(v) else None,
                                 iqr=float(v.quantile(.75)-v.quantile(.25)) if len(v) else None))
    provenance=dict(module_version=VERSION,mask_sha256=array_hash(labels),source=source or {},shape=list(labels.shape),dtype=str(labels.dtype),
                    instance_count=len(rows),donor_count='NOT INFERRED',sample_id=sample_id,donor_id=donor_id,
                    um_per_pixel=um_per_pixel,calibration_source=calibration_source,
                    transformations='None; temporary dense indexing preserves supplied IDs and region membership',
                    measurement_conventions='scikit-image regionprops; 4-neighbour perimeter estimator; ellipse axes from second moments; orientation relative to row axis, radians [-pi/2,pi/2]; circularity 4*pi*A/P^2, not clipped; equivalent-circle diameter 2*sqrt(A/pi).',
                    orientation_summary='Axial orientation distribution only; no ordinary arithmetic median across angle wrap.',
                    distribution_scope='One uploaded field; descriptive instance distributions, not independent biological replicates or a complete donor sample.',
                    missing='Undefined values remain null, never zero substitutes',note=NOTE,
                    dependencies={n:metadata.version(n) for n in ('numpy','scipy','scikit-image','pandas')},
                    implementation_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
    return dict(instances=rows,distributions=distribution,provenance=provenance)


def decode_instances(payload,name):
    labels,source=decode(payload,name,Settings(),mask=True)
    return labels,source


def synthetic_instances():
    labels=np.zeros((80,120),dtype=np.uint16)
    labels[10:30,10:30]=7
    labels[10:30,30:50]=91  # touching but distinct annotated IDs
    yy,xx=np.ogrid[:80,:120]
    labels[((yy-52)/10)**2+((xx-82)/19)**2<=1]=400
    return labels
