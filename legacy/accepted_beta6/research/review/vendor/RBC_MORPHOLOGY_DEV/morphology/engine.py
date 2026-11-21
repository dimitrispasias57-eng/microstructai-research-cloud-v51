"""Measurements of supplied 2-D instances under protocol v1. No segmentation model."""
from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
import importlib.metadata
import math
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage.measure import find_contours
from . import VERSION

METRICS = ('area_px2','perimeter_px','equivalent_diameter_px','major_axis_px','minor_axis_px',
           'aspect_ratio','circularity','eccentricity','solidity','orientation_rad','convex_area_px2','extent')
LENGTHS = ('perimeter_px','equivalent_diameter_px','major_axis_px','minor_axis_px')
AREAS = ('area_px2','convex_area_px2')
NOTE = 'DEVELOPMENT: supplied-instance geometry, not independent cell identification, diagnosis, aggregation or deformability validation.'

@dataclass(frozen=True)
class Config:
    identity_confirmed: bool = False
    mode: str = 'instances'
    roi: tuple | None = None
    exclude_border: bool = False
    minimum_area: int = 1
    um_per_pixel: float | None = None
    calibration_source: str = ''
    image_id: str = ''
    sample_id: str = ''
    donor_id: str = ''

def validate_labels(labels):
    a = np.asarray(labels)
    if a.ndim != 2 or a.dtype.kind not in 'uib' or np.any(a < 0):
        raise ValueError('Require a nonnegative 2-D integer mask, not floats, RGB or a volume.')
    if not a.size or a.size > 1_000_000:
        raise ValueError('Require 1 to 1,000,000 pixels; no resizing.')
    if int(a.max()) > 2**63-1:
        raise ValueError('Instance IDs must fit signed 64-bit integers; no remapping on import.')
    ids = np.unique(a); ids = ids[ids>0]
    if not 1 <= len(ids) <= 2000:
        raise ValueError('Require 1 to 2000 positive IDs; none were silently removed.')
    return a

def stack_to_labels(stack):
    a = np.asarray(stack)
    if a.ndim != 3 or a.dtype.kind not in 'uib' or a.shape[0] < 1 or a.shape[0] > 2000 or a.size > 8_000_000:
        raise ValueError('Declared cell stack must be N x H x W integer/bool, <=2000 planes and <=8 million pixels.')
    out = np.zeros(a.shape[1:],dtype=np.int32)
    for i, plane in enumerate(a):
        values = np.unique(plane)
        if len(values)>2 or np.any(values<0) or (len(values)==2 and values[0]!=0):
            raise ValueError('Every stack plane must be one binary cell mask.')
        positive = plane>0
        if not positive.any(): raise ValueError(f'Empty cell mask in plane {i+1}; not silently skipped.')
        if np.any(positive & (out>0)): raise ValueError('Overlapping cell masks: pixel ownership is ambiguous; not resolved automatically.')
        out[positive]=i+1
    return validate_labels(out)

def measure(mask):
    """Single nonempty region; product path, independent of reference.py."""
    y,x = np.nonzero(mask); points=np.column_stack((x,y)).astype(float)
    area=float(len(points)); centered=points-points.mean(axis=0)
    eigenvalues,eigenvectors=np.linalg.eigh(centered.T@centered/area)
    small,large=np.maximum(eigenvalues,0)
    major,minor=4*math.sqrt(large),4*math.sqrt(small)
    angle=None
    if large-small > 1e-12*max(1.,large):
        v=eigenvectors[:,1];angle=float((math.atan2(v[1],v[0])+math.pi/2)%math.pi-math.pi/2)
    contours=find_contours(np.pad(mask.astype(float),1),.5,fully_connected='high')
    perimeter=float(sum(np.linalg.norm(np.diff(c,axis=0),axis=1).sum() for c in contours))
    corners=np.unique((points[:,None,:]+np.array([[-.5,-.5],[-.5,.5],[.5,-.5],[.5,.5]])).reshape(-1,2),axis=0)
    hull_area=float(ConvexHull(corners).volume)
    bbox_area=float((x.max()-x.min()+1)*(y.max()-y.min()+1))
    return dict(area_px2=area,perimeter_px=perimeter,equivalent_diameter_px=2*math.sqrt(area/math.pi),
                major_axis_px=major,minor_axis_px=minor,aspect_ratio=major/minor if minor>0 else None,
                circularity=4*math.pi*area/perimeter**2 if perimeter>0 else None,
                eccentricity=math.sqrt(max(0.,1-small/large)) if large>0 else None,
                solidity=area/hull_area,orientation_rad=angle,convex_area_px2=hull_area,extent=area/bbox_area)

def summarize(rows):
    included=[r for r in rows if r['included']]
    keys=list(METRICS)+[k for k in (included[0] if included else {}) if k.endswith(('_um','_um2'))]
    summary=[]
    for key in keys:
        values=[r[key] for r in included if r.get(key) is not None and math.isfinite(r[key])]
        item=dict(metric=key,n_included=len(included),n_finite=len(values),n_missing=len(included)-len(values))
        if key=='orientation_rad':
            if values:
                z=np.exp(2j*np.asarray(values)).mean();strength=float(abs(z))
                mean=float((np.angle(z)/2+math.pi/2)%math.pi-math.pi/2) if strength>1e-12 else None
            else: mean=None;strength=None
            item.update(axial_mean_rad=mean,axial_resultant=strength)
        else:
            q=np.quantile(values,[.25,.5,.75],method='linear') if values else [None]*3
            item.update(q25=q[0],median=q[1],q75=q[2],iqr=q[2]-q[0] if values else None,
                        minimum=min(values) if values else None,maximum=max(values) if values else None)
        summary.append(item)
    return summary

def analyze(labels, config=Config(), source=None):
    a=validate_labels(labels)
    if config.mode not in ('instances','components'): raise ValueError('Unknown input mode.')
    if config.mode=='instances' and not config.identity_confirmed:
        raise ValueError('Confirm that each positive ID denotes one supplied instance; binary components are not automatically cells.')
    if config.mode=='components':
        if len(np.unique(a[a>0]))!=1: raise ValueError('Connected-component mode expects binary foreground, not labelled IDs.')
        a,count=ndimage.label(a>0,structure=np.ones((3,3)))
        if count>2000: raise ValueError('More than 2000 connected components; no automatic filtering.')
    if isinstance(config.minimum_area,bool) or not isinstance(config.minimum_area,int) or config.minimum_area<1:
        raise ValueError('Minimum area must be a positive integer.')
    scale=config.um_per_pixel
    if scale is not None and (isinstance(scale,bool) or not np.isscalar(scale) or not math.isfinite(scale) or scale<=0 or not config.calibration_source.strip()):
        raise ValueError('Require positive finite isotropic micrometres/pixel and calibration source. Anisotropic scale is unsupported.')
    h,w=a.shape;roi=config.roi or (0,0,w,h)
    if len(roi)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) for v in roi):
        raise ValueError('ROI must be four integer coordinates.')
    x0,y0,x1,y1=roi
    if not (0<=x0<x1<=w and 0<=y0<y1<=h): raise ValueError('ROI is outside image or empty.')
    view=a[y0:y1,x0:x1]
    ids,counts=np.unique(a,return_counts=True);original_counts={int(i):int(n) for i,n in zip(ids,counts) if i>0}
    original_ids=np.array(list(original_counts))
    full_dense=np.zeros(a.shape,np.int32);full_positive=a>0
    full_dense[full_positive]=np.searchsorted(original_ids,a[full_positive])+1
    full_boxes=ndimage.find_objects(full_dense)
    if sum((b[0].stop-b[0].start)*(b[1].stop-b[1].start) for b in full_boxes if b)>4_000_000:
        raise ValueError('Full-image bounding-box budget exceeded; no labels removed.')
    original_connected={int(i):ndimage.label(a[b]==i,structure=np.ones((3,3)))[1]==1 for i,b in zip(original_ids,full_boxes)}
    positive=view>0;local_ids=np.unique(view[positive]);dense=np.zeros(view.shape,np.int32)
    dense[positive]=np.searchsorted(local_ids,view[positive])+1
    boxes=ndimage.find_objects(dense);lookup={int(i):box for i,box in zip(local_ids,boxes)}
    if sum((b[0].stop-b[0].start)*(b[1].stop-b[1].start) for b in boxes if b)>4_000_000:
        raise ValueError('Bounding-box budget exceeded; no labels removed.')
    rows=[]
    for instance_id,total in original_counts.items():
        row=dict(instance_id=instance_id,image_id=config.image_id,sample_id=config.sample_id,donor_id=config.donor_id,
                 object_semantics='supplied instance (identity asserted)' if config.mode=='instances' else 'connected component; physical-cell identity unestablished',
                 original_area_px2=total,included=False,qc=[],exclusion_reasons=[])
        row.update({k:None for k in METRICS})
        box=lookup.get(instance_id)
        if box is None:
            row.update(touches_border=False,roi_clipped=True,geometrically_valid=False)
            row['exclusion_reasons'].append('OUTSIDE_ROI');rows.append(row);continue
        mask=view[box]==instance_id
        border=box[0].start==0 or box[1].start==0 or box[0].stop==view.shape[0] or box[1].stop==view.shape[1]
        clipped=int(mask.sum())<total
        connected=original_connected[instance_id] and ndimage.label(mask,structure=np.ones((3,3)))[1]==1
        row.update(touches_border=border,roi_clipped=clipped,geometrically_valid=connected,
                   centroid_x_px=float(np.nonzero(mask)[1].mean()+box[1].start+x0),
                   centroid_y_px=float(np.nonzero(mask)[0].mean()+box[0].start+y0))
        if border:row['qc'].append('BORDER_TRUNCATION_POSSIBLE')
        if clipped:row['qc'].append('ROI_CLIPPED')
        if not connected:
            row['qc'].append('DISCONNECTED_ID');row['exclusion_reasons'].append('INVALID_INSTANCE')
        else:
            row.update(measure(mask))
            if row['area_px2']<5:row['qc'].append('LOW_PIXEL_SUPPORT')
            if row['minor_axis_px']==0:row['qc'].append('ZERO_MINOR_AXIS')
            if row['orientation_rad'] is None:row['qc'].append('ORIENTATION_UNDEFINED_ISOTROPIC')
            if row['circularity']>1:row['qc'].append('CIRCULARITY_ABOVE_ONE_RASTER_ESTIMATOR')
            if np.any(ndimage.binary_fill_holes(mask)&~mask):row['qc'].append('HOLES_PRESENT_NOT_FILLED')
            if row['area_px2']<config.minimum_area:row['exclusion_reasons'].append('BELOW_EXPLICIT_MINIMUM_AREA')
        if border and config.exclude_border:row['exclusion_reasons'].append('EXPLICIT_BORDER_EXCLUSION')
        row['included']=not row['exclusion_reasons']
        if scale is not None:
            for key in LENGTHS:row[key[:-3]+'_um']=row[key]*scale if row[key] is not None else None
            for key in AREAS:row[key[:-4]+'_um2']=row[key]*scale**2 if row[key] is not None else None
        rows.append(row)
    n=sum(r['included'] for r in rows);roi_area=(x1-x0)*(y1-y0)
    provenance=dict(version=VERSION,definition_version='v1',settings=asdict(config),roi=list(roi),source=source or {},
        decoded_sha256=sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest(),
        engine_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        input_count=len(rows),included_count=n,excluded_count=len(rows)-n,
        valid_geometry_count=sum(r['geometrically_valid'] for r in rows),biological_replicates='NOT INFERRED',
        transformations='Explicit ROI clipping only; no resizing, smoothing, filling or automatic cell splitting. Components mode explicitly uses 8-connectivity.',
        dependencies={k:importlib.metadata.version(k) for k in ('numpy','scipy','scikit-image')},note=NOTE)
    return dict(instances=rows,summaries=summarize(rows),provenance=provenance,
                phenotype=dict(morphology='Per-instance projected size, elongation, compactness and axial orientation',
                    population='Included-instance descriptive distributions, not independent donor replicates',
                    spatial=dict(included_instances_per_px2=n/roi_area,included_instances_per_um2=n/(roi_area*scale**2) if scale else None),
                    aggregation='NOT CALCULATED: sparse supplied-cell masks do not establish gap/aggregation semantics',
                    deformability='DESIGN ONLY; static shape is not mechanical response',coagulation='DESIGN ONLY; not aggregation'))

def combine_samples(results, *, nonoverlapping_fields_confirmed=False):
    if not nonoverlapping_fields_confirmed:raise ValueError('Explicit nonoverlapping fields confirmation required; repeated cells must not be double-counted.')
    groups={};seen=set();settings_by_sample={}
    for result in results:
        p=result['provenance'];s=p['settings'];image=s['image_id'];sample=s['sample_id'];donor=s['donor_id']
        if not image or not sample:raise ValueError('Explicit image and sample IDs required for pooling.')
        token=(sample,image)
        if token in seen:raise ValueError('Duplicate image in sample.')
        seen.add(token)
        if not p.get('decoded_sha256') or not p.get('engine_sha256'):
            raise ValueError('Missing measurement source provenance; do not pool.')
        contract=(donor,s['um_per_pixel'],s['calibration_source'],s['mode'],s['exclude_border'],s['minimum_area'],
                  p['version'],p['definition_version'],p['engine_sha256'])
        if sample in settings_by_sample and settings_by_sample[sample]!=contract:
            raise ValueError('Sample has conflicting donor/calibration/QC/semantics; do not pool.')
        settings_by_sample[sample]=contract
        group=groups.setdefault(sample,dict(sample_id=sample,donor_id=donor,image_ids=[],instances=[]))
        group['image_ids'].append(image);group['instances'].extend(result['instances'])
    for group in groups.values():
        group['summaries']=summarize(group['instances'])
        group['biological_replicates']='A sample-level descriptive pool, not an increase in donor replication'
    return list(groups.values())
