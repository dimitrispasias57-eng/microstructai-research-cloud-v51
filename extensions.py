"""v5.1 transport adapters. All numerical work calls unchanged historical functions."""
import contextlib, hashlib, json, sys, threading
import backend as b
import numpy as np
import pandas as pd
from modules import morphology_intelligence as mi, structural_intelligence as si, v4_diagnostics as diag
from modules import metrics as frozen_metrics
from legacy.RBC_MORPHOLOGY_DEV.morphology import engine as morphology
from full_product.tracking import Calibration, measure_tracks
from full_product.tracking_import_v2 import import_tracks

# A shared capacity lock contains no user data; all results belong to the caller/session.
CAPACITY=b.ANALYSIS_SLOT

@contextlib.contextmanager
def capacity():
    if not CAPACITY.acquire(blocking=False):raise ValueError('Another heavy calculation is running. Retry shortly; no inputs were discarded.')
    try:yield
    finally:CAPACITY.release()

def verify_helpers():
    entries=json.loads((b.ROOT/'HELPER_MANIFEST.json').read_text())
    for row in entries:
        if hashlib.sha256((b.ROOT/row['path']).read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Historical helper integrity failure: '+row['path'])
    return len(entries)

def review_engine():
    """Import exact historical transaction engine and its sealed proposal adapters."""
    verify_helpers()
    for folder in ('legacy/RBC_MORPHOLOGY_DEV','legacy/accepted_beta6/research/product','legacy/accepted_beta6/research/review'):
        p=str(b.ROOT/folder)
        if p not in sys.path:sys.path.insert(0,p)
    # Native imports the same frozen morphology engine first, but its minimal
    # package omits I/O. Extend only the import search path after identity check.
    import morphology as native_morphology
    import morphology.engine as native_engine
    if b.native.sha_bytes(b.ROOT.joinpath('legacy/RBC_MORPHOLOGY_DEV/morphology/engine.py').read_bytes())!=b.native.sha_bytes(__import__('pathlib').Path(native_engine.__file__).read_bytes()):
        raise RuntimeError('Cannot share non-identical morphology engines.')
    morphology_path=str(b.ROOT/'legacy/RBC_MORPHOLOGY_DEV/morphology')
    if morphology_path not in native_morphology.__path__:native_morphology.__path__.append(morphology_path)
    import review_core
    return review_core

def generic_scores(records,group_id,comparable):
    """Same RC2 complete-case eligibility and score rules, never RBC score aliases."""
    review_engine()
    from historical_ui import general_metrics as gm
    rows=[]
    for r in records:
        valid=r.get('review')=='HUMAN_REVIEWED'
        row={'filename':r['name'],'valid':valid}
        if valid:
            values=frozen_metrics.compute_object_mode_metrics(r['mask'].astype(float),r['mask'])
            row.update({k:values[k] for k in gm.FEATURES})
        rows.append(row)
    return gm.score_group(pd.DataFrame(rows),enabled=True,comparable=comparable,group_id=group_id)

def drawn_roi(value,shape,identity):
    """UI coordinate adapter: outward rounding, source pixels, zero-based crop contract."""
    import math
    if not isinstance(value,dict) or value.get('input_id')!=identity:raise ValueError('Rectangle belongs to a different source.')
    bounds=value.get('bounds')
    if not isinstance(bounds,list) or len(bounds)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in bounds):raise ValueError('Rectangle needs four finite coordinates.')
    x0,x1=sorted(bounds[::2]);y0,y1=sorted(bounds[1::2])
    if not(0<=x0<x1<=shape[1] and 0<=y0<y1<=shape[0]):raise ValueError('Draw a rectangle within the source image; no clipping.')
    roi=[math.floor(x0),math.floor(y0),math.ceil(x1)-math.floor(x0),math.ceil(y1)-math.floor(y0)]
    if min(roi[2:])<2:raise ValueError('ROI needs at least two source pixels in each dimension.')
    return roi

def area_slopes(records):
    rows=[]
    for r in records:
        slope,r2,bins=frozen_metrics.compute_gap_area_slope_matlab_like(r['mask'])
        rows.append(dict(filename=r['name'],area_distribution_slope=slope,fit_R2=r2,occupied_bins=bins))
    return pd.DataFrame(rows)

def archive_groups(payload):
    """Stricter transport budget before the unchanged safe archive reader."""
    import io,zipfile
    from modules.web_uploads import read_sample_archive
    if len(payload)>32*1024**2:raise ValueError('Cloud archive limit: 32 MiB.')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries=[v for v in archive.infolist() if not v.is_dir()]
        if len(entries)>200 or sum(v.file_size for v in entries)>32*1024**2 or any(v.file_size>8*1024**2 for v in entries):raise ValueError('Archive limit: 200 files, 32 MiB expanded, 8 MiB per file. No frames were removed.')
    groups=read_sample_archive(payload)
    pixels=0
    for name,files in groups.items():
        decoded=b.decode_batch([(f.name,f.getvalue()) for f in files]);pixels+=sum(v['image'].size for v in decoded)
        if pixels>16_000_000:raise ValueError('Archive exceeds 16 million channel-pixels total. No resizing or frame dropping.')
    return groups

def supplied_instances(image,*,confirmed=False,components=False,scale=None,scale_source='',image_id='',sample_id='',exclude_border=False,roi=None):
    cfg=morphology.Config(identity_confirmed=confirmed,mode='components' if components else 'instances',
        um_per_pixel=scale,calibration_source=scale_source,image_id=image_id,sample_id=sample_id,
        exclude_border=exclude_border,roi=roi)
    with capacity():return morphology.analyze(image,cfg)

def check_heavy(records,max_frames=12,max_pixels=1_000_000):
    if not records or len(records)>max_frames or sum(np.asarray(r['mask']).size for r in records)>max_pixels:
        raise ValueError(f'Advanced structural analysis limit: {max_frames} observations and {max_pixels:,} total pixels. No sampling or resizing was applied. Larger runs remain local.')
    from scipy.ndimage import label
    if any(label(r['mask'],np.ones((3,3)))[1]>500 for r in records):
        raise ValueError('Advanced structure requires at most 500 connected regions per observation for cloud resource safety. No regions were removed.')

def structural_sequence(records,table,params):
    check_heavy(records)
    if len(records)!=len(table):raise ValueError('Every observation must have its matching metric row; no silent truncation.')
    previews=[dict(filename=r['name'],mask=r['mask'],image=b.plane(r.get('original_roi',r.get('image')),'Grayscale / luminance')) for r in records]
    with capacity():return si.analyze_structural_sequence(previews,table,dict(params,enableStructuralIntelligence=True))

def static_structure(record):
    check_heavy([record],max_frames=1,max_pixels=250_000)
    with capacity():return mi.analyze_static_gap_structure(record['mask'])

def robustness(record,include_dark=False):
    check_heavy([record],max_frames=1,max_pixels=250_000)
    image=record.get('original_roi',record.get('image'))
    with capacity():
        table,summary=si.gap_boundary_robustness(record['mask'])
        quality=diag.image_diagnostics(image,record.get('metadata'))+diag.mask_diagnostics(record['mask'])
        result={'boundary_perturbations':table,'summary':summary,'diagnostics':pd.DataFrame(quality)}
        if include_dark:
            if image is None:raise ValueError('An original image is required for dark-domain threshold sensitivity.')
            result['dark_domain']=si.analyze_cellrich_representation(b.plane(image,'Grayscale / luminance'),si.structural_settings_from_params({}))
        return result

def supplied_tracks(payload,um_per_pixel,seconds_per_frame,spatial_evidence,temporal_evidence,sequence_id):
    verify_helpers()
    imported=import_tracks(payload,sequence_id=sequence_id)
    calibration=Calibration(um_per_pixel=um_per_pixel,seconds_per_frame=seconds_per_frame,
        spatial_source=spatial_evidence,temporal_source=temporal_evidence)
    with capacity():return measure_tracks(imported,calibration)

def recording_batch(groups,labels,reference_payloads=None,fps=1.):
    """Upload-only wrapper; never permits browser visitors to select server paths."""
    if not 1<=len(groups)<=4:raise ValueError('Use 1–4 independent recordings per cloud batch.')
    all_payloads=[(group+'/'+f.name,f.getvalue()) for group,files in groups.items() for f in files]
    b.decode_batch(all_payloads)  # Enforce whole-batch limits, no arbitrary splitting.
    m=b.rbc_engine()
    if not m.WEB_DEPLOYMENT:raise ValueError('Batch transport requires the frozen web deployment mode; desktop file access is disabled here.')
    if set(labels.sampleName)!=set(groups) or len(labels)!=len(groups) or labels.sampleName.duplicated().any():raise ValueError('Exactly one explicit row for every uploaded recording is required.')
    if any(r['imageFolder']!=r['sampleName'] for r in labels.to_dict('records')):raise ValueError('Recording keys, not filesystem paths, are required.')
    if isinstance(fps,bool) or not np.isfinite(fps) or fps<=0:raise ValueError('Documented positive uniform frame rate required.')
    params=dict(b.rbc_defaults(),matlab_fps=float(fps),batch_use_matlab_csv=bool(reference_payloads),batch_uploaded_matlab_csv_bytes=reference_payloads or {},batch_validation_mode='RBC Gap Mode',batch_input_mode='Uploaded independent recording folders')
    with capacity(),b.ENGINE_LOCK:return m.run_batch_validation(labels,params,image_uploads=groups)

def legacy_object_recording(frames,params):
    """Transport only: exact historical Object Mode dispatcher and user settings."""
    import copy
    if not frames or len({r['image'].shape for r in frames})!=1:raise ValueError('One complete recording requires consistent dimensions/channels.')
    b.decode_batch([(r['name'],r['payload']) for r in frames])
    engine=b.rbc_engine();settings=copy.deepcopy(params)
    if settings.get('selectedObjectMethod') not in engine.OBJECT_ANALYSIS_METHODS:raise ValueError('Select an existing Object Mode method explicitly.')
    with capacity(),b.ENGINE_LOCK:
        result=engine.analyze_dataset([b.Upload(r['name'],r['payload']) for r in frames],[],'Object Mode',settings)
    if len(result['metrics'])!=len(frames) or len(result['previews'])!=len(frames):raise ValueError('Historical Object Mode did not return complete coverage. No partial batch was accepted.')
    return result

def tables_in(value,prefix=''):
    """Export all existing tabular outputs without recalculation or schema flattening."""
    out={}
    if isinstance(value,pd.DataFrame):out[prefix or 'results']=value
    elif isinstance(value,dict):
        for name,child in value.items():
            if isinstance(child,(dict,pd.DataFrame)):out.update(tables_in(child,(prefix+'_' if prefix else '')+name))
    return out
