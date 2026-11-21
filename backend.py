"""New transport/orchestration layer; quantitative functions remain vendored unchanged."""
from pathlib import Path
from functools import lru_cache, wraps
import sys, hashlib, io, json, importlib.util, threading, copy
from legacy.accepted_beta6.research.product import runtime_profile
runtime_profile.configure()
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parent
ENGINE=ROOT/'legacy/accepted_beta6/research/engine'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'native'))
sys.path.insert(0,str(ENGINE))
import runtime as native
import exports as native_exports
from metrics import metrics as reference_metrics
from modules import structural_core, segmentation
from modules.pca_analysis import run_pca, finite_numeric_columns
from modules.metrics import compute_object_mode_metrics
from modules.morphology_intelligence import local_thickness_analysis
from modules.v4_ingestion import load_frames, display_u8
from modules.io_utils import natural_sort_key
from legacy.RBC_DEFORMABILITY_DEV.core import dynamics
from legacy.accepted_beta6.research.product import rbc_temporal
VERSION='5.1-rc1'
ENGINE_LOCK=threading.Lock()
ANALYSIS_SLOT=threading.BoundedSemaphore(1)

def bounded_analysis(fn):
    @wraps(fn)
    def run(*args,**kwargs):
        if not ANALYSIS_SLOT.acquire(blocking=False):raise ValueError('Server is processing another heavy analysis. Retry shortly; no inputs were discarded.')
        try:return fn(*args,**kwargs)
        finally:ANALYSIS_SLOT.release()
    return run

def verify_vendors():
    if not runtime_profile.describe()['profile_active']:
        raise RuntimeError('Historical numerical CPU profile is not active. Restart the application process; no guard was bypassed.')
    import importlib.metadata
    required={'numpy':'2.3.5','scipy':'1.18.0','scikit-image':'0.25.2','scikit-learn':'1.7.2','pandas':'2.3.3','pillow':'12.0.0'}
    for name,version in required.items():
        if importlib.metadata.version(name)!=version:raise RuntimeError('Numerical runtime version mismatch: '+name)
    if importlib.metadata.version('torch').split('+')[0]!='2.9.1':raise RuntimeError('Frozen native PyTorch version mismatch.')
    entries=json.loads((ROOT/'VENDOR_MANIFEST.json').read_text())
    for rel,digest in entries.items():
        if native.sha_bytes((ROOT/rel).read_bytes())!=digest:
            raise RuntimeError('Frozen source integrity failure: '+rel)
    native.verify_science()
    return len(entries)

class Upload(io.BytesIO):
    def __init__(self,name,payload):
        super().__init__(payload);self.name=name;self.size=len(payload)

def decode_batch(payloads):
    if not 1<=len(payloads)<=200:raise ValueError('Use 1–200 images per run. Larger recordings are local/advanced only.')
    if sum(len(b) for _,b in payloads)>32*1024**2:raise ValueError('Total upload exceeds 32 MiB. No files were removed.')
    if len({n.casefold() for n,_ in payloads})!=len(payloads):raise ValueError('Duplicate filenames prevent unambiguous provenance.')
    frames=[];pixels=0
    for name,payload in sorted(payloads,key=lambda item:natural_sort_key(item[0])):
        if len(payload)>8*1024**2:raise ValueError('Maximum individual file size: 8 MiB.')
        try:
            if payload[:4] in (b'II*\x00',b'MM\x00*',b'II+\x00',b'MM\x00+'):
                import tifffile
                with tifffile.TiffFile(io.BytesIO(payload)) as tf:
                    if len(tf.pages)!=1 or np.prod(tf.pages[0].shape)>3_000_000:
                        raise ValueError('Single bounded 2-D TIFF plane required; no stack flattening.')
            else:
                with Image.open(io.BytesIO(payload)) as im:
                    if im.width*im.height>1_000_000:raise ValueError('Maximum 1 million spatial pixels per image.')
            frame=load_frames(Upload(name,payload))[0]
        except OSError as exc:raise ValueError('Cannot decode '+name+': '+str(exc)) from exc
        a=frame.pixels
        if a.ndim==3 and a.shape[-1]==4:raise ValueError('Export an explicit RGB or grayscale plane; alpha interpretation is not assumed.')
        if a.shape[0]*a.shape[1]>1_000_000:raise ValueError('Maximum 1 million spatial pixels per image; no resizing is performed.')
        pixels+=a.size
        if pixels>16_000_000:raise ValueError('Run exceeds 16 million channel-pixels. Complete larger recordings are local/advanced only; do not split temporal analysis arbitrarily.')
        frames.append({'name':name,'payload':payload,'image':a,'metadata':frame.metadata,'sha256':native.sha_bytes(payload)})
    return frames

def binary(a):
    a=np.asarray(a)
    if a.ndim==3:
        if not (np.array_equal(a[...,0],a[...,1]) and np.array_equal(a[...,0],a[...,2])):
            raise ValueError('A colour image is not a supplied binary mask.')
        a=a[...,0]
    values=set(np.unique(a).tolist())
    if not (values<={0,1} or values<={0,255}):
        raise ValueError('Supply a lossless binary 0/1 or 0/255 mask, or choose Original image to generate a proposal.')
    return a!=0

def plane(a,channel):
    if a.ndim==2:
        if channel!='Grayscale / luminance':raise ValueError('Selected colour channel is absent.')
        return a.copy()
    if channel in ('Red','Green','Blue'):return a[...,['Red','Green','Blue'].index(channel)].copy()
    return segmentation.to_grayscale_float01(a)

def crop(a,roi):
    if roi is None:return a.copy()
    x,y,w,h=roi
    if any(isinstance(v,bool) or int(v)!=v for v in roi) or min(x,y)<0 or min(w,h)<2 or x+w>a.shape[1] or y+h>a.shape[0]:
        raise ValueError('ROI must be an in-bounds rectangle: zero-based x, y, width, height. No clipping is applied.')
    return a[y:y+h,x:x+w].copy()

@bounded_analysis
def propose_general(frames,config):
    params={'background_correction':config.get('illumination',False),'background_sigma':config.get('background_scale',25.),
            'threshold_method':config.get('threshold','otsu'),'min_object_size':config.get('min_area',40),
            'fill_holes':config.get('fill_holes',False)}
    fn=segmentation.segment_dark_objects if config.get('polarity')=='dark' else segmentation.segment_bright_objects
    records=[]
    for f in frames:
        a=crop(f['image'],config.get('roi')); selected=plane(a,config['channel'])
        mask=binary(a) if config.get('supplied') else fn(selected,params)
        corrected=selected if config.get('supplied') else segmentation._preprocess(segmentation.to_grayscale(selected),params)
        records.append({**f,'original_roi':a,'selected':selected,'corrected':corrected,'mask':mask,'review':'NOT_REVIEWED'})
    return records

@bounded_analysis
def measure_general(records):
    rows=[];objects=[]
    for r in records:
        values,table=structural_core.measure_mask(r['mask'])
        # Additional existing generic spectral, intensity and thickness descriptors.
        extra=compute_object_mode_metrics(r['selected'],r['mask'])
        thickness,_,_=local_thickness_analysis(r['mask'])
        values.update({k:v for k,v in extra.items() if k not in values and k not in ('filename','mode')})
        values.update(thickness)
        rows.append({'filename':r['name'],'source_sha256':r['sha256'],'mask_sha256':native.sha_bytes(r['mask'].tobytes()),'review':r.get('review','NOT_REVIEWED'),**values})
        table=table.copy();table.insert(0,'filename',r['name']);objects.append(table)
    return pd.DataFrame(rows),pd.concat(objects,ignore_index=True)

@lru_cache(maxsize=1)
def rbc_engine():
    spec=importlib.util.spec_from_file_location('unified_frozen_rbc',ENGINE/'app.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def rbc_defaults():
    m=rbc_engine()
    return dict(m.DEFAULT_CAMERA_PRESETS['USB camera RBC default'],rbcAnalysisMethod='Standard RBC Gap Mode',
                rbcSegmentationInputSource='MATLAB Iraw01 processed grayscale',matlab_fps=1.,removeModes=1)

@bounded_analysis
def analyze_rbc(frames,params):
    if len({r['image'].shape for r in frames})!=1:raise ValueError('One recording requires consistent image dimensions/channels.')
    m=rbc_engine();p=copy.deepcopy(params)
    uploads=[Upload(r['name'],r['payload']) for r in frames]
    with ENGINE_LOCK:result=m.analyze_dataset(uploads,[],'RBC Gap Mode',p)
    if len(result['metrics'])!=len(frames) or len(result['previews'])!=len(frames):
        raise RuntimeError('Frozen engine returned incomplete coverage; results were not accepted.')
    return result

def compare_masks(pred,ref):
    if pred.shape!=ref.shape:raise ValueError('Prediction and reference must have exactly the same dimensions; no registration or resizing.')
    if not ref.any():raise ValueError('Empty reference: whole-cell geometry/boundary comparison is not evaluable.')
    # Existing comparison includes single-region shape interpretation, not cell identity.
    return reference_metrics(pred,ref)

def json_safe(v):
    if isinstance(v,dict):return {str(k):json_safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [json_safe(x) for x in v]
    if isinstance(v,np.ndarray):return json_safe(v.tolist())
    if isinstance(v,np.generic):return json_safe(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    if isinstance(v,(str,int,float,bool)) or v is None:return v
    return str(v)

def fixture_frames(n=8):
    """First-party software fixtures only; never biological accuracy evidence."""
    yy,xx=np.mgrid[:128,:160];out=[]
    for i in range(n):
        a=np.full((128,160),35,dtype=np.uint8)
        for j in range(5):
            mask=((xx-(20+j*29))**2/(6+i*.5+j)**2+(yy-(24+(j%3)*35))**2/(9+j)**2)<1
            a[mask]=190+min(i*4,30)
        out.append((f'synthetic_frame_{i:03d}.png',native.png_bytes(a)))
    return out
