"""Product orchestration; frozen engine modules are imported without edits."""
from __future__ import annotations
import base64
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata as metadata
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'engine'))
import numpy as np
import pandas as pd
from PIL import Image
import tifffile
from modules.structural_core import measure_mask, CORE_VERSION, SCHEMA_VERSION
from modules.v4_masks import supplied_mask, crop, validate_roi
from modules.v4_ingestion import array_hash, orient, display_u8
from modules.v4_diagnostics import image_diagnostics, mask_diagnostics
from definitions import METRICS, definition_rows, QC_NOTE, COMPONENT_WARNING, SCOPE

VERSION = '4.1.0-beta.6'
MAX_PIXELS = 1_000_000
MAX_BYTES = 8 * 1024 * 1024
MAX_BATCH_PIXELS = 8_000_000
MAX_BATCH_BYTES = 32 * 1024 * 1024
MAX_BATCH_FILES = 10
MAX_COMPONENTS = 20_000
# Shared capacity control only: no uploads or results are stored in this object.
ANALYSIS_SLOT = threading.BoundedSemaphore(1)

def safe(value):
    if isinstance(value, dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [safe(v) for v in value]
    if isinstance(value, np.generic): return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value

def json_bytes(value):
    return json.dumps(safe(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')

def digest(value): return hashlib.sha256(json_bytes(value)).hexdigest()

@dataclass(frozen=True)
class Settings:
    foreground: str = 'nonzero/high'
    meaning: str = 'User-defined foreground'
    interpretation: str = 'binary'
    label_union_confirmed: bool = False
    threshold: float | None = None
    roi: tuple | None = None
    um_per_pixel: float | None = None
    scale_source: str = ''
    stack_policy: str = 'reject'
    page: int = 1
    apply_orientation: bool = False

def validate_settings(s):
    if not s.meaning.strip(): raise ValueError('Describe what the foreground represents.')
    if s.foreground not in ('nonzero/high','zero/low'): raise ValueError('Choose foreground polarity explicitly.')
    if s.interpretation == 'labels' and not s.label_union_confirmed:
        raise ValueError('Confirm that integer labels will be merged into one foreground; this is not instance analysis.')
    if s.um_per_pixel is not None:
        if isinstance(s.um_per_pixel,bool) or not np.isfinite(s.um_per_pixel) or s.um_per_pixel <= 0:
            raise ValueError('Calibration must be a positive finite micrometres-per-pixel value.')
        if not s.scale_source.strip(): raise ValueError('Describe the calibration source; metadata is not inferred.')
    if s.stack_policy not in ('reject','selected'): raise ValueError('Choose reject stacks or an explicit 1-based page.')

def decode(payload, name, settings, mask=False):
    if not payload or len(payload)>MAX_BYTES: raise ValueError('Empty file or file exceeds the hosted 8 MiB limit. Export a supported smaller 2-D file.')
    transformations=[]
    try:
        if payload[:4] in (b'II*\0',b'MM\0*',b'II+\0',b'MM\0+'):
            with tifffile.TiffFile(io.BytesIO(payload)) as tif:
                count=len(tif.pages)
                if count!=1 and settings.stack_policy=='reject': raise ValueError(f'TIFF has {count} pages. Select a page explicitly; no page was discarded silently.')
                if settings.page<1 or int(settings.page)!=settings.page or settings.page>count: raise ValueError(f'TIFF page must be between 1 and {count}.')
                if (tif.is_ome or any(p.subifds for p in tif.pages)) and settings.stack_policy!='selected': raise ValueError('Complex TIFF requires an explicitly selected 2-D page.')
                index=settings.page-1 if settings.stack_policy=='selected' else 0
                page=tif.pages[index]
                if isinstance(page,tifffile.TiffFrame): page=page.aspage()
                if page.axes not in ('YX','YXS') or page.photometric.name not in ('MINISBLACK','MINISWHITE','RGB'): raise ValueError('Unsupported TIFF axes/photometric. Export a 2-D grayscale or RGB page; no automatic conversion.')
                if math.prod(page.shape[:2])>MAX_PIXELS: raise ValueError('Image exceeds the hosted 1-million-pixel guard. No resizing is performed.')
                a=page.asarray()
                orientation=int(page.tags[274].value) if 274 in page.tags else 1
                source_depth=page.bitspersample
                decoder='tifffile '+tifffile.__version__
                fmt='TIFF'
                photometric=page.photometric.name
        else:
            with Image.open(io.BytesIO(payload)) as im:
                fmt=im.format
                if fmt not in ('PNG','BMP','JPEG'): raise ValueError('Supported inputs: PNG, TIFF, BMP; JPEG only as optional image.')
                if mask and fmt=='JPEG': raise ValueError('JPEG masks are lossy and unsupported. Export a lossless PNG/TIFF mask.')
                if im.width*im.height>MAX_PIXELS: raise ValueError('Image exceeds the hosted 1-million-pixel guard. No resizing is performed.')
                if im.mode not in ('1','L','I','I;16','I;16L','I;16B','RGB'): raise ValueError('Unsupported palette/alpha/colour mode. Export explicit grayscale or RGB; no silent colour conversion.')
                count=getattr(im,'n_frames',1)
                if count!=1: raise ValueError('Animated or multipage non-TIFF input is unsupported. Export a single frame.')
                orientation=int(im.getexif().get(274,1))
                a=np.array(im)
                source_depth=1 if im.mode=='1' else (16 if '16' in im.mode else a.dtype.itemsize*8)
                decoder='Pillow '+Image.__version__
                index=0; photometric=None
    except (ValueError,OSError,KeyError,IndexError,EOFError) as exc:
        raise ValueError(f'{name}: cannot read supported input. {exc}') from exc
    if a.ndim not in (2,3) or (a.ndim==3 and a.shape[2]!=3) or a.dtype.kind not in 'buif' or not np.isfinite(a).all(): raise ValueError(f'{name}: require finite 2-D grayscale or interleaved RGB pixels.')
    before=list(a.shape)
    if orientation!=1:
        if not settings.apply_orientation: raise ValueError(f'{name}: orientation tag {orientation}; explicitly enable orientation handling or export an upright file.')
        a=orient(a,orientation);transformations.append(f'Explicit orientation {orientation} applied')
    return a,dict(filename=name,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload),dimensions=before,analysed_orientation_dimensions=list(a.shape),dtype=str(a.dtype),bit_depth=int(source_depth),format=fmt,decoder=decoder,pages=count,page_1based=index+1,orientation_tag=orientation,photometric=photometric,transformations=transformations,decoded_sha256=array_hash(a))

def prepare(mask_bytes, mask_name, settings, image_bytes=None, image_name=None, identity=None):
    validate_settings(settings)
    a,mp=decode(mask_bytes,mask_name,settings,mask=True)
    img,ip=decode(image_bytes,image_name,settings) if image_bytes is not None else (None,None)
    binary,interpretation=supplied_mask(a,interpretation=settings.interpretation,foreground=settings.foreground,threshold=settings.threshold,image_shape=img.shape if img is not None else None)
    roi=validate_roi(settings.roi,binary.shape)
    transformations=list(mp['transformations'])
    if settings.foreground=='zero/low': transformations.append('Explicit foreground polarity inversion')
    if settings.interpretation!='binary': transformations.append('Explicit mask interpretation: '+settings.interpretation)
    if settings.roi is not None: transformations.append('Explicit rectangular ROI crop; 1-based x,y,width,height')
    warnings=list(interpretation['warnings'])
    if mp.get('photometric')=='MINISWHITE': warnings.append('MINISWHITE numeric values preserved; no photometric inversion. Confirm foreground.')
    if img is not None: warnings.append('Equal dimensions do not prove image-mask alignment. Overlay review is an internal check.')
    return dict(mask=crop(binary,roi),image=crop(img,roi) if img is not None else None,original=binary,settings=asdict(settings),identity=identity or {},mask_input=mp,image_input=ip,roi=list(roi),transformations=transformations or ['none; supplied binary mask unchanged'],warnings=warnings,interpretation=interpretation)

def physical_metrics(values, scale):
    if scale is None: return {}
    if isinstance(scale,bool) or not math.isfinite(scale) or scale<=0: raise ValueError('Invalid physical calibration.')
    result={}
    for key,value in values.items():
        unit=METRICS[key][3]
        if unit=='px': result[key+'_um']=value*scale
        elif unit=='px²': result[key+'_um2']=value*scale*scale
        elif unit=='1/px': result[key+'_per_um']=value/scale
    return result

def numerical_payload(result):
    return dict(metrics=result['metrics'],physical_metrics=result['physical_metrics'],components=result['components'])

def analyze(prepared):
    mask=prepared['mask']
    from scipy import ndimage
    _, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count > MAX_COMPONENTS:
        raise ValueError('More than 20,000 connected components: exceeds the hosted report/resource guard. No components were deleted.')
    values,components=measure_mask(mask)
    physical=physical_metrics(values,prepared['settings']['um_per_pixel'])
    if prepared['settings']['um_per_pixel'] is not None:
        scale=prepared['settings']['um_per_pixel']
        components=components.assign(area_um2=components.area_px2*scale**2,equivalent_diameter_um=components.equivalent_diameter_px*scale)
    diagnostics=image_diagnostics(prepared['image'])+mask_diagnostics(mask)
    diagnostics.append(dict(category='ANALYSIS ELIGIBILITY',diagnostic='validated_2d_binary_mask',status='PASS',value=True,reason='Eligible for defined mask measurements; not proof of scientific interpretation or segmentation accuracy.'))
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'engine/modules').glob('*.py'))}
    provenance=dict(app_version=VERSION,core_version=CORE_VERSION,metric_schema=SCHEMA_VERSION,product_schema='4.1-product-1',timestamp_utc=datetime.now(timezone.utc).isoformat(),settings=prepared['settings'],settings_sha256=digest(prepared['settings']),identity=prepared['identity'],mask_input=prepared['mask_input'],image_input=prepared['image_input'],roi_1based_xywh=prepared['roi'],analysed_dimensions=list(mask.shape),foreground_definition=prepared['settings']['meaning'],mask_sha256=array_hash(mask),transformations=prepared['transformations'],source_hashes=source_hashes,dependencies={n:metadata.version(n) for n in ('numpy','scipy','scikit-image','pandas','pillow','tifffile')},qc_note=QC_NOTE,mask_review='User confirmation is an internal check, not independent validation',calibration='Isotropic user-supplied micrometres/pixel; no metadata inference',network_conventions='The network descriptor internally filters components <20 px²; original supplied mask and geometric descriptors remain unfiltered. Legacy radial Xi and anisotropy use their documented working-array transforms.')
    result=dict(metrics=values,physical_metrics=physical,components=components.to_dict('records'),diagnostics=diagnostics,provenance=provenance,warnings=prepared['warnings'],preview_png=preview_png(mask,prepared['image']))
    provenance['runtime']={'python':sys.version,'platform':sys.platform,'machine':__import__('platform').machine()}
    from runtime_profile import describe
    provenance['runtime']['numerical_dispatch'] = describe()
    from hosted_runtime import provenance as hosted_runtime_provenance
    provenance['runtime']['hosted_release'] = hosted_runtime_provenance()
    provenance['delivery']={'model':'hosted Streamlit candidate','live_acceptance_passed':False,
                            'file_byte_limit':MAX_BYTES,'image_pixel_limit':MAX_PIXELS,
                            'mask_batch_limit':MAX_BATCH_FILES,'batch_byte_limit':MAX_BATCH_BYTES,
                            'combined_batch_pixel_limit':MAX_BATCH_PIXELS,'component_limit':MAX_COMPONENTS,
                            'processing_location':'hosting server; not browser-only'}
    try: provenance['dependencies']['imagecodecs']=metadata.version('imagecodecs')
    except metadata.PackageNotFoundError: provenance['dependencies']['imagecodecs']='not installed'
    provenance['product_source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'product').glob('*.py'))}
    result['provenance']['numerical_sha256']=digest(numerical_payload(result))
    return result

def preview_png(mask,image=None):
    if image is None:
        a=np.zeros((*mask.shape,3),dtype=np.uint8);a[mask]=[20,190,210]
    else:
        a=display_u8(image)
        if a.ndim==2:a=np.repeat(a[...,None],3,axis=2)
        a=a.copy(); a[mask]=(a[mask].astype(float)*.6+np.array([20,190,210])*.4).astype(np.uint8)
    b=io.BytesIO(); Image.fromarray(a).save(b,format='PNG');return b.getvalue()

def metric_rows(result):
    return [dict(key=k,name=METRICS[k][2],group=METRICS[k][0],tier=METRICS[k][1],value=safe(v),units=METRICS[k][3]) for k,v in result['metrics'].items()]

def csv_bytes(rows):
    rows=list(rows)
    if not rows:return b''
    columns=list(dict.fromkeys(k for r in rows for k in r))
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=columns);writer.writeheader()
    for row in rows:
        # Formula-injection mitigation for textual metadata only, never numeric measurements.
        safe_row={k:("'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else safe(v)) for k,v in row.items()}
        writer.writerow(safe_row)
    return buf.getvalue().encode('utf-8-sig')

def summary_row(result):
    p=result['provenance']
    return dict(**{k:p['identity'].get(k,'') for k in ('sample_id','group','condition','replicate','specimen_id')},filename=p['mask_input']['filename'],mask_file_sha256=p['mask_input']['sha256'],analysed_mask_sha256=p['mask_sha256'],numerical_sha256=p['numerical_sha256'],app_version=VERSION,core_version=CORE_VERSION,settings_sha256=p['settings_sha256'],provenance_json=json_bytes(p).decode(),**result['metrics'],**result['physical_metrics'])

def report_html(result):
    e=lambda x:html.escape(str(x))
    def table(rows):
        if not rows:return '<p>None.</p>'
        cols=list(rows[0]); return '<table><tr>'+''.join('<th>'+e(c)+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+e(r.get(c,''))+'</td>' for c in cols)+'</tr>' for r in rows)+'</table>'
    p=result['provenance']
    inputs=[dict(role=role,filename=meta['filename'],dimensions=str(meta['dimensions']),bit_depth=meta['bit_depth'],sha256=meta['sha256'])
            for role,meta in [('mask',p['mask_input']),('image',p['image_input'])] if meta is not None]
    record={'foreground_definition':p['foreground_definition'],'polarity':p['settings']['foreground'],'roi_1based_xywh':p['roi_1based_xywh'],'calibration_um_per_pixel':p['settings']['um_per_pixel'],'calibration_source':p['settings']['scale_source'],'transformations':p['transformations'],'identity':p['identity']}
    parts=[
        '<!doctype html><html><head><meta charset="utf-8"><title>MicroStructAI Research report</title><style>body{font:15px system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#172b3a}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #bccbd4;padding:7px;text-align:left;overflow-wrap:anywhere}th{background:#edf5f7}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}img{max-width:100%;max-height:440px;image-rendering:auto}h2{margin-top:30px} @media print{body{margin:10mm;padding:0}tr{break-inside:avoid}h2{break-after:avoid}}</style></head><body>',
        '<h1>MicroStructAI Research</h1><p>Structural characterization — supplied/reviewed 2-D mask</p><p>'+e(VERSION+' | '+CORE_VERSION+' | '+p['timestamp_utc'])+'</p><p>'+e(SCOPE)+'</p>',
        '<h2>Analysis record</h2>'+table(inputs)+'<pre>'+e(json_bytes(record).decode())+'</pre>',
        '<h2>Preview / overlay</h2><p>Display-only image scaling; cyan marks foreground. Not independent validation.</p><img alt="Analysed mask overlay" src="data:image/png;base64,'+base64.b64encode(result['preview_png']).decode()+'">',
        '<h2>Native metric table</h2>'+table(metric_rows(result)),
        '<h2>Additional physical-unit values</h2>'+table([dict(metric=k,value=safe(v),units=('µm²' if k.endswith('_um2') else '1/µm' if k.endswith('_per_um') else 'µm')) for k,v in result['physical_metrics'].items()]),
        '<h2>QC and warnings</h2><p>'+e(QC_NOTE)+'</p>'+table(result['diagnostics'])+'<pre>'+e('\n'.join(result['warnings']))+'</pre>',
        '<h2>Metric definitions and interpretation</h2><p>Tier 1: direct geometry. Tier 2: spatial/topological descriptors. Tier 3: network-specific descriptors; physical meaning depends on foreground semantics.</p>'+table(definition_rows()),
        '<h2>Limitations</h2><p>'+e(COMPONENT_WARNING)+'</p><p>No clinical, diagnostic, universal-segmentation or validated aggregation claim. ROI, scale, connectivity, resolution and mask preparation affect results. No inferential p-values or universal structural-health score. NaN/nonfinite values are exported as unavailable/null, not replaced with zero; historical empty-component mean/median conventions remain zero.</p>',
        '<h2>Complete reproducibility manifest</h2><pre>'+e(json_bytes(p).decode())+'</pre></body></html>',
    ]
    return ''.join(parts).encode('utf-8')

def bundle(result):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        files={'report.html':report_html(result),'provenance.json':json_bytes(result['provenance']),'metrics.csv':csv_bytes([summary_row(result)]),'components.csv':csv_bytes(result['components']),'definitions.json':json_bytes(definition_rows()),'diagnostics.json':json_bytes(result['diagnostics']),'preview.png':result['preview_png']}
        for name,data in files.items():
            info=zipfile.ZipInfo(name,date_time=(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
    return b.getvalue()

def export_result(result,folder):
    target=Path(folder)
    target.mkdir(parents=True,exist_ok=True)
    dest=target/(result['provenance']['mask_sha256'][:12]+'_'+result['provenance']['numerical_sha256'][:12]+'.zip')
    # No overwrites. Identical repeat exports get a new suffix without modifying prior evidence.
    for index in range(10000):
        candidate=dest if index==0 else dest.with_stem(dest.stem+'_'+str(index))
        try:
            with candidate.open('xb') as f:f.write(bundle(result))
            return candidate
        except FileExistsError:continue
        except OSError as exc:raise ValueError('Cannot write output folder. Choose a writable local folder with a shorter path. '+str(exc)) from exc
    raise ValueError('Too many identical exports in this folder.')

class BatchJob:
    def __init__(self, prepared, errors=()):
        self.prepared=prepared;self.results=[];self.errors=list(errors);self.stop=threading.Event();self.done=False;self.progress=0;self.elapsed_seconds=0
    def run(self):
        started=time.monotonic()
        acquired=ANALYSIS_SLOT.acquire(blocking=False)
        try:
            if not acquired:
                self.errors.extend(dict(filename=p['mask_input']['filename'],status='not_run_busy',error='Server analysis slot is busy. Please retry shortly.') for p in self.prepared)
                return
            for i,p in enumerate(self.prepared):
                if time.monotonic()-started > 60:
                    self.errors.extend(dict(filename=q['mask_input']['filename'],status='not_run_time_budget',error='60-second batch budget reached between files. Split the batch.') for q in self.prepared[i:]);break
                if self.stop.is_set():
                    self.errors.extend(dict(filename=q['mask_input']['filename'],status='not_run_interrupted',error='Stopped by user before this file.') for q in self.prepared[i:]);break
                try:self.results.append(analyze(p))
                except Exception as exc:self.errors.append(dict(filename=p['mask_input']['filename'],status='failed',error=str(exc)))
                self.progress=i+1
        finally:
            if acquired:ANALYSIS_SLOT.release()
            self.elapsed_seconds=time.monotonic()-started
            self.done=True
    def start(self):
        threading.Thread(target=self.run,daemon=True).start();return self

def comparison(rows,metric,unit='image'):
    """Descriptive only: explicit aggregation unit, no inferred independence."""
    df=pd.DataFrame(rows)
    if metric not in df:raise ValueError('Selected metric is unavailable.')
    for k in ('group','condition'): 
        if k not in df:df[k]=''
    df[metric]=pd.to_numeric(df[metric],errors='coerce')
    df=df.replace([np.inf,-np.inf],np.nan)
    if unit!='image':
        if unit not in ('sample_id','specimen_id') or unit not in df or df[unit].fillna('').eq('').any():raise ValueError('Explicit IDs are required for sample/specimen summaries; independence is not inferred.')
        df=df.groupby(['group','condition',unit],dropna=False,as_index=False)[metric].median()
    stats=df.groupby(['group','condition'],dropna=False)[metric].agg(n_finite='count',median='median',q1=lambda s:s.quantile(.25),q3=lambda s:s.quantile(.75)).reset_index()
    stats['IQR']=stats.q3-stats.q1;stats['summary_unit']=unit
    return df,stats
