"""Session-owned review transactions wrapping unchanged measurement engines.

No uploaded bytes/results are stored in module globals or process caches.
Acceptance binds to exact mask/image/settings/algorithm; QC cannot accept.
"""
from __future__ import annotations
import bootstrap
from bootstrap import HERE, CODES, VERSION, PRODUCT_VERSION, DELIVERY, PATH_SOURCE, PATH_HASH
import base64
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import html
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import service
from morphology import engine as morphology
from morphology import io as morphology_io

UNREVIEWED = 'UNREVIEWED AUTOMATIC MASK'
ACCEPTED = 'REVIEWED / ACCEPTED MASK'
SUPPLIED = 'USER-SUPPLIED MASK'
AUTO_NOTE = 'Automatic segmentation is a proposed mask and may contain errors. Review the mask before using quantitative measurements.'
ACCEPT_NOTE = 'The user reviewed and accepted this mask for analysis.'
QC_NOTE = 'This does not establish segmentation accuracy.'
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 1_000_000
MAX_BATCH = 100
PAGE_SIZE = 10
MAX_BATCH_BYTES = 32 * 1024 * 1024
MAX_BATCH_PIXELS = 8_000_000

def sha(data): return hashlib.sha256(data).hexdigest()
def utc(): return datetime.now(timezone.utc).isoformat()
def workflow_hashes():
    return {name:sha((HERE/name).read_bytes()) for name in ('bootstrap.py','review_core.py','review_ui.py')}
def array_hash(a):
    return sha(str(a.dtype).encode() + str(a.shape).encode() + np.ascontiguousarray(a).tobytes())
def png(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, format='PNG'); return b.getvalue()
def frozen_array(a):
    a = np.ascontiguousarray(a).copy(); a.setflags(write=False); return a

@dataclass(frozen=True)
class Context:
    mode: str = 'structural'
    meaning: str = 'User-defined foreground'
    polarity: str = 'nonzero/high'
    roi: tuple | None = None  # 1-based x,y,width,height, same as structural core
    um_per_pixel: float | None = None
    calibration_source: str = ''
    label_union_confirmed: bool = False
    identity_confirmed: bool = False

def decode(data, name, *, mask=False):
    if not data or len(data) > MAX_BYTES: raise ValueError('Empty file or above 8 MiB limit.')
    # Check dimensions before delegating to the unchanged input decoder.
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > MAX_PIXELS: raise ValueError('Above 1 million pixels; no resizing.')
    except (OSError, EOFError) as exc: raise ValueError('Cannot decode file; it may be corrupted.') from exc
    return service.decode(data, Path(name).name, service.Settings(), mask=mask)[0]

def validate_mask(a, kind):
    if kind not in ('binary', 'instances'): raise ValueError('Unknown mask interpretation.')
    if a.ndim != 2 or a.dtype.kind not in 'bui' or np.any(a < 0):
        raise ValueError('Use a 2-D nonnegative integer grayscale mask; no RGB or silent conversion.')
    if a.size > MAX_PIXELS or not a.size: raise ValueError('Invalid mask size.')
    if kind == 'binary':
        values = np.unique(a)
        if len(values) > 2 or (len(values) == 2 and values[0] != 0):
            raise ValueError('Binary mask must contain zero and at most one foreground value. Choose instances explicitly for IDs.')
    else: morphology.validate_labels(a)
    return frozen_array(a)

def validate_context(c, mask, kind):
    if c.mode not in ('structural', 'morphology'): raise ValueError('Unknown measurement mode.')
    if kind == 'instances' and c.polarity != 'nonzero/high':
        raise ValueError('Instance background is zero; polarity inversion would destroy identity.')
    service.validate_roi(c.roi, mask.shape)
    service.validate_settings(service.Settings(meaning=c.meaning, foreground=c.polarity,
        um_per_pixel=c.um_per_pixel, scale_source=c.calibration_source))

@dataclass
class Record:
    name: str
    mask: np.ndarray
    mask_bytes: bytes
    image: np.ndarray | None
    image_bytes: bytes | None
    image_name: str | None
    kind: str
    context: Context
    automatic: bool
    algorithm: dict
    state: str
    decision: str
    original_mask_sha256: str
    original_mask_file_sha256: str
    original_mask_bytes: bytes = b''
    replacements: list = field(default_factory=list)
    modified: bool = False
    replacement_uploaded: bool = False
    review_timestamp: str | None = None
    accepted_token: str | None = None
    revision: int = 0
    events: list = field(default_factory=list)
    result: dict | None = None
    result_token: str | None = None

    def token(self):
        return service.digest(dict(mask=array_hash(self.mask), mask_file=sha(self.mask_bytes),
            image=sha(self.image_bytes) if self.image_bytes is not None else None,
            decoded_image=array_hash(self.image) if self.image is not None else None,
            name=self.name, kind=self.kind, context=asdict(self.context), algorithm=self.algorithm,
            review_workflow_source_hashes=workflow_hashes()))

    @property
    def queue_status(self):
        if self.decision == 'REJECT': return 'REJECTED'
        if self.replacement_uploaded: return 'REPLACED'
        if self.decision == 'ACCEPT' and self.eligible: return 'ACCEPTED'
        return 'UNREVIEWED'

    def retained_bytes(self):
        return len(self.original_mask_bytes) + len(self.image_bytes or b'') + sum(len(x['bytes']) for x in self.replacements)

    @property
    def eligible(self):
        if self.decision in ('REJECT', 'REVIEW'): return False
        if not self.automatic and self.state == SUPPLIED: return True
        return self.state == ACCEPTED and self.decision == 'ACCEPT' and self.accepted_token == self.token()

    def invalidate(self):
        self.result = self.result_token = None
        self.accepted_token = self.review_timestamp = None
        self.state = UNREVIEWED if self.automatic else SUPPLIED
        # Even supplied masks require a fresh decision after correction/settings
        # changes. In particular, changing a setting cannot un-reject a mask.
        self.decision = 'REVIEW'
        self.revision += 1

    def decide(self, action, confirmed=False):
        if action not in ('ACCEPT', 'REJECT', 'REVIEW'): raise ValueError('Unknown review decision.')
        if action == 'ACCEPT' and not confirmed: raise ValueError('Explicit visual-review confirmation required.')
        self.invalidate()
        self.decision = action
        if action == 'ACCEPT':
            self.review_timestamp = utc()
            self.accepted_token = self.token()
            self.state = ACCEPTED
        self.events.append(dict(action=action, timestamp_utc=utc(), token=self.token()))

    def replace(self, data, name):
        if len(self.replacements) >= 10: raise ValueError('At most 10 retained replacements per case. Export and start a new queue.')
        a = validate_mask(decode(data, name, mask=True), self.kind)
        if a.shape != self.mask.shape: raise ValueError('Replacement must have exactly the original dimensions; no resizing.')
        self.mask, self.mask_bytes = a, bytes(data)
        self.modified = array_hash(a) != self.original_mask_sha256
        self.replacement_uploaded = True
        self.replacements.append(dict(filename=Path(name).name, bytes=bytes(data),
            file_sha256=sha(data), decoded_sha256=array_hash(a), timestamp_utc=utc()))
        self.invalidate()
        # Replacement needs an explicit new review even when original was supplied.
        self.decision = 'REVIEW'
        self.events.append(dict(action='REPLACE', filename=Path(name).name,
            file_sha256=sha(data), timestamp_utc=utc(), token=self.token()))

    def configure(self, context):
        validate_context(context, self.mask, self.kind)
        if context != self.context:
            self.context = context; self.invalidate()
            self.events.append(dict(action='SETTINGS_CHANGED', timestamp_utc=utc(), token=self.token()))

    def provenance(self):
        if not self.eligible: raise ValueError('Measurement/export blocked: this mask needs explicit review and acceptance.')
        source = ('AUTOMATIC + USER REPLACED' if self.replacement_uploaded else 'AUTOMATIC + USER-ACCEPTED') if self.automatic else 'USER-SUPPLIED'
        return dict(workflow_version=VERSION, app_version=PRODUCT_VERSION, review_workflow_source_hashes=workflow_hashes(), mask_source=source, review_status=self.state,
            queue_status=self.queue_status, replacement_mask_sha256=sha(self.replacements[-1]['bytes']) if self.replacements else None,
            replacement_history=[{k:v for k,v in x.items() if k!='bytes'} for x in self.replacements],
            decision=self.decision, segmentation=self.algorithm, review_timestamp_utc=self.review_timestamp,
            user_modified_mask=self.modified, replacement_uploaded=self.replacement_uploaded,
            user_correction='YES' if self.modified else 'NO',
            roi_1based_xywh=list(service.validate_roi(self.context.roi, self.mask.shape)),
            calibration_um_per_pixel=self.context.um_per_pixel, calibration_source=self.context.calibration_source,
            input_image_sha256=sha(self.image_bytes) if self.image_bytes is not None else None,
            decoded_image_sha256=array_hash(self.image) if self.image is not None else None,
            input_image_filename=self.image_name,
            original_mask_sha256=self.original_mask_sha256, original_mask_file_sha256=self.original_mask_file_sha256,
            final_mask_sha256=array_hash(self.mask), final_mask_file_sha256=sha(self.mask_bytes),
            final_mask_hash_convention='SHA256(dtype text + shape tuple text + C-order decoded bytes); full image before ROI',
            mask_kind=self.kind, positive_ids=[int(v) for v in np.unique(self.mask) if v > 0] if self.kind == 'instances' else None,
            acceptance_token=self.accepted_token, input_contract_sha256=self.token(), context=asdict(self.context),
            events=self.events.copy(), review_note=ACCEPT_NOTE if self.state == ACCEPTED else 'User-supplied mask; acceptance not inferred.',
            limitation='User review is an internal check, not independent validation. '+QC_NOTE,
            execution=DELIVERY+'; user review is not independent validation.')

def make_record(name, mask_bytes, *, image_bytes=None, image_name=None, kind='binary', context=Context(), algorithm=None):
    mask = validate_mask(decode(mask_bytes, name, mask=True), kind)
    image = decode(image_bytes, image_name or 'image.png') if image_bytes is not None else None
    if image is not None and image.shape[:2] != mask.shape: raise ValueError('Image/mask dimensions differ; no resizing.')
    validate_context(context, mask, kind)
    automatic = algorithm is not None
    r = Record(Path(name).name, mask, bytes(mask_bytes), frozen_array(image) if image is not None else None,
        bytes(image_bytes) if image_bytes is not None else None, Path(image_name).name if image_name else None,
        kind, context, automatic, dict(algorithm or {'name':'User supplied', 'version':'not applicable'}),
        UNREVIEWED if automatic else SUPPLIED, 'REVIEW' if automatic else 'SUPPLIED', array_hash(mask), sha(mask_bytes), bytes(mask_bytes))
    r.events.append(dict(action='PROPOSED' if automatic else 'SUPPLIED', timestamp_utc=utc(), token=r.token()))
    return r

def checked_module(path, expected, name):
    if sha(path.read_bytes()) != expected: raise ValueError('Frozen segmentation source hash mismatch; stopped.')
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module

def propose(image_bytes, name, adapter='PathOlOgics v0.1', context=Context()):
    image = decode(image_bytes, name)
    if adapter == 'PathOlOgics v0.1':
        module = checked_module(PATH_SOURCE, PATH_HASH, 'frozen_pathologics')
        config = dict(id='max_darkness_t1.00_c0', channel='max_darkness', factor=1.0, closing_radius=0)
        mask = module.segment(image, config).astype(np.uint8)*255; kind='binary'
        algorithm = dict(name='PATHOLOGICS TARGET OBJECT MASK', version='v0.1', implementation=module.VERSION,
                         source_sha256=PATH_HASH, settings=config, scope='80x80 uint8 RGB stained-smear target crops; not necessarily one RBC')
    elif adapter == 'RedTell v0.2':
        path = CODES/'RBC_SEGMENTATION_V02_DEV/seg02.py'
        settings_path = CODES/'RBC_SEGMENTATION_V02_DEV/protocol/V02_FINAL_SETTINGS.json'
        settings_hash='c1d67283a3eb872c0d1c30f6c1ad617baf760a81c3674a68029ecc4c3fb95d10'
        if sha(settings_path.read_bytes()) != settings_hash: raise ValueError('Frozen RedTell settings hash mismatch.')
        sys.path.insert(0, str(CODES/'RBC_SEGMENTATION_DEV'))
        module = checked_module(path, '04d74ec35ada34adad4dbbf5de511ca0abb48673bcd713e311766cf9682a8d0c', 'frozen_redtell')
        config=json.loads(settings_path.read_text())['segmentation']
        mask=module.predict(image, config); kind='instances'
        if not mask.any(): raise ValueError('No proposed instances. Nothing was accepted. Supply a corrected labelled mask.')
        mask=mask.astype(np.uint16)
        algorithm=dict(name='RedTell instance proposal', version=module.VERSION, source_sha256=sha(path.read_bytes()),
                       settings=config, settings_sha256=settings_hash, scope='Domain-dependent DEVELOPMENT; no shape-based acceptance')
    else: raise ValueError('Unknown fixed adapter.')
    return make_record(Path(name).stem+'_proposal.png', png(mask), image_bytes=image_bytes,
                       image_name=name, kind=kind, context=context, algorithm=algorithm)

def foreground(r):
    return r.mask > 0 if r.context.polarity == 'nonzero/high' else r.mask == 0

def warnings(r):
    fg = foreground(r); notes=[]
    if not fg.any(): notes.append('EMPTY FOREGROUND')
    if fg.all(): notes.append('FULL FOREGROUND')
    if fg[0].any() or fg[-1].any() or fg[:,0].any() or fg[:,-1].any(): notes.append('BORDER CONTACT: possible truncated objects')
    if r.kind == 'instances':
        ids=np.unique(r.mask); ids=ids[ids>0]
        # Sparse IDs preserved; geometry plausibility only, never biological shape rejection.
        disconnected=[int(i) for i in ids if ndimage.label(r.mask==i, np.ones((3,3)))[1]>1]
        if disconnected: notes.append('DISCONNECTED IDs (inspect possible split/identity error): '+str(disconnected))
    return notes

def render(r, view='overlay', opacity=.4, ids=False):
    fg=foreground(r)
    if r.image is None: rgb=np.full((*fg.shape,3),235,np.uint8)
    else:
        rgb=service.display_u8(r.image)
        if rgb.ndim==2: rgb=np.repeat(rgb[...,None],3,axis=2)
        rgb=rgb.copy()
    if view=='original': return png(rgb)
    labels=r.mask if r.kind=='instances' else fg.astype(np.uint8)
    if r.kind=='instances':
        palette=np.stack(((labels*47+40)%210+35,(labels*83+50)%210+35,(labels*113+60)%210+35),axis=-1).astype(np.uint8)
    else: palette=np.full_like(rgb, [20,190,210])
    if view=='mask': rgb=np.zeros_like(rgb); rgb[fg]=palette[fg]
    elif view=='boundary':
        from skimage.segmentation import find_boundaries
        boundary=find_boundaries(labels,connectivity=2,mode='inner') & fg
        rgb[boundary]=[255,35,65]
    else: rgb[fg]=(rgb[fg]*(1-opacity)+palette[fg]*opacity).astype(np.uint8)
    im=Image.fromarray(rgb)
    if r.context.roi is not None and view in ('overlay','boundary'):
        x,y,w,h=service.validate_roi(r.context.roi,r.mask.shape)
        ImageDraw.Draw(im).rectangle((x-1,y-1,x+w-2,y+h-2),outline=(255,215,0),width=1)
    if ids and r.kind=='instances':
        draw=ImageDraw.Draw(im)
        for i in np.unique(labels[labels>0]):
            yy,xx=np.nonzero(labels==i); j=len(xx)//2
            draw.text((int(xx[j]),int(yy[j])),str(int(i)),fill='white',stroke_width=1,stroke_fill='black')
    b=io.BytesIO(); im.save(b,format='PNG'); return b.getvalue()

def analyze(r):
    review=r.provenance()  # Gate before calling any engine.
    c=r.context
    if c.mode=='structural':
        count=ndimage.label(foreground(r),np.ones((3,3)))[1]
        if count>20_000:raise ValueError('Above 20,000 components; no components were silently removed.')
        if r.kind=='instances' and not c.label_union_confirmed:
            raise ValueError('Explicit union confirmation required for structural analysis of labels; original IDs are retained separately.')
        settings=service.Settings(foreground=c.polarity, meaning=c.meaning,
            interpretation='labels' if r.kind=='instances' else 'binary', label_union_confirmed=c.label_union_confirmed,
            roi=c.roi, um_per_pixel=c.um_per_pixel, scale_source=c.calibration_source)
        prepared=service.prepare(r.mask_bytes,r.name,settings,r.image_bytes,r.image_name)
        result=service.analyze(prepared)
    else:
        x,y,w,h=service.validate_roi(c.roi,r.mask.shape)
        config=morphology.Config(identity_confirmed=c.identity_confirmed, mode='instances' if r.kind=='instances' else 'components',
            roi=(x-1,y-1,x-1+w,y-1+h), um_per_pixel=c.um_per_pixel, calibration_source=c.calibration_source,image_id=r.name)
        labels=r.mask if r.kind=='instances' else foreground(r).astype(np.uint8)
        result=morphology.analyze(labels,config,source={'input_mask_file_sha256':sha(r.mask_bytes)})
    result['provenance']['mask_review']=review
    result['provenance']['review_candidate_version']=VERSION
    result['provenance']['delivery']={'model':DELIVERY, 'live_acceptance_passed':False}
    result['accepted_overlay_png']=render(r,'overlay',.4,ids=r.kind=='instances')
    result['review_mode']=c.mode
    r.result=result; r.result_token=r.token()
    return result

def current_result(r):
    r.provenance()
    if r.result is None or r.result_token!=r.token(): raise ValueError('Run analysis for the current reviewed mask and settings first.')
    return r.result

def report(r):
    result=current_result(r); p=result['provenance']['mask_review']; e=html.escape
    header='<section><h1>Mask review / acceptance record</h1>'
    for key in ('mask_source','review_status','review_timestamp_utc'):
        header+='<p><strong>'+key.replace('_',' ').upper()+': </strong>'+e(str(p[key]))+'</p>'
    header+='<p>SEGMENTATION VERSION: '+e(str(p['segmentation'].get('version')))+ '</p><p>'+e(p['review_note'])+'</p>'
    header+='<p>USER MODIFICATION: '+p['user_correction']+'</p>'
    header+='<p>'+e(p['limitation'])+'</p><p>Full-resolution mask overlay (display only). ROI and calibration are recorded below.</p>'
    header+='<img style="max-width:100%;image-rendering:pixelated" alt="Accepted or supplied mask overlay" src="data:image/png;base64,'+base64.b64encode(result['accepted_overlay_png']).decode()+'"><pre>'+e(service.json_bytes(p).decode())+'</pre></section>'
    if r.context.mode=='structural':
        base=service.report_html(result).decode(); return base.replace('<body>','<body>'+header,1).encode()
    base=morphology_io.html_report(result).decode()
    return base.replace('<h1>','<body>'+header+'<h1>',1).replace('</html>','</body></html>').encode()

def bundle(r):
    result=current_result(r); b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('report.html',report(r));z.writestr('provenance.json',service.json_bytes(result['provenance']))
        z.writestr('mask_review.json',service.json_bytes(result['provenance']['mask_review']))
        z.writestr('accepted_overlay.png',result['accepted_overlay_png'])
        z.writestr('original_mask'+Path(r.name).suffix,r.original_mask_bytes)
        for i,item in enumerate(r.replacements):
            z.writestr(f'replacements/{i+1:02d}'+Path(item['filename']).suffix,item['bytes'])
        # Preserve the exact accepted upload bytes and numeric IDs, not a screenshot.
        z.writestr('final_mask'+Path(next((ev['filename'] for ev in reversed(r.events) if ev['action']=='REPLACE'),r.name)).suffix,r.mask_bytes)
        if r.context.mode=='structural':
            z.writestr('metrics.csv',service.csv_bytes([service.summary_row(result)]))
            z.writestr('components.csv',service.csv_bytes(result['components']))
            z.writestr('definitions.json',service.json_bytes(service.definition_rows()))
        else:
            z.writestr('instances.csv',morphology_io.csv_bytes(result['instances']))
            z.writestr('summaries.csv',morphology_io.csv_bytes(result['summaries']))
        payload={k:v for k,v in result.items() if not k.endswith('_png')}
        z.writestr('measurements.json',service.json_bytes(payload))
    return b.getvalue()

def batch_bundle(records):
    b=io.BytesIO(); included=[]; status=[]; rows=[]
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        for i,r in enumerate(records):
            exportable=r.eligible and r.result is not None and r.result_token==r.token()
            status.append(dict(filename=r.name,queue_status=r.queue_status,review_status=r.state,decision=r.decision,exported=exportable,
                image_filename=r.image_name,image_sha256=sha(r.image_bytes) if r.image_bytes else None,
                original_mask_sha256=r.original_mask_file_sha256,final_selected_mask_sha256=sha(r.mask_bytes),
                segmentation_version=r.algorithm.get('version'),replacement_mask_sha256=sha(r.replacements[-1]['bytes']) if r.replacements else None,
                review_timestamp_utc=r.review_timestamp,context=asdict(r.context),app_version=PRODUCT_VERSION))
            if exportable:
                z.writestr(f'case_{i+1:04d}.zip',bundle(r)); included.append(r.name)
                common=dict(filename=r.name,mask_source=r.provenance()['mask_source'],review_status=r.state,queue_status=r.queue_status)
                if r.context.mode=='structural': rows.append({**service.summary_row(r.result),**common})
                else: rows.append({**common,'measurement_route':'morphology','details':f'case_{i+1:04d}.zip'})
        if not included: raise ValueError('No current accepted/supplied analyses to export.')
        z.writestr('review_status.json',service.json_bytes(status))
        z.writestr('samples.csv',service.csv_bytes(rows))
    return b.getvalue()

class ReviewSession:
    def __init__(self): self.records={}; self.selection_token=None
    def sync(self, selection_token):
        if selection_token!=self.selection_token:
            self.records={};self.selection_token=selection_token;return True
        return False
    def reset(self): self.records={};self.selection_token=None
    def summary(self):
        result={k:0 for k in ('TOTAL','ACCEPTED','REJECTED','REPLACED','UNREVIEWED')}
        for r in self.records.values(): result['TOTAL']+=1;result[r.queue_status]+=1
        return result
    def filtered(self, status='ALL'):
        if status not in ('ALL','UNREVIEWED','ACCEPTED','REJECTED','REPLACED'):raise ValueError('Unknown status filter.')
        return [r for r in self.records.values() if status=='ALL' or r.queue_status==status]
    def page(self, status='ALL', number=1):
        records=self.filtered(status)
        if number<1:raise ValueError('Page is 1-based.')
        return records[(number-1)*PAGE_SIZE:number*PAGE_SIZE]
    def replace(self, record, data, name):
        if not any(r is record for r in self.records.values()):raise ValueError('Case is not in this session.')
        if sum(r.retained_bytes() for r in self.records.values())+len(data)>MAX_BATCH_BYTES:
            raise ValueError('Replacement exceeds 32 MiB retained-source limit; original/history will not be discarded.')
        record.replace(data,name)
    def add(self, key, record):
        if key in self.records: raise ValueError('Duplicate case key.')
        all_records=[*self.records.values(),record]
        if len(all_records)>MAX_BATCH: raise ValueError('At most 100 cases per review queue (10 per page).')
        if sum(r.retained_bytes() for r in all_records)>MAX_BATCH_BYTES:
            raise ValueError('Combined upload limit is 32 MiB.')
        if sum(r.mask.size+(r.image.shape[0]*r.image.shape[1] if r.image is not None else 0) for r in all_records)>MAX_BATCH_PIXELS:
            raise ValueError('Combined decoded limit is 8 million pixels.')
        self.records[key]=record
