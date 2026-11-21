"""Bounded mask import and explicit, traceable export. No segmentation."""
import csv
import io
import json
import html
import zipfile
from hashlib import sha256
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from .engine import validate_labels, stack_to_labels, NOTE

MAX_BYTES=16*1024*1024

def load_mask(data, name, *, stack=False):
    if not data or len(data)>MAX_BYTES:raise ValueError('Empty file or above 16 MiB limit.')
    if Path(name).suffix.lower() not in ('.png','.tif','.tiff','.bmp'):
        raise ValueError('Use lossless PNG/TIFF/BMP; no JPEG masks.')
    try:
        with Image.open(io.BytesIO(data)) as im:
            n=getattr(im,'n_frames',1);w,h=im.size
            if w*h>1_000_000 or n*w*h>8_000_000 or n>2000:
                raise ValueError('Decoded pixel/plane limit exceeded; no resizing.')
            if n!=1 and not stack:raise ValueError('Multi-page TIFF requires explicit separate-cell-stack mode, not a z/time volume.')
            planes=[]
            for index in range(n):
                im.seek(index)
                if im.mode not in ('1','L','I','I;16','I;16L','I;16B'):
                    raise ValueError('Integer grayscale mask required; RGB/palette masks are not inferred as labels.')
                plane=np.array(im)
                if plane.shape!=(h,w):raise ValueError('Stack planes must share identical dimensions.')
                planes.append(plane)
    except (UnidentifiedImageError,OSError,EOFError) as exc:
        raise ValueError('Cannot decode this mask; it may be corrupted.') from exc
    result=stack_to_labels(np.stack(planes)) if stack else validate_labels(planes[0])
    return result,dict(filename=Path(name).name,file_sha256=sha256(data).hexdigest(),
                       shape=list(result.shape),input_type='declared separate-cell stack' if stack else 'integer label image',
                       planes=n,bytes=len(data))

def load_image(data, shape):
    if not data or len(data)>MAX_BYTES:raise ValueError('Original image exceeds 16 MiB or is empty.')
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.size!=(shape[1],shape[0]) or getattr(im,'n_frames',1)!=1:
                raise ValueError('Original image must be single-frame and exactly match mask dimensions; no resize.')
            return np.array(im.convert('RGB'))
    except (UnidentifiedImageError,OSError) as exc:raise ValueError('Original image cannot be decoded.') from exc

def csv_bytes(rows):
    keys=list(dict.fromkeys(k for row in rows for k in row));stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
    for row in rows:
        cleaned={}
        for key,value in row.items():
            if isinstance(value,(dict,list,tuple)):value=json.dumps(value,ensure_ascii=False)
            if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@','\t','\r','\n')):value="'"+value
            cleaned[key]='' if value is None else value
        writer.writerow(cleaned)
    return stream.getvalue().encode('utf-8-sig')

def json_bytes(result):return json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False).encode('utf-8')

def html_report(result):
    def table(rows):
        if not rows:return '<p>No rows.</p>'
        keys=list(dict.fromkeys(k for row in rows for k in row))
        return '<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+html.escape('NOT EVALUABLE' if row.get(k) is None else str(row[k]))+'</td>' for k in keys)+'</tr>' for row in rows)+'</tbody></table>'
    p=result['provenance']
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>RBC morphology development report</title>'
       '<style>body{font:15px system-ui;margin:32px}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:6px}pre{white-space:pre-wrap}h1{color:#17384c}</style>'
       '<h1>MicroStructAI RBC — supplied-instance morphology</h1><p>'+html.escape(NOTE)+'</p>'
       '<p>Included '+str(p['included_count'])+' / '+str(p['input_count'])+'. Excluded '+str(p['excluded_count'])+'. Null values are not zeros. No biological significance tests.</p>'
       '<p>Area: px²; perimeter/diameter/axes: px. Documented calibration adds µm/µm². Orientation: radians from horizontal, axial; shape ratios are dimensionless. '
       'Perimeter uses 0.5 marching-square contours including holes. Moment axes use pixel centres; convex area uses pixel-square corners. Border/ROI clipping can distort shape.</p>'
       '<h2>Descriptive image summaries</h2>'+table(result['summaries'])+'<h2>All instances, including exclusions</h2>'+table(result['instances'])+
       '<h2>Structural phenotype (not a universal score)</h2><pre>'+html.escape(json.dumps(result['phenotype'],indent=2))+'</pre>'
       '<h2>Provenance</h2><pre>'+html.escape(json_bytes(p).decode())+'</pre></html>').encode('utf-8')

def export_bundle(result, figure_pngs=None):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('instances.csv',csv_bytes(result['instances']));z.writestr('summaries.csv',csv_bytes(result['summaries']))
        z.writestr('report.html',html_report(result));z.writestr('measurements_and_provenance.json',json_bytes(result))
        for name,data in (figure_pngs or {}).items():z.writestr(Path(name).name,data)
    return out.getvalue()
