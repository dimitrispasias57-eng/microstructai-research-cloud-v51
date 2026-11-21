"""Filename-only compatibility amendment; frozen track mathematics unchanged."""
import hashlib
import re
import xml.etree.ElementTree as ET
from .tracking import import_cimrak

VERSION='0.1.0-dev.2-import-only'

def import_tracks(payload, *, sequence_id):
    if not payload or len(payload)>8*1024*1024 or b'<!DOCTYPE' in payload.upper() or b'<!ENTITY' in payload.upper():
        raise ValueError('Require bounded XML without DTD/entity declarations.')
    try:root=ET.fromstring(payload)
    except ET.ParseError as exc:raise ValueError('Malformed XML.') from exc
    images=root.findall('images/image')
    names=[i.findtext('src','') for i in images]
    bases=[n.replace('\\','/').split('/')[-1] for n in names]
    if all(re.fullmatch(r'image\d+\.png',n,re.I) for n in bases):
        return import_cimrak(payload,sequence_id=sequence_id)
    matches=[re.fullmatch(r'video(\d+)_(\d+)\.tiff?',n,re.I) for n in bases]
    if not matches or not all(matches) or len({m.group(1) for m in matches})!=1:
        raise ValueError('Unsupported or mixed source filename schema/recording.')
    originals={}
    for image,m,original in zip(images,matches,names):
        frame=int(m.group(2));originals[frame]=original
        image.find('src').text=f'image{frame}.png'
    result=import_cimrak(ET.tostring(root),sequence_id=sequence_id)
    for row in result['rows']:row['source_frame']=originals[row['frame']]
    result['source_sha256']=hashlib.sha256(payload).hexdigest()
    result['frame_convention']='Numeric video<recording>_<frame>.tiff suffix retained under dated import amendment 01; not acquisition seconds.'
    result['import_adapter_version']=VERSION
    return result
