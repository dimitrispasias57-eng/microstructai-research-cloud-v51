"""MicroStructAI isolated crop-target DEVELOPMENT prototype 0.1, not deployed.

Inference accepts image pixels and a fixed configuration only. Reference masks,
filenames, coordinates, labels and per-object reference sizes are not inputs.
"""
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
VERSION='pathologics-crop-dev-0.1'
STRUCT=np.ones((3,3),bool)
CANDIDATES=[dict(id=f'{channel}_t{factor:.2f}_c{radius}',channel=channel,factor=factor,closing_radius=radius)
            for channel in ('green_darkness','max_darkness') for factor in (.85,1.,1.15) for radius in (0,1)]

def segment(rgb,config):
    a=np.asarray(rgb)
    if a.dtype!=np.uint8 or a.shape!=(80,80,3):raise ValueError('Require 80x80 RGB uint8; no resizing')
    signal=255-a[...,1] if config['channel']=='green_darkness' else 255-a.min(axis=2)
    if signal.max()==signal.min():return np.zeros((80,80),bool)
    raw=signal>float(threshold_otsu(signal))*config['factor']
    if config['closing_radius']==1:
        # Explicit zero exterior, no reference-driven repair or target fitting.
        raw=ndimage.binary_closing(raw,structure=STRUCT)
    filled=ndimage.binary_fill_holes(raw)
    labels,n=ndimage.label(filled,STRUCT)
    if n==0:return np.zeros_like(raw)
    yy,xx=np.nonzero(labels);ids=labels[yy,xx]
    min_dist=np.full(n+1,np.inf)
    np.minimum.at(min_dist,ids,(yy-39.5)**2+(xx-39.5)**2)
    areas=np.bincount(labels.ravel(),minlength=n+1)
    chosen=min(range(1,n+1),key=lambda i:(min_dist[i],-areas[i],i))
    return labels==chosen

def agreement(pred,ref,boundary=True):
    if pred.shape!=ref.shape or pred.dtype!=bool or ref.dtype!=bool:raise ValueError('Require matching boolean masks')
    p=int(pred.sum());r=int(ref.sum())
    if r==0:raise ValueError('Empty reference not a valid target')
    tp=int(np.count_nonzero(pred&ref));fp=p-tp;fn=r-tp
    out=dict(tp=tp,fp=fp,fn=fn,reference_area_px2=r,predicted_area_px2=p,
             dice=2*tp/(p+r),iou=tp/(tp+fp+fn),precision=tp/p if p else None,recall=tp/r,
             signed_area_bias_fraction=(p-r)/r,signed_area_bias_px2=p-r,
             boundary_mean_px=None,boundary_p95_px=None)
    if boundary and p:
        pb=pred&~ndimage.binary_erosion(pred,STRUCT,border_value=0)
        rb=ref&~ndimage.binary_erosion(ref,STRUCT,border_value=0)
        distances=np.concatenate((ndimage.distance_transform_edt(~rb)[pb],ndimage.distance_transform_edt(~pb)[rb]))
        out['boundary_mean_px']=float(distances.mean());out['boundary_p95_px']=float(np.quantile(distances,.95))
    return out
