"""Classical development RBC instance adapter. No trained weights or reference input."""
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, morphology, segmentation
from . import VERSION

def normalize(image):
    a=np.asarray(image)
    if a.ndim!=2 or a.dtype.kind not in 'uif' or a.size>1_000_000 or not np.isfinite(a).all():
        raise ValueError('Require finite native 2-D grayscale, <=1 million pixels; no resizing')
    x=a.astype(np.float64)
    lo,hi=np.percentile(x,[1,99])
    if hi<=lo:return np.zeros_like(x),0.
    return np.clip((x-lo)/(hi-lo),0,1),float(hi-lo)

def predict(image, settings):
    x,contrast=normalize(image)
    method=settings['method']
    if not contrast:fg=np.zeros(x.shape,bool)
    elif method=='canny':
        edge=feature.canny(x,sigma=settings['sigma'],low_threshold=settings['low'],high_threshold=settings['high'],use_quantiles=False)
        edge=ndi.binary_closing(edge,structure=morphology.disk(settings['close']))
        fg=ndi.binary_fill_holes(edge)
    elif method=='dark_rim':
        residual=ndi.gaussian_filter(x,settings['background_sigma'])-ndi.gaussian_filter(x,settings['sigma'])
        edge=residual>settings['threshold']
        edge=ndi.binary_closing(edge,structure=morphology.disk(settings['close']))
        fg=ndi.binary_fill_holes(edge)
    else:raise ValueError('Unknown frozen method')
    if settings['erode']>0:fg=ndi.binary_erosion(fg,structure=morphology.disk(settings['erode']))
    cc,n=ndi.label(fg,structure=np.ones((3,3)));counts=np.bincount(cc.ravel())
    keep=counts>=settings['minimum_area'];keep[0]=False;fg=keep[cc]
    distance=ndi.distance_transform_edt(fg)
    coords=feature.peak_local_max(distance,min_distance=settings['peak_distance'],threshold_abs=settings['minimum_peak_height'],exclude_border=False,labels=fg.astype(np.uint8))
    markers=np.zeros(fg.shape,np.int32)
    for k,(y,xcoord) in enumerate(coords,1):markers[y,xcoord]=k
    # Any surviving connected foreground component without a peak gets its own deterministic maximum.
    cc,n=ndi.label(fg,structure=np.ones((3,3)))
    for k,box in enumerate(ndi.find_objects(cc),1):
        if box is None:continue
        region=cc[box]==k
        if not np.any(markers[box][region]):
            values=np.where(region,distance[box],-1);y,xcoord=np.unravel_index(np.argmax(values),values.shape)
            markers[box[0].start+y,box[1].start+xcoord]=int(markers.max())+1
    labels=segmentation.watershed(-distance,markers,mask=fg,connectivity=1,watershed_line=False)
    # Preserve marker-derived identities; only split disconnected supports explicitly, then sequential IDs.
    out=np.zeros(labels.shape,np.int32);next_id=0
    for ident,box in enumerate(ndi.find_objects(labels),1):
        if box is None:continue
        pieces,n=ndi.label(labels[box]==ident,structure=np.ones((3,3)))
        for j in range(1,n+1):
            support=pieces==j
            if int(support.sum())<settings['minimum_area']:continue
            next_id+=1;view=out[box];view[support]=next_id
    return out,dict(version=VERSION,contrast_p99_p1=contrast,foreground_fraction=float(np.mean(out>0)),predicted_instances=next_id)
