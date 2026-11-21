"""RBC instance proposals v0.2.0-dev. Image-only, classical, no trained weights."""
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, morphology, segmentation, filters, measure
from rbcseg.adapter import normalize, predict as old_predict

VERSION='0.2.0-dev'

def clean(labels, minimum, erosion=0):
    out=np.zeros(labels.shape,np.int32);ident=0
    for k,box in enumerate(ndi.find_objects(labels),1):
        if box is None:continue
        mask=labels[box]==k
        if erosion:mask=ndi.binary_erosion(np.pad(mask,erosion),structure=morphology.disk(erosion))[erosion:-erosion,erosion:-erosion]
        cc,n=ndi.label(mask,structure=np.ones((3,3)))
        for j in range(1,n+1):
            m=cc==j
            if m.sum()<minimum:continue
            ident+=1;out[box][m]=ident
    return out

def predict(image,s):
    x,contrast=normalize(image)
    if s['family']=='baseline':return old_predict(image,s['baseline'])[0]
    if s['family']=='distance':
        base=s['baseline'].copy();base.update(minimum_area=s['minimum_area'],peak_distance=s['peak_distance'])
        return clean(old_predict(image,base)[0],s['minimum_area'],s['shrink'])
    if s['family'] not in ('interiors','seeded'):raise ValueError('Unknown family')
    if not contrast:return np.zeros(x.shape,np.int32)
    edge=feature.canny(x,sigma=s['sigma'],low_threshold=s['low'],high_threshold=s['high'],use_quantiles=False)
    if s['close']:edge=ndi.binary_closing(edge,structure=morphology.disk(s['close']))
    cc,_=ndi.label(~edge) # 4-connectivity complementary to diagonal rim
    border=set(np.unique(np.concatenate((cc[0],cc[-1],cc[:,0],cc[:,-1]))))
    labels=np.zeros(x.shape,np.int32);n=0
    for p in measure.regionprops(cc):
        if p.label in border or not s['minimum_area']<=p.area<=s['maximum_area'] or p.solidity<s['solidity']:continue
        n+=1;labels[p.slice][p.image]=n
    if s['family']=='interiors':
        if s['expand']:labels=segmentation.expand_labels(labels,distance=s['expand'])
        return clean(labels,s['minimum_area'],s['shrink'])
    fg=ndi.binary_fill_holes(edge)
    markers=np.zeros(x.shape,np.int32)
    for p in measure.regionprops(labels):
        # One marker at maximum interior distance; deterministic raster tie-break.
        d=ndi.distance_transform_edt(np.pad(p.image,1))[1:-1,1:-1]
        y,z=np.unravel_index(d.argmax(),d.shape);markers[p.bbox[0]+y,p.bbox[1]+z]=p.label
    elevation=filters.sobel(x) if s['surface']=='gradient' else -ndi.distance_transform_edt(fg)
    result=segmentation.watershed(elevation,markers,mask=fg,connectivity=1)
    return clean(result,s['minimum_area'],s['shrink'])

def features(image,labels):
    x,contrast=normalize(image);props=measure.regionprops(labels);n=len(props)
    areas=np.array([p.area for p in props]);med=float(np.median(areas)) if n else 0
    frac=lambda values:float(np.mean(values)) if n else 1.
    return dict(n=n,occupancy=float(np.mean(labels>0)),median_area=med,
        small_fraction=frac(areas<1000),large_fraction=frac(areas>4000),
        size_tail_fraction=frac((areas<.5*med)|(areas>1.8*med)),
        irregular_fraction=frac([p.solidity<.90 or p.axis_major_length/max(p.axis_minor_length,1e-12)>1.7 for p in props]),
        border_fraction=frac([p.bbox[0]==0 or p.bbox[1]==0 or p.bbox[2]==x.shape[0] or p.bbox[3]==x.shape[1] for p in props]),
        contrast=contrast)
