from __future__ import annotations
import math
import numpy as np
from scipy import ndimage as ndi


def boundary(mask):
    m=mask.astype(bool)
    if not m.any(): return np.zeros_like(m)
    return m ^ ndi.binary_erosion(m)

def surface_distances(a,b):
    ba,bb=boundary(a),boundary(b)
    if not ba.any() or not bb.any(): return None,None
    db=ndi.distance_transform_edt(~bb); da=ndi.distance_transform_edt(~ba)
    return db[ba], da[bb]

def per_case_metrics(pred,gt):
    pred=pred.astype(bool); gt=gt.astype(bool); h,w=gt.shape; penalty=math.sqrt(h*h+w*w)
    inter=np.logical_and(pred,gt).sum(); ps=pred.sum(); gs=gt.sum(); union=np.logical_or(pred,gt).sum()
    dice=(2*inter)/(ps+gs) if (ps+gs)>0 else 1.0; iou=inter/union if union>0 else 1.0
    failure=False
    if not pred.any() and gt.any():
        failure=True; hd=hd95=asd=penalty
    elif pred.any() and not gt.any():
        failure=True; hd=hd95=asd=penalty
    elif not pred.any() and not gt.any(): hd=hd95=asd=0.0
    else:
        d1,d2=surface_distances(pred,gt); all_d=np.concatenate([d1,d2]); hd=float(all_d.max()); hd95=float(np.percentile(all_d,95)); asd=float((d1.sum()+d2.sum())/(len(d1)+len(d2)))
    return {'dice':float(dice),'iou':float(iou),'hd':hd,'hd95':hd95,'asd':asd,'empty_pred_failure':failure}
