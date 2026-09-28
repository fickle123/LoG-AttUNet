from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi


def soft_dice_loss(logits,y,eps=1e-6):
    p=torch.sigmoid(logits); inter=(p*y).sum((1,2,3)); den=p.sum((1,2,3))+y.sum((1,2,3)); return (1-(2*inter+eps)/(den+eps)).mean()


def sdf_l1(pred,target): return F.l1_loss(pred,target)


def _distance_to_boundary(mask: np.ndarray) -> np.ndarray:
    m=mask.astype(bool)
    if not m.any(): return np.full(mask.shape, math.sqrt(mask.shape[0]**2+mask.shape[1]**2),np.float32)
    er=ndi.binary_erosion(m); b=m^er
    d=ndi.distance_transform_edt(~b).astype(np.float32)
    return d


def hd_dt_loss(logits,y):
    p=torch.sigmoid(logits); hard=(p.detach().cpu().numpy()>0.5); gt=(y.detach().cpu().numpy()>0.5)
    maps=[]; diag=math.sqrt(y.shape[-2]**2+y.shape[-1]**2)
    for i in range(y.shape[0]):
        dgt=_distance_to_boundary(gt[i,0])/diag; dp=_distance_to_boundary(hard[i,0])/diag; maps.append(dgt*dgt+dp*dp)
    w=torch.from_numpy(np.stack(maps)[:,None]).to(device=y.device,dtype=y.dtype)
    return (((p-y)**2)*w).mean()


def hd_lambda(epoch_one_based:int):
    if epoch_one_based<=5: return 0.0
    if epoch_one_based<=10: return 0.10*((epoch_one_based-5)/5.0)
    return 0.10


def stage1_loss(out,y,sdf_target,epoch:int,multi_task:bool=True):
    bce=F.binary_cross_entropy_with_logits(out['logits'],y); dice=soft_dice_loss(out['logits'],y)
    if not multi_task: return 0.55*bce+0.45*dice, {'bce':float(bce.detach()),'dice_loss':float(dice.detach())}
    h=hd_dt_loss(out['logits'],y); sdf=sdf_l1(out['sdf'],sdf_target); a2=F.binary_cross_entropy_with_logits(out['aux2'],y); a3=F.binary_cross_entropy_with_logits(out['aux3'],y)
    lam=hd_lambda(epoch); total=.45*bce+.40*dice+lam*h+.05*sdf+.20*a2+.10*a3
    return total, {'bce':float(bce.detach()),'dice_loss':float(dice.detach()),'hd_dt':float(h.detach()),'hd_lambda':lam,'sdf':float(sdf.detach()),'aux2':float(a2.detach()),'aux3':float(a3.detach())}


def stage2_loss(out,y,sdf_target,multi_task:bool=True):
    bce=F.binary_cross_entropy_with_logits(out['logits'],y); dice=soft_dice_loss(out['logits'],y)
    if not multi_task: return .55*bce+.45*dice, {'bce':float(bce.detach()),'dice_loss':float(dice.detach())}
    sdf=sdf_l1(out['sdf'],sdf_target); a2=F.binary_cross_entropy_with_logits(out['aux2'],y); total=.50*bce+.45*dice+.03*sdf+.02*a2
    return total, {'bce':float(bce.detach()),'dice_loss':float(dice.detach()),'sdf':float(sdf.detach()),'aux2':float(a2.detach())}
