from __future__ import annotations
from pathlib import Path
import math
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .data import robust_metadata_table, find_metadata_columns


def normalize_name(x: str) -> str:
    s=Path(str(x).strip()).name
    for ext in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'):
        if s.lower().endswith(ext): s=s[:-len(ext)]; break
    if s.endswith('_Annotation'): s=s[:-11]
    return s


def load_hc_metadata(data_root: str|Path) -> dict[str,dict]:
    df=robust_metadata_table(data_root); cols=find_metadata_columns(df)
    if cols['name'] is None or cols['hc'] is None:
        raise ValueError(f"Could not identify filename/official HC columns in {list(df.columns)}")
    result={}
    for _,r in df.iterrows():
        cid=normalize_name(r[cols['name']])
        def val(col):
            if col is None: return None
            try:
                z=float(r[col]); return z if np.isfinite(z) else None
            except Exception: return None
        pixel=val(cols['pixel']); sx=val(cols['sx']); sy=val(cols['sy'])
        if sx is None: sx=pixel
        if sy is None: sy=pixel
        if sx is None or sy is None:
            raise ValueError(f"No pixel spacing for {cid}; columns={cols}")
        hc=val(cols['hc'])
        if hc is None: raise ValueError(f"No official HC for {cid}")
        result[cid]={"spacing_x_mm":sx,"spacing_y_mm":sy,"official_hc_mm":hc}
    return result


def largest_component_fill(mask: np.ndarray) -> np.ndarray:
    m=mask.astype(bool)
    if not m.any(): return np.zeros_like(m,dtype=np.uint8)
    lab,n=ndi.label(m)
    counts=np.bincount(lab.ravel()); counts[0]=0
    keep=int(np.argmax(counts)); out=(lab==keep)
    out=ndi.binary_fill_holes(out)
    return out.astype(np.uint8)


def outer_contour(mask: np.ndarray) -> np.ndarray|None:
    m=(mask>0).astype(np.uint8)
    contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours: return None
    c=max(contours,key=cv2.contourArea).reshape(-1,2).astype(np.float32)
    return c if len(c)>=5 else None


def ellipse_circumference_mm(ellipse) -> float:
    # cv2 axes are full diameters in the same units as input points.
    a=float(ellipse[1][0])/2.0; b=float(ellipse[1][1])/2.0
    if a<=0 or b<=0: return float('nan')
    h=((a-b)**2)/((a+b)**2 + 1e-12)
    return math.pi*(a+b)*(1 + (3*h)/(10+math.sqrt(max(4-3*h,1e-12))))


def fit_physical_ellipse(points_xy_px: np.ndarray, sx_mm: float, sy_mm: float):
    pts=np.asarray(points_xy_px,np.float32).copy()
    if len(pts)<5: return None
    pts[:,0]*=float(sx_mm); pts[:,1]*=float(sy_mm)
    try:
        e=cv2.fitEllipse(pts.reshape(-1,1,2))
        return e
    except cv2.error:
        return None


def ellipse_sample_points(ellipse, n=1440) -> np.ndarray:
    (cx,cy),(d1,d2),angle=ellipse
    a=d1/2.0; b=d2/2.0; th=np.linspace(0,2*np.pi,n,endpoint=False)
    x=a*np.cos(th); y=b*np.sin(th)
    r=np.deg2rad(angle); cr,sr=np.cos(r),np.sin(r)
    xr=x*cr-y*sr+cx; yr=x*sr+y*cr+cy
    return np.column_stack([xr,yr]).astype(np.float32)


def point_to_ellipse_distances_mm(points_xy_px,sx_mm,sy_mm,ellipse):
    pts=np.asarray(points_xy_px,np.float32).copy(); pts[:,0]*=sx_mm; pts[:,1]*=sy_mm
    sample=ellipse_sample_points(ellipse)
    d,_=cKDTree(sample).query(pts,k=1)
    return np.asarray(d,float)
