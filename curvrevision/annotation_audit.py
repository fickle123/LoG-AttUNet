from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
from .data import discover_hc18_cases, annotation_points, ellipse_mask_from_annotation
from .geometry import load_hc_metadata, fit_physical_ellipse, ellipse_circumference_mm, point_to_ellipse_distances_mm
from .utils import dump_json
from .metrics import per_case_metrics


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',default='../fetal_dataset'); ap.add_argument('--output',default='outputs/annotation_audit'); args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    cases=discover_hc18_cases(args.data_root); meta=load_hc_metadata(args.data_root)
    rows=[]
    for c in cases:
        if c.case_id not in meta: raise KeyError(f'metadata missing {c.case_id}')
        m=meta[c.case_id]; pts=annotation_points(c.annotation_path); ellipse_mask,info=ellipse_mask_from_annotation(c.annotation_path,return_info=True)
        ann=cv2.imread(c.annotation_path,cv2.IMREAD_GRAYSCALE)>0
        raw_filled=ndi.binary_fill_holes(ann)
        enclosed_area=int(raw_filled.sum()-ann.sum())
        raw_closed=bool(enclosed_area>0)
        shape_metrics=per_case_metrics(ellipse_mask,raw_filled.astype(np.uint8)) if raw_closed else {'dice':np.nan,'iou':np.nan,'hd':np.nan,'hd95':np.nan,'asd':np.nan}
        e=fit_physical_ellipse(pts,m['spacing_x_mm'],m['spacing_y_mm'])
        if e is None:
            hc=np.nan; d=np.array([],float); physical_fit='failed'
        else:
            hc=ellipse_circumference_mm(e); d=point_to_ellipse_distances_mm(pts,m['spacing_x_mm'],m['spacing_y_mm'],e); physical_fit='ellipse'
        rows.append({
            'case_id':c.case_id,'n_annotation_pixels':len(pts),'mask_generation_method':info['fit_method'],'mask_fit_error':info.get('fit_error'),
            'touches_border':info['touches_border'],'raw_annotation_closed':raw_closed,'raw_enclosed_area_px':enclosed_area,'ellipse_vs_rawfill_dice':shape_metrics['dice'],'ellipse_vs_rawfill_hd_px':shape_metrics['hd'],'ellipse_vs_rawfill_hd95_px':shape_metrics['hd95'],'ellipse_vs_rawfill_asd_px':shape_metrics['asd'],'physical_fit_method':physical_fit,'pixel_x_mm':m['spacing_x_mm'],'pixel_y_mm':m['spacing_y_mm'],
            'official_hc_mm':m['official_hc_mm'],'ellipse_hc_mm':hc,'hc_error_mm':hc-m['official_hc_mm'] if np.isfinite(hc) else np.nan,
            'hc_abs_error_mm':abs(hc-m['official_hc_mm']) if np.isfinite(hc) else np.nan,
            'point_fit_mean_mm':float(d.mean()) if len(d) else np.nan,'point_fit_p95_mm':float(np.percentile(d,95)) if len(d) else np.nan,'point_fit_max_mm':float(d.max()) if len(d) else np.nan,
        })
    df=pd.DataFrame(rows); df.to_csv(out/'annotation_audit_per_case.csv',index=False)
    failed=int((df.physical_fit_method=='failed').sum()); fallback=int((df.mask_generation_method=='convex_hull').sum()); border=int(df.touches_border.sum())
    summary={
        'n_cases':len(df),'physical_ellipse_fit_failures':failed,'mask_generation_convex_hull_fallbacks':fallback,'border_touch_cases':border,
        'ellipse_vs_official_hc_mae_mm':float(df.hc_abs_error_mm.mean()),'ellipse_vs_official_hc_bias_mm':float(df.hc_error_mm.mean()),
        'ellipse_vs_official_hc_rmse_mm':float(np.sqrt(np.mean(df.hc_error_mm.dropna().to_numpy()**2))),
        'annotation_point_to_ellipse_mean_mm':float(df.point_fit_mean_mm.mean()),'annotation_point_to_ellipse_p95_case_mean_mm':float(df.point_fit_p95_mm.mean()),
        'raw_annotation_closed_cases':int(df.raw_annotation_closed.sum()),'ellipse_vs_rawfill_mean_dice_on_closed_cases':float(df.loc[df.raw_annotation_closed,'ellipse_vs_rawfill_dice'].mean()),
        'ellipse_vs_rawfill_mean_hd95_px_on_closed_cases':float(df.loc[df.raw_annotation_closed,'ellipse_vs_rawfill_hd95_px'].mean()),
        'note':'All failures/fallbacks are explicitly counted; no silent fallback is omitted.'
    }
    dump_json(summary,out/'annotation_audit_summary.json')
    df[df.touches_border].to_csv(out/'border_touch_cases.csv',index=False)
    df[df.mask_generation_method!='ellipse'].to_csv(out/'mask_fallback_cases.csv',index=False)
    print(pd.Series(summary).to_string())

if __name__=='__main__': main()
