from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data import discover_hc18_cases
from .geometry import load_hc_metadata, largest_component_fill, outer_contour, fit_physical_ellipse, ellipse_circumference_mm
from .utils import dump_json


def hc_metrics(g):
    valid=g[np.isfinite(g.pred_hc_mm)].copy(); fail=int((~np.isfinite(g.pred_hc_mm)).sum())
    if len(valid)==0:
        return {'n_cases':len(g),'fit_failures':fail,'valid_cases':0,'mae_mm':np.nan,'rmse_mm':np.nan,'bias_mm':np.nan,'mape_percent':np.nan,'loa_low_mm':np.nan,'loa_high_mm':np.nan}
    d=valid.pred_hc_mm-valid.official_hc_mm; bias=float(d.mean()); sd=float(d.std(ddof=1)) if len(d)>1 else 0.0
    return {'n_cases':len(g),'fit_failures':fail,'valid_cases':len(valid),'mae_mm':float(np.abs(d).mean()),'rmse_mm':float(np.sqrt(np.mean(d*d))),
            'bias_mm':bias,'mape_percent':float((np.abs(d)/valid.official_hc_mm).mean()*100.0),'loa_low_mm':bias-1.96*sd,'loa_high_mm':bias+1.96*sd}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metrics',default='outputs/aggregate/all_per_image_metrics.csv'); ap.add_argument('--data-root',default='../fetal_dataset'); ap.add_argument('--output',default='outputs/hc_analysis')
    ap.add_argument('--experiment',default='core'); ap.add_argument('--budget',default='original48'); ap.add_argument('--selection-rule',default='best_dice'); ap.add_argument('--models',nargs='+',default=['full_log_shallow']); args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    mdf=pd.read_csv(args.metrics); mdf=mdf[(mdf.experiment==args.experiment)&(mdf.budget==args.budget)&(mdf.selection_rule==args.selection_rule)&(mdf.model.isin(args.models))].copy()
    if len(mdf)==0: raise RuntimeError('No matching completed outer-test metrics. Run collect first and check experiment/budget/model.')
    cases={c.case_id:c for c in discover_hc18_cases(args.data_root)}; meta=load_hc_metadata(args.data_root)
    rows=[]
    # one prediction per case per seed after combining 5 outer folds
    for (model,seed),g in mdf.groupby(['model','seed']):
        if g.case_id.nunique()!=999 or g.duplicated('case_id').any():
            raise RuntimeError(f'{model} seed {seed} is not complete 5-fold OOF (unique={g.case_id.nunique()}, duplicates={g.duplicated("case_id").sum()})')
        for _,r in g.iterrows():
            cid=str(r.case_id); c=cases[cid]; md=meta[cid]; pred_path=Path(r.run_dir)/'predictions_256'/f'{cid}.png'
            pred=cv2.imread(str(pred_path),cv2.IMREAD_GRAYSCALE)
            if pred is None: raise FileNotFoundError(pred_path)
            pred=(pred>127).astype(np.uint8)
            pred=cv2.resize(pred,(c.width,c.height),interpolation=cv2.INTER_NEAREST)
            proc=largest_component_fill(pred); contour=outer_contour(proc)
            fit_status='ok'; hc=np.nan
            if contour is None: fit_status='empty_or_too_short'
            else:
                e=fit_physical_ellipse(contour,md['spacing_x_mm'],md['spacing_y_mm'])
                if e is None: fit_status='ellipse_fit_failed'
                else: hc=ellipse_circumference_mm(e)
            rows.append({'model':model,'seed':int(seed),'case_id':cid,'fold':int(r.fold),'pred_hc_mm':hc,'official_hc_mm':md['official_hc_mm'],
                         'error_mm':hc-md['official_hc_mm'] if np.isfinite(hc) else np.nan,'fit_status':fit_status,'prediction_path':str(pred_path),
                         'hc_postprocess':'largest component + hole filling; original-resolution nearest-neighbor restoration'})
    per=pd.DataFrame(rows); per.to_csv(out/'hc_per_case_per_seed.csv',index=False)
    summaries=[]
    for (model,seed),g in per.groupby(['model','seed']): summaries.append({'model':model,'seed':seed,**hc_metrics(g)})
    pd.DataFrame(summaries).to_csv(out/'hc_summary_by_seed.csv',index=False)
    # Dataset-level inference: average seed-specific HC per image before statistics; seeds are not independent images.
    agg=per.groupby(['model','case_id','official_hc_mm'],as_index=False).agg(pred_hc_mm_mean_successful=('pred_hc_mm','mean'),n_successful_seeds=('pred_hc_mm','count'),n_seeds=('seed','nunique'))
    agg['all_seeds_success']=agg.n_successful_seeds.eq(agg.n_seeds)
    # Do not silently average around a failed seed-specific ellipse fit. A dataset-level case is valid only if every formal seed succeeded.
    agg['pred_hc_mm']=agg.pred_hc_mm_mean_successful.where(agg.all_seeds_success,np.nan)
    agg['error_mm']=agg.pred_hc_mm-agg.official_hc_mm; agg.to_csv(out/'hc_per_case_seed_averaged.csv',index=False)
    aggregate_summaries=[]
    for model,g in agg.groupby('model'):
        sm={'model':model,**hc_metrics(g),'cases_with_any_seed_fit_failure':int((~g.all_seeds_success).sum())}; aggregate_summaries.append(sm); dump_json(sm,out/f'{model}_hc_summary.json')
        valid=g[np.isfinite(g.pred_hc_mm)].copy(); mean=(valid.pred_hc_mm+valid.official_hc_mm)/2; diff=valid.pred_hc_mm-valid.official_hc_mm
        bias=sm['bias_mm']; lo=sm['loa_low_mm']; hi=sm['loa_high_mm']
        fig,ax=plt.subplots(figsize=(7.2,5.2)); ax.scatter(mean,diff,s=12,alpha=.55); ax.axhline(bias,linestyle='--',linewidth=1.2); ax.axhline(lo,linestyle=':',linewidth=1.2); ax.axhline(hi,linestyle=':',linewidth=1.2)
        ax.set_xlabel('Mean of predicted and reference HC (mm)'); ax.set_ylabel('Predicted - reference HC (mm)'); ax.set_title(f'Bland-Altman: {model}')
        fig.tight_layout(); fig.savefig(out/f'bland_altman_{model}.png',dpi=300); fig.savefig(out/f'bland_altman_{model}.pdf'); plt.close(fig)
    pd.DataFrame(aggregate_summaries).to_csv(out/'hc_summary_seed_averaged.csv',index=False)
    print(pd.DataFrame(aggregate_summaries).to_string(index=False))

if __name__=='__main__': main()
