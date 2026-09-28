from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata


def bootstrap_ci(x, n=10000, seed=2026):
    x=np.asarray(x,float); rng=np.random.default_rng(seed); N=len(x)
    vals=np.empty(n,float)
    for i in range(n): vals[i]=x[rng.integers(0,N,N)].mean()
    return np.percentile(vals,[2.5,97.5])


def signflip_p(x,n=10000,seed=2026):
    x=np.asarray(x,float); obs=abs(x.mean()); rng=np.random.default_rng(seed); count=1
    # Monte-Carlo paired sign-flip permutation test.
    for _ in range(n):
        stat=abs((x*rng.choice([-1.0,1.0],size=len(x))).mean())
        count += stat >= obs - 1e-15
    return count/(n+1)


def rank_biserial_improvement(x):
    x=np.asarray(x,float); nz=x!=0
    if not nz.any(): return 0.0
    x=x[nz]; r=rankdata(np.abs(x)); den=r.sum()
    return float((r[x>0].sum()-r[x<0].sum())/den)


def holm_adjust(p):
    p=np.asarray(p,float); m=len(p); order=np.argsort(p); out=np.empty(m,float); running=0.0
    for j,idx in enumerate(order):
        val=(m-j)*p[idx]; running=max(running,val); out[idx]=min(running,1.0)
    return out


def complete_group(g): return g.case_id.nunique()==999 and not g.duplicated('case_id').any() and g.fold.nunique()==5


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='outputs/aggregate/all_per_image_metrics.csv'); ap.add_argument('--output',default='outputs/statistics')
    ap.add_argument('--experiment',default='core'); ap.add_argument('--budget',default='stage1'); ap.add_argument('--selection-rule',default='best_dice'); ap.add_argument('--ours',default='full_log_attention_only')
    ap.add_argument('--baselines',nargs='+',default=['full_standard','attention_resunet','resunet']); ap.add_argument('--bootstrap',type=int,default=10000); ap.add_argument('--permutations',type=int,default=10000); args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(args.input)
    df=df[(df.experiment==args.experiment)&(df.budget==args.budget)&(df.selection_rule==args.selection_rule)].copy()
    # Average the three training seeds at the IMAGE level. Seeds are not treated as independent patients/images.
    seed_counts=df.groupby(['model','seed']).case_id.nunique().reset_index()
    complete_models=[]
    for model,g in df.groupby('model'):
        ok=True
        for seed,gs in g.groupby('seed'):
            if not complete_group(gs): ok=False
        if ok: complete_models.append(model)
    df=df[df.model.isin(complete_models)]
    image=df.groupby(['model','case_id'],as_index=False)[['dice','iou','hd','hd95','asd']].mean()
    # Fold-level variability after averaging seeds per image.
    case_fold=df.groupby(['model','fold','case_id'],as_index=False)[['dice','iou','hd','hd95','asd']].mean()
    fold_means=case_fold.groupby(['model','fold'],as_index=False)[['dice','iou','hd','hd95','asd']].mean()
    fold_long=fold_means.melt(id_vars=['model','fold'],var_name='metric',value_name='fold_mean')
    fold_summary=fold_long.groupby(['model','metric']).fold_mean.agg(['mean','std','count']).reset_index()
    fold_summary['fold_sem']=fold_summary['std']/np.sqrt(fold_summary['count'])
    # t(4)=2.776 for five outer folds; calculate dynamically when SciPy is available.
    from scipy.stats import t as student_t
    crit=fold_summary['count'].apply(lambda n: student_t.ppf(.975,n-1) if n>1 else np.nan)
    fold_summary['fold95_low']=fold_summary['mean']-crit*fold_summary['fold_sem']; fold_summary['fold95_high']=fold_summary['mean']+crit*fold_summary['fold_sem']
    fold_means.to_csv(out/f'fold_means_{args.experiment}_{args.budget}_{args.selection_rule}_{args.ours}.csv',index=False)
    fold_summary.to_csv(out/f'fold_variability_{args.experiment}_{args.budget}_{args.selection_rule}_{args.ours}.csv',index=False)
    metrics=['dice','iou','hd','hd95','asd']; rows=[]
    for baseline in args.baselines:
        for metric in metrics:
            a=image[image.model==args.ours][['case_id',metric]].rename(columns={metric:'ours'})
            b=image[image.model==baseline][['case_id',metric]].rename(columns={metric:'base'})
            m=a.merge(b,on='case_id',how='inner')
            if len(m)!=999: raise RuntimeError(f'{args.ours} vs {baseline} {metric}: expected 999 paired images, got {len(m)}')
            raw=(m.ours-m.base).to_numpy()
            improvement=raw if metric in ('dice','iou') else -raw
            ci=bootstrap_ci(raw,args.bootstrap,2026)
            try: wp=float(wilcoxon(raw,alternative='two-sided',zero_method='wilcox').pvalue)
            except ValueError: wp=1.0
            rows.append({'ours':args.ours,'baseline':baseline,'metric':metric,'n_pairs':len(raw),'mean_ours':m.ours.mean(),'mean_baseline':m.base.mean(),
                         'mean_delta_ours_minus_baseline':raw.mean(),'median_delta_ours_minus_baseline':np.median(raw),'bootstrap95_low':ci[0],'bootstrap95_high':ci[1],
                         'mean_improvement_positive_is_better':improvement.mean(),'wilcoxon_p':wp,'signflip_p':signflip_p(raw,args.permutations,2026),
                         'rank_biserial_improvement':rank_biserial_improvement(improvement)})
    res=pd.DataFrame(rows)
    for metric in metrics:
        idx=res.metric==metric
        res.loc[idx,'wilcoxon_p_holm']=holm_adjust(res.loc[idx,'wilcoxon_p'].to_numpy())
        res.loc[idx,'signflip_p_holm']=holm_adjust(res.loc[idx,'signflip_p'].to_numpy())
    # Also expose a conservative global family correction across every
    # baseline×metric hypothesis in this table.  The per-metric correction is
    # retained because it is often the most interpretable family in the paper.
    res['wilcoxon_p_holm_global']=holm_adjust(res['wilcoxon_p'].to_numpy())
    res['signflip_p_holm_global']=holm_adjust(res['signflip_p'].to_numpy())
    res.to_csv(out/f'paired_{args.experiment}_{args.budget}_{args.selection_rule}_{args.ours}.csv',index=False)
    image.to_csv(out/f'image_level_seed_averaged_{args.experiment}_{args.budget}_{args.selection_rule}_{args.ours}.csv',index=False)
    # OOF 95% bootstrap CI for each model metric (image-level seed average)
    cis=[]
    for (model,metric),g in image.melt(id_vars=['model','case_id'],var_name='metric',value_name='value').groupby(['model','metric']):
        ci=bootstrap_ci(g.value.to_numpy(),args.bootstrap,4040); cis.append({'model':model,'metric':metric,'mean':g.value.mean(),'median':g.value.median(),'bootstrap95_low':ci[0],'bootstrap95_high':ci[1],'n_images':len(g)})
    pd.DataFrame(cis).to_csv(out/f'bootstrap_ci_{args.experiment}_{args.budget}_{args.selection_rule}_{args.ours}.csv',index=False)
    print(res.to_string(index=False))

if __name__=='__main__': main()
